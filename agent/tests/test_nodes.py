"""Tests for individual nodes — stubbed deps end-to-end."""
import hashlib

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient

from embed.jsonl import ChunkRecord
from embed.qdrant import DENSE_DIM, ensure_collection, upsert_batch
from retrieve.reranker import ScoredChunk

from agent.nodes import (
    AgentDeps,
    _merge_candidates,
    _occurrence_ids_from,
    finalize_node,
    make_decide_continue,
    make_graph_expand_node,
    make_retrieve_node,
    make_synthesize_node,
)
from agent.state import initial_state, scored_chunk_to_dict


COLL = "test_nodes"


def _unit_vec(direction: int) -> list[float]:
    v = [0.0] * DENSE_DIM
    v[direction] = 1.0
    return v


def _rec(text: str, *, idx: int = 0, source: str = "tsb") -> ChunkRecord:
    h = hashlib.sha256(f"{source}:{idx}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=f"{source}/doc{idx:03d}", source_url=f"u-{idx}",
        section_title="", page=idx + 1, bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=h, lang="en", text=text,
    )


# ─── stubs ───────────────────────────────────────────────────────────────────

class StubEmbedder:
    def __init__(self, axis: int = 0): self._axis = axis
    def embed(self, texts): return [_unit_vec(self._axis) for _ in texts]


class StubReranker:
    def __init__(self, scores: dict[str, float]): self._scores = scores
    def score(self, query, passages):
        return [self._scores.get(p, 0.0) for p in passages]


class StubLLM:
    def __init__(self, reply: str = "stub answer"):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
    def chat(self, system, user):
        self.calls.append((system, user))
        return self.reply


class FakeGraphSession:
    def __init__(self, table): self.table = table
    def __enter__(self): return self
    def __exit__(self, *e): return None
    def run(self, cypher, **params):
        return iter([self.table[i] for i in params["ids"] if i in self.table])


class FakeGraphDriver:
    def __init__(self, table: dict[str, dict]): self._table = table
    def session(self, **kw): return FakeGraphSession(self._table)
    def close(self): pass


# ─── helpers ─────────────────────────────────────────────────────────────────

def _make_deps(qclient, llm=None, graph=None, reranker=None, embedder=None,
               *, anchored=False, top_n_docs=3, char_budget=24_000):
    return AgentDeps(
        embedder=embedder or StubEmbedder(0),
        reranker=reranker or StubReranker({}),
        qdrant=qclient,
        neo4j=graph or FakeGraphDriver({}),
        llm=llm or StubLLM(),
        collection=COLL,
        ann_k=10, top_k=5,
        anchored=anchored, top_n_docs=top_n_docs, char_budget=char_budget,
    )


@pytest.fixture
def qclient() -> QdrantClient:
    c = QdrantClient(":memory:")
    ensure_collection(c, COLL)
    return c


# ─── _merge_candidates / _occurrence_ids_from ────────────────────────────────

def test_merge_candidates_dedupes_by_chunk_hash():
    a = scored_chunk_to_dict(ScoredChunk(_rec("a", idx=0), ann_score=0.5, rerank_score=0.3))
    b = scored_chunk_to_dict(ScoredChunk(_rec("b", idx=1), ann_score=0.6, rerank_score=0.7))
    a_dup = dict(a)  # same chunk_hash as a
    out = _merge_candidates([a], [b, a_dup], top_k=10)
    assert len(out) == 2
    assert {c["text"] for c in out} == {"a", "b"}


def test_merge_candidates_orders_by_rerank():
    a = scored_chunk_to_dict(ScoredChunk(_rec("a", idx=0), ann_score=0.5, rerank_score=0.1))
    b = scored_chunk_to_dict(ScoredChunk(_rec("b", idx=1), ann_score=0.5, rerank_score=0.9))
    out = _merge_candidates([a], [b], top_k=10)
    assert [c["text"] for c in out] == ["b", "a"]


def test_merge_candidates_truncates_to_top_k():
    cands = [
        scored_chunk_to_dict(ScoredChunk(_rec(t, idx=i), 0.5, float(i)))
        for i, t in enumerate(list("abcde"))
    ]
    out = _merge_candidates([], cands, top_k=2)
    assert len(out) == 2


def test_occurrence_ids_from_skips_tc():
    cands = [
        scored_chunk_to_dict(ScoredChunk(_rec("x", idx=0, source="tsb"), 0.5, 0.5)),
        scored_chunk_to_dict(ScoredChunk(_rec("y", idx=1, source="tc"), 0.5, 0.5)),
    ]
    ids = _occurrence_ids_from(cands)
    assert ids == ["doc000"]


# ─── retrieve_node ───────────────────────────────────────────────────────────

def test_retrieve_node_populates_candidates_and_increments_hop(qclient):
    rec = _rec("alpha", idx=0)
    upsert_batch(qclient, COLL, [rec], [_unit_vec(0)])
    deps = _make_deps(
        qclient,
        embedder=StubEmbedder(0),
        reranker=StubReranker({"alpha": 0.8}),
    )
    node = make_retrieve_node(deps)
    state = initial_state("q")
    update = node(state)
    assert update["hop"] == 1
    assert len(update["candidates"]) == 1
    assert update["candidates"][0]["text"] == "alpha"
    assert update["trace"][-1]["node"] == "retrieve"


def test_retrieve_node_merges_across_hops(qclient):
    r1, r2 = _rec("alpha", idx=0), _rec("beta", idx=1)
    upsert_batch(qclient, COLL, [r1, r2], [_unit_vec(0), _unit_vec(1)])
    deps = _make_deps(
        qclient, embedder=StubEmbedder(0),
        reranker=StubReranker({"alpha": 0.5, "beta": 0.4}),
    )
    node = make_retrieve_node(deps)
    state = initial_state("q")
    state.update(node(state))  # hop 1
    assert state["hop"] == 1
    state.update(node(state))  # hop 2 — reformulated query, deduplicated
    assert state["hop"] == 2
    # No dupes despite two retrievals.
    hashes = [c["chunk_hash"] for c in state["candidates"]]
    assert len(hashes) == len(set(hashes))


def test_retrieve_node_reformulates_query_on_hop_2(qclient):
    """§2: trace must record reformulated=True on hop 2."""
    rec = _rec("flapper valve froze fuel starvation", idx=0)
    upsert_batch(qclient, COLL, [rec], [_unit_vec(0)])
    deps = _make_deps(
        qclient, embedder=StubEmbedder(0),
        reranker=StubReranker({"flapper valve froze fuel starvation": 0.6}),
    )
    node = make_retrieve_node(deps)
    state = initial_state("fuel starvation")
    state.update(node(state))  # hop 1 — no reformulation (hop==0 at entry)
    assert state["trace"][-1].get("reformulated") is False
    assert state.get("reformulated_query") is None

    state.update(node(state))  # hop 2 — reformulation fires
    assert state["trace"][-1].get("reformulated") is True
    # reformulated_query should be set and contain the original + novel tokens
    ref_q = state.get("reformulated_query")
    assert ref_q is not None
    assert ref_q.startswith("fuel starvation")
    assert len(ref_q) > len("fuel starvation")


def test_retrieve_node_honours_excluded_hashes(qclient):
    """§3: chunks in excluded_chunk_hashes must not appear in candidates."""
    r1 = _rec("fuel exhaustion alpha", idx=0)
    r2 = _rec("fuel exhaustion beta", idx=1)
    upsert_batch(qclient, COLL, [r1, r2], [_unit_vec(0), _unit_vec(0)])
    deps = _make_deps(
        qclient, embedder=StubEmbedder(0),
        reranker=StubReranker({r1.text: 0.9, r2.text: 0.8}),
    )
    node = make_retrieve_node(deps)
    state = initial_state("fuel exhaustion", excluded_chunk_hashes=[r1.chunk_hash])
    update = node(state)
    returned_hashes = {c["chunk_hash"] for c in update["candidates"]}
    assert r1.chunk_hash not in returned_hashes, "excluded hash must not appear"
    assert r2.chunk_hash in returned_hashes, "non-excluded hash must still appear"


# ─── retrieve_node (anchored) ─────────────────────────────────────────────────

def test_agentdeps_anchored_default_on(monkeypatch):
    monkeypatch.delenv("RETRIEVE_ANCHORED", raising=False)
    deps = AgentDeps(
        embedder=StubEmbedder(0), reranker=StubReranker({}),
        qdrant=QdrantClient(":memory:"), neo4j=FakeGraphDriver({}),
        llm=StubLLM(), collection=COLL,
    )
    assert deps.anchored is True


def test_agentdeps_anchored_env_optout(monkeypatch):
    monkeypatch.setenv("RETRIEVE_ANCHORED", "0")
    deps = AgentDeps(
        embedder=StubEmbedder(0), reranker=StubReranker({}),
        qdrant=QdrantClient(":memory:"), neo4j=FakeGraphDriver({}),
        llm=StubLLM(), collection=COLL,
    )
    assert deps.anchored is False


def test_anchored_pulls_full_doc_not_just_title_page(qclient):
    # doc000 has a keyword-rich "title" chunk (high ANN+rerank) plus content
    # chunks the title-only search would never surface. Anchored mode must
    # return the content chunks too.
    title = _rec("fuel exhaustion forced landing report", idx=0)
    body1 = _rec("the engine quit because the tanks ran dry", idx=0)
    body2 = _rec("recommendation check fuel gauges before flight", idx=0)
    # same doc (idx=0 → tsb/doc000) but distinct chunk hashes via text
    for r in (title, body1, body2):
        upsert_batch(qclient, COLL, [r], [_unit_vec(0)])
    deps = _make_deps(
        qclient, anchored=True,
        reranker=StubReranker({
            title.text: 0.99, body1.text: 0.80, body2.text: 0.85,
        }),
    )
    node = make_retrieve_node(deps)
    update = node(initial_state("fuel exhaustion"))
    texts = {c["text"] for c in update["candidates"]}
    assert title.text in texts and body1.text in texts and body2.text in texts
    assert update["trace"][-1]["anchored"] is True


def test_anchored_respects_char_budget(qclient):
    big = "x" * 5000
    r1 = _rec(big + "1", idx=0)
    r2 = _rec(big + "2", idx=0)
    r3 = _rec(big + "3", idx=0)
    for r in (r1, r2, r3):
        upsert_batch(qclient, COLL, [r], [_unit_vec(0)])
    deps = _make_deps(
        qclient, anchored=True, char_budget=6000,
        reranker=StubReranker({r1.text: 0.9, r2.text: 0.8, r3.text: 0.7}),
    )
    node = make_retrieve_node(deps)
    update = node(initial_state("q"))
    # budget 6000, each chunk ~5001 chars → only the top-ranked one fits
    assert len(update["candidates"]) == 1
    assert update["candidates"][0]["text"] == r1.text


def _rec_at(doc: str, page: int, text: str) -> ChunkRecord:
    h = hashlib.sha256(f"{doc}:{page}:{text}".encode()).hexdigest()
    return ChunkRecord(doc_id=doc, source_url=None, section_title="",
                       page=page, bbox=[0, 0, 0, 0], chunk_hash=h, lang="en", text=text)


def test_anchored_returns_reading_order(qclient):
    # two docs; pages out of rerank order — result must sort by (doc_id, page)
    a_p3 = _rec_at("tsb/doc000", 3, "a3")
    a_p1 = _rec_at("tsb/doc000", 1, "a1")
    b_p2 = _rec_at("tsb/doc001", 2, "b2")
    for r in (a_p3, a_p1, b_p2):
        upsert_batch(qclient, COLL, [r], [_unit_vec(0)])
    deps = _make_deps(
        qclient, anchored=True,
        reranker=StubReranker({"a3": 0.9, "a1": 0.5, "b2": 0.7}),
    )
    node = make_retrieve_node(deps)
    update = node(initial_state("q"))
    assert [c["text"] for c in update["candidates"]] == ["a1", "a3", "b2"]


# ─── graph_expand_node ───────────────────────────────────────────────────────

def test_graph_expand_pulls_occurrences():
    rich_row = {
        "occ_id": "doc000", "occ_url": "u-0",
        "findings": [{"text": "Fuel tanks empty.", "category": "cause", "lang": "en",
                      "source_doc_id": "tsb/doc000", "page": 5, "cites_reg": None}],
        "recommendations": [], "direct_regs": [], "acs": [],
    }
    table = {"doc000": rich_row}
    deps = _make_deps(QdrantClient(":memory:"), graph=FakeGraphDriver(table))
    node = make_graph_expand_node(deps)
    state = initial_state("q")
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a", idx=0, source="tsb"), 0.5, 0.5)),
    ]
    update = node(state)
    assert len(update["graph_context"]) == 1
    assert update["graph_context"][0]["occ_id"] == "doc000"
    assert update["graph_context"][0]["findings"][0]["text"] == "Fuel tanks empty."
    # graph_expand now also runs the concentration-gated outward hop, so it
    # appends a second "graph_broaden" trace entry after "graph_expand".
    trace_nodes = [t["node"] for t in update["trace"]]
    assert "graph_expand" in trace_nodes
    broaden = update["trace"][-1]
    assert broaden["node"] == "graph_broaden"
    # 1 distinct doc ≤ threshold → the hop fires (the mock yields no siblings).
    assert broaden["fired"] is True and broaden["distinct_docs"] == 1
    assert update["recurring_context"] == []


def test_graph_expand_handles_no_tsb_candidates():
    deps = _make_deps(QdrantClient(":memory:"), graph=FakeGraphDriver({}))
    node = make_graph_expand_node(deps)
    state = initial_state("q")
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a", idx=0, source="tc"), 0.5, 0.5)),
    ]
    update = node(state)
    assert update["graph_context"] == []


def test_graph_expand_skips_broaden_when_already_broad():
    # 4 distinct docs > broaden_when_docs_lte default (3) → the outward hop must
    # NOT fire (retrieval is already broad; nothing to broaden).
    table = {
        f"doc{i:03d}": {"occ_id": f"doc{i:03d}", "occ_url": "u", "findings": [],
                        "recommendations": [], "direct_regs": [], "acs": []}
        for i in range(4)
    }
    deps = _make_deps(QdrantClient(":memory:"), graph=FakeGraphDriver(table))
    node = make_graph_expand_node(deps)
    state = initial_state("q")
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a", idx=i, source="tsb"), 0.5, 0.5))
        for i in range(4)
    ]
    update = node(state)
    broaden = update["trace"][-1]
    assert broaden["node"] == "graph_broaden"
    assert broaden["distinct_docs"] == 4 and broaden["fired"] is False
    assert update["recurring_context"] == []


# ─── decide_continue ─────────────────────────────────────────────────────────

def test_decide_continue_routes_to_retrieve_when_low_score_and_more_hops():
    deps = _make_deps(QdrantClient(":memory:"))
    decide = make_decide_continue(deps)
    state = initial_state("q", max_hops=2)
    state["hop"] = 1
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a"), ann_score=0.5, rerank_score=0.1)),
    ]
    assert decide(state) == "retrieve"


def test_decide_continue_routes_to_synthesize_on_high_score():
    deps = _make_deps(QdrantClient(":memory:"))
    decide = make_decide_continue(deps)
    state = initial_state("q", max_hops=2)
    state["hop"] = 1
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a"), ann_score=0.9, rerank_score=0.9)),
    ]
    assert decide(state) == "synthesize"


def test_decide_continue_routes_to_synthesize_when_max_hops_reached():
    deps = _make_deps(QdrantClient(":memory:"))
    decide = make_decide_continue(deps)
    state = initial_state("q", max_hops=2)
    state["hop"] = 2  # at the cap
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a"), ann_score=0.1, rerank_score=0.0)),
    ]
    assert decide(state) == "synthesize"


# ─── synthesize_node ─────────────────────────────────────────────────────────

def test_synthesize_calls_llm_with_prompt(qclient):
    llm = StubLLM("draft text")
    deps = _make_deps(qclient, llm=llm)
    node = make_synthesize_node(deps)
    state = initial_state("what?")
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("alpha", idx=0), 0.5, 0.5)),
    ]
    update = node(state)
    assert update["draft"] == "draft text"
    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert "aerospace" in system.lower()
    assert "what?" in user
    assert "alpha" in user
    assert update["trace"][-1]["node"] == "synthesize"


# ─── finalize_node ───────────────────────────────────────────────────────────

def test_finalize_copies_draft_to_final():
    state = initial_state("q")
    state["draft"] = "the answer is X"
    update = finalize_node(state)
    assert update["final"] == "the answer is X"
    assert update["trace"][-1]["node"] == "finalize"


def test_finalize_handles_none_draft():
    state = initial_state("q")
    update = finalize_node(state)
    assert update["final"] == ""


class UnloadTracker:
    def __init__(self):
        self.unloaded = False

    def unload(self):
        self.unloaded = True


def test_synthesize_node_calls_unload_when_sequential_vram_enabled(qclient, monkeypatch):
    monkeypatch.setenv("SEQUENTIAL_VRAM_UNLOAD", "1")
    monkeypatch.setattr("agent.nodes.wait_for_free_vram", lambda *a, **k: True)
    llm = StubLLM("draft text")
    embedder = UnloadTracker()
    reranker = UnloadTracker()
    deps = _make_deps(qclient, llm=llm, embedder=embedder, reranker=reranker)
    node = make_synthesize_node(deps)
    node(initial_state("what?"))
    assert embedder.unloaded is True
    assert reranker.unloaded is True


def test_synthesize_node_skips_unload_when_disabled(qclient, monkeypatch):
    # SEQUENTIAL_VRAM_UNLOAD=0 (big-GPU setups): retrieval stays resident.
    monkeypatch.setenv("SEQUENTIAL_VRAM_UNLOAD", "0")
    llm = StubLLM("draft text")
    embedder = UnloadTracker()
    reranker = UnloadTracker()
    deps = _make_deps(qclient, llm=llm, embedder=embedder, reranker=reranker)
    node = make_synthesize_node(deps)
    node(initial_state("what?"))
    assert embedder.unloaded is False
    assert reranker.unloaded is False
