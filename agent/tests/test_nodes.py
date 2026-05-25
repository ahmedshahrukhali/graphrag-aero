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

def _make_deps(qclient, llm=None, graph=None, reranker=None, embedder=None):
    return AgentDeps(
        embedder=embedder or StubEmbedder(0),
        reranker=reranker or StubReranker({}),
        qdrant=qclient,
        neo4j=graph or FakeGraphDriver({}),
        llm=llm or StubLLM(),
        collection=COLL,
        ann_k=10, top_k=5,
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
    state.update(node(state))  # hop 2 — same query, deduplicated
    assert state["hop"] == 2
    # No dupes despite two retrievals.
    hashes = [c["chunk_hash"] for c in state["candidates"]]
    assert len(hashes) == len(set(hashes))


# ─── graph_expand_node ───────────────────────────────────────────────────────

def test_graph_expand_pulls_occurrences():
    table = {"doc000": {"id": "doc000", "source_url": "u-0", "lang": "en"}}
    deps = _make_deps(QdrantClient(":memory:"), graph=FakeGraphDriver(table))
    node = make_graph_expand_node(deps)
    state = initial_state("q")
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a", idx=0, source="tsb"), 0.5, 0.5)),
    ]
    update = node(state)
    assert update["graph_context"] == [{"id": "doc000", "source_url": "u-0", "lang": "en"}]
    assert update["trace"][-1]["node"] == "graph_expand"


def test_graph_expand_handles_no_tsb_candidates():
    deps = _make_deps(QdrantClient(":memory:"), graph=FakeGraphDriver({}))
    node = make_graph_expand_node(deps)
    state = initial_state("q")
    state["candidates"] = [
        scored_chunk_to_dict(ScoredChunk(_rec("a", idx=0, source="tc"), 0.5, 0.5)),
    ]
    update = node(state)
    assert update["graph_context"] == []


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


def test_synthesize_node_calls_unload_on_dependencies(qclient):
    llm = StubLLM("draft text")
    embedder = UnloadTracker()
    reranker = UnloadTracker()
    deps = _make_deps(qclient, llm=llm, embedder=embedder, reranker=reranker)
    node = make_synthesize_node(deps)
    state = initial_state("what?")
    node(state)
    assert embedder.unloaded is True
    assert reranker.unloaded is True
