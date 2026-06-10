"""End-to-end test of the compiled LangGraph agent (§3: no HITL interrupt).

Uses MemorySaver (no Postgres) + stubbed deps (no real models, no real Qdrant
collection beyond the in-memory one).  Since interrupt_before was removed in
§3, a single graph.invoke() runs retrieve→graph_expand→synthesize→finalize
and returns the completed state with both draft and final set.
"""
import hashlib

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("langgraph")

from qdrant_client import QdrantClient

from embed.jsonl import ChunkRecord
from embed.qdrant import DENSE_DIM, ensure_collection, upsert_batch
from retrieve.reranker import ScoredChunk

from agent.checkpoint import make_memory_saver
from agent.graph import build_graph
from agent.nodes import AgentDeps
from agent.state import initial_state, scored_chunk_to_dict


COLL = "test_graph"


def _unit_vec(d: int) -> list[float]:
    v = [0.0] * DENSE_DIM
    v[d] = 1.0
    return v


def _rec(text: str, *, idx: int) -> ChunkRecord:
    h = hashlib.sha256(f"tsb/{idx}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=f"tsb/doc{idx:03d}", source_url=f"u-{idx}",
        section_title="", page=idx + 1, bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=h, lang="en", text=text,
    )


class StubEmbedder:
    def embed(self, texts): return [_unit_vec(0) for _ in texts]


class StubReranker:
    def __init__(self, scores): self._scores = scores
    def score(self, q, ps): return [self._scores.get(p, 0.5) for p in ps]


class StubLLM:
    def __init__(self, reply="DRAFT"): self.reply = reply
    def chat(self, s, u, history=None): return self.reply


class FakeGraphSession:
    def __init__(self, table): self.table = table
    def __enter__(self): return self
    def __exit__(self, *e): return None
    def run(self, cypher, **params):
        return iter([self.table[i] for i in params["ids"] if i in self.table])


class FakeGraphDriver:
    def __init__(self, table): self._t = table
    def session(self, **kw): return FakeGraphSession(self._t)
    def close(self): pass


@pytest.fixture
def qclient():
    c = QdrantClient(":memory:")
    ensure_collection(c, COLL)
    # Seed two candidates; second has a higher score so synthesize triggers.
    recs = [_rec("alpha", idx=0), _rec("beta", idx=1)]
    upsert_batch(c, COLL, recs, [_unit_vec(0), _unit_vec(0)])
    return c


def _make_deps(qclient, *, llm=None, reranker_scores=None):
    return AgentDeps(
        embedder=StubEmbedder(),
        reranker=StubReranker(reranker_scores or {"alpha": 0.95, "beta": 0.6}),
        qdrant=qclient,
        neo4j=FakeGraphDriver({"doc000": {
            "occ_id": "doc000", "occ_url": "u-0",
            "findings": [], "recommendations": [], "direct_regs": [], "acs": [],
        }}),
        llm=llm or StubLLM("DRAFT ANSWER"),
        collection=COLL,
        ann_k=10, top_k=5,
        confidence_threshold=0.5,
    )


# ─── end-to-end: single invoke completes fully (§3: no HITL) ─────────────────

def test_runs_to_completion_in_one_invoke(qclient):
    """§3: no interrupt_before — a single invoke returns draft AND final."""
    cp = make_memory_saver()
    graph = build_graph(_make_deps(qclient), checkpointer=cp)
    config = {"configurable": {"thread_id": "t1"}}

    done = graph.invoke(initial_state("q"), config=config)
    # Draft carries the LLM answer plus the deterministic Sources block.
    assert done.get("draft").startswith("DRAFT ANSWER")
    assert "**Sources:**" in done.get("draft")
    assert done.get("final") == done.get("draft")
    # Graph is at END — no pending nodes.
    assert tuple(graph.get_state(config).next) == ()


# ─── multi-hop ───────────────────────────────────────────────────────────────

def test_low_confidence_triggers_another_retrieve_hop(qclient):
    """All scores < 0.5 → graph loops back to retrieve until max_hops."""
    cp = make_memory_saver()
    deps = _make_deps(qclient, reranker_scores={"alpha": 0.1, "beta": 0.2})
    graph = build_graph(deps, checkpointer=cp)
    config = {"configurable": {"thread_id": "t3"}}

    paused = graph.invoke(initial_state("q", max_hops=2), config=config)
    # max_hops=2 → retrieve runs at least twice.
    assert paused["hop"] >= 2


# ─── trace ───────────────────────────────────────────────────────────────────

def test_in_band_trace_records_each_node(qclient):
    """§3: single invoke → all four nodes appear in trace."""
    cp = make_memory_saver()
    graph = build_graph(_make_deps(qclient), checkpointer=cp)
    config = {"configurable": {"thread_id": "t4"}}
    done = graph.invoke(initial_state("q"), config=config)
    nodes_seen = [t["node"] for t in done["trace"]]
    assert "retrieve" in nodes_seen
    assert "graph_expand" in nodes_seen
    assert "synthesize" in nodes_seen
    assert "finalize" in nodes_seen
