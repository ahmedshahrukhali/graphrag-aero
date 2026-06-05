"""Shared fixtures: stub deps + in-memory Qdrant + TestClient factory.

No tests in this directory touch the network. Everything pluggable through
the create_app(deps_builder=...) factory.
"""
from __future__ import annotations

import hashlib
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("qdrant_client")

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from embed.jsonl import ChunkRecord
from embed.qdrant import DENSE_DIM, ensure_collection, upsert_batch


COLL = "test_backend"


def _unit_vec(direction: int) -> list[float]:
    v = [0.0] * DENSE_DIM
    v[direction] = 1.0
    return v


def _record(text: str, *, doc_id: str, lang: str = "en", idx: int = 0) -> ChunkRecord:
    h = hashlib.sha256(f"{doc_id}:{idx}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=doc_id,
        source_url=f"https://example.test/{doc_id}.pdf",
        section_title=f"§{idx}",
        page=idx + 1,
        bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=h,
        lang=lang,
        text=text,
    )


class StubEmbedder:
    def __init__(self, axis_for_query: dict[str, int] | None = None) -> None:
        self._axes = axis_for_query or {}
        self.unloaded = False

    def embed(self, texts):
        return [_unit_vec(self._axes.get(t, 0)) for t in texts]

    def unload(self) -> None:
        self.unloaded = True


class StubReranker:
    """Identity reranker — preserves ANN order (sort is stable)."""

    def __init__(self) -> None:
        self.unloaded = False

    def score(self, query, passages):
        return [1.0 for _ in passages]

    def unload(self) -> None:
        self.unloaded = True


class StubLLM:
    def __init__(self, reply: str = "stubbed draft answer.") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply

    def chat_stream(self, system: str, user: str):
        self.calls.append((system, user))
        yield self.reply


class StubFeedbackStore:
    """In-memory feedback store for tests — never returns prior matches."""

    def __init__(self) -> None:
        self.rejections: list[dict] = []
        self.resolved_ids: list[int] = []
        self._next_id = 1

    def write_rejection(self, query, query_embedding, answer, chunk_hashes, terms=None):
        row_id = self._next_id
        self._next_id += 1
        self.rejections.append(dict(
            id=row_id, query=query, answer=answer,
            chunk_hashes=list(chunk_hashes), terms=list(terms or []),
        ))
        return row_id

    def find_similar(self, query_embedding, *, threshold=None):
        return []

    def resolve(self, row_id: int) -> None:
        self.resolved_ids.append(row_id)


class StubNeo4jSession:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, *_a, **_kw): return iter([])


class StubNeo4jDriver:
    closed = False
    def session(self, **_kw): return StubNeo4jSession()
    def close(self) -> None: self.closed = True


@pytest.fixture
def qdrant_client():
    c = QdrantClient(":memory:")
    ensure_collection(c, COLL)
    return c


@pytest.fixture
def populated_qdrant(qdrant_client):
    records = [
        _record("fuel exhaustion narrative",  doc_id="tsb/doc-A", lang="en", idx=0),
        _record("flight controls failure",    doc_id="tsb/doc-B", lang="en", idx=1),
        _record("alimentation en carburant",  doc_id="tsb/doc-C", lang="fr", idx=2),
    ]
    upsert_batch(qdrant_client, COLL, records, [_unit_vec(0), _unit_vec(1), _unit_vec(2)])
    return qdrant_client


@pytest.fixture
def stub_deps(populated_qdrant):
    """A BackendDeps wired entirely with stubs.

    ``checkpointer`` is ``None`` here so /retrieve and /healthz tests don't
    need langgraph installed. Tests that exercise /query or /resume request
    the ``stub_deps_with_checkpointer`` fixture, which fills it in.
    """
    from agent.nodes import AgentDeps
    from backend.deps import BackendDeps

    embedder = StubEmbedder(axis_for_query={
        "fuel": 0, "controls": 1, "carburant": 2,
        # default 0 for anything else
    })
    reranker = StubReranker()
    llm = StubLLM()
    neo4j = StubNeo4jDriver()
    feedback_store = StubFeedbackStore()

    agent_deps = AgentDeps(
        embedder=embedder,
        reranker=reranker,
        qdrant=populated_qdrant,
        neo4j=neo4j,
        llm=llm,
        collection=COLL,
        ann_k=3, top_k=3,  # small corpus
    )

    pings = {"qdrant_ok": True, "neo4j_ok": True, "ollama_ok": True}

    def _pq():
        if not pings["qdrant_ok"]:
            raise RuntimeError("qdrant down")

    def _pn():
        if not pings["neo4j_ok"]:
            raise RuntimeError("neo4j down")

    def _po():
        if not pings["ollama_ok"]:
            raise RuntimeError("ollama down")

    deps = BackendDeps(
        agent_deps=agent_deps,
        checkpointer=None,
        collection=COLL,
        ping_qdrant=_pq, ping_neo4j=_pn, ping_ollama=_po,
        feedback_store=feedback_store,
    )
    # Hand back the knobs so individual tests can flip them.
    deps._pings = pings           # type: ignore[attr-defined]
    deps._stubs = {                # type: ignore[attr-defined]
        "embedder": embedder, "reranker": reranker, "llm": llm,
        "feedback_store": feedback_store,
    }
    return deps


@pytest.fixture
def stub_deps_with_checkpointer(stub_deps):
    """``stub_deps`` plus a real MemorySaver — required by the agent
    endpoints. Skips when langgraph isn't installed."""
    pytest.importorskip("langgraph")
    from agent.checkpoint import make_memory_saver
    stub_deps.checkpointer = make_memory_saver()
    return stub_deps


@pytest.fixture
def make_client(stub_deps, request):
    """Returns a callable that yields a started TestClient. Each call enters
    the context manager (so lifespan runs) and registers cleanup on the
    fixture's finalizer — old starlette doesn't run lifespan otherwise."""
    from backend.app import create_app

    started: list[TestClient] = []

    def _factory(*, otel_exporter: Any | None = None) -> TestClient:
        app = create_app(
            deps_builder=lambda: stub_deps,
            otel_exporter=otel_exporter,
            install_otel=otel_exporter is not None,
        )
        client = TestClient(app)
        client.__enter__()
        started.append(client)
        return client

    def _cleanup():
        for c in started:
            try:
                c.__exit__(None, None, None)
            except Exception:
                pass

    request.addfinalizer(_cleanup)
    return _factory
