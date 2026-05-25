"""Backend dependency wiring.

Mirrors ``agent.run._build_deps`` but lifted up so FastAPI's lifespan can
build it once at startup and stash it on ``app.state``. Each request reuses
the same wired ``AgentDeps`` + Qdrant client; sequential VRAM is enforced
inside ``retrieve_and_rerank`` and the synthesize node (which calls
``embedder.unload()`` / ``reranker.unload()`` before Ollama generates).

The CLI in ``agent.run`` and this module share the same ``LazyEmbedder`` /
``LazyReranker`` pattern; keep them in sync.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# ─── lazy model wrappers (kept here so tests can import + stub) ──────────────

class _LazySession:
    """Holds a ``retrieve.vram.ModelSession`` open across the process lifetime.

    Opening is deferred until first use so /healthz never triggers a model
    load. ``unload()`` releases the session — the synthesize node calls this
    via ``LazyEmbedder.unload()`` / ``LazyReranker.unload()`` so the LLM has
    VRAM to work with.
    """

    def __init__(self, factory: Callable[[], Any], name: str) -> None:
        self._factory = factory
        self._name = name
        self._session = None
        self._instance = None

    def ensure_loaded(self):
        if self._session is None:
            from retrieve.vram import ModelSession
            self._session = ModelSession(self._factory, name=self._name)
            self._instance = self._session.__enter__()
        return self._instance

    def unload(self) -> None:
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
            self._instance = None


class LazyEmbedder:
    def __init__(self, lazy: _LazySession) -> None:
        self._lazy = lazy

    def embed(self, texts):
        return self._lazy.ensure_loaded().embed(texts)

    def unload(self) -> None:
        self._lazy.unload()


class LazyReranker:
    def __init__(self, lazy: _LazySession) -> None:
        self._lazy = lazy

    def score(self, query, passages):
        return self._lazy.ensure_loaded().score(query, passages)

    def unload(self) -> None:
        self._lazy.unload()


# ─── service container ──────────────────────────────────────────────────────

@dataclass
class BackendDeps:
    """Everything an endpoint needs. Built at lifespan startup, frozen after."""
    agent_deps: Any       # agent.nodes.AgentDeps
    checkpointer: Any     # langgraph checkpointer (Memory or Postgres)
    collection: str
    # Health pings. Stored as callables so tests can inject deterministic stubs.
    ping_qdrant: Callable[[], None]
    ping_neo4j: Callable[[], None]
    ping_ollama: Callable[[], None]


def build_default_deps() -> BackendDeps:
    """Wire production deps: real Qdrant, real Neo4j, lazy BGE-M3, lazy
    reranker, Ollama LLM, and a Postgres checkpointer if ``POSTGRES_DSN`` is
    set (else MemorySaver — fine for dev).
    """
    from qdrant_client import QdrantClient

    from agent.checkpoint import make_memory_saver, make_postgres_saver
    from agent.llm import OllamaLLM
    from agent.nodes import AgentDeps
    from embed.qdrant import QdrantConfig
    from graph.client import make_driver

    cfg = QdrantConfig.from_env()
    qdrant = QdrantClient(host=cfg.host, port=cfg.port)

    def embedder_factory():
        from embed.bge_m3 import BGE_M3Embedder
        return BGE_M3Embedder()

    def reranker_factory():
        from retrieve.reranker import BGE_RerankerV2M3
        return BGE_RerankerV2M3()

    embedder = LazyEmbedder(_LazySession(embedder_factory, "bge-m3"))
    reranker = LazyReranker(_LazySession(reranker_factory, "bge-reranker-v2-m3"))
    neo4j = make_driver()
    llm = OllamaLLM()

    agent_deps = AgentDeps(
        embedder=embedder, reranker=reranker,
        qdrant=qdrant, neo4j=neo4j, llm=llm,
        collection=cfg.collection,
    )
    checkpointer = (
        make_postgres_saver() if os.environ.get("POSTGRES_DSN") else make_memory_saver()
    )

    def _ping_qdrant():
        qdrant.get_collections()

    def _ping_neo4j():
        with neo4j.session() as s:
            list(s.run("RETURN 1 AS ok"))

    def _ping_ollama():
        # ollama.Client has no .ping(); a cheap call is .list().
        from ollama import Client  # type: ignore
        Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434")).list()

    return BackendDeps(
        agent_deps=agent_deps,
        checkpointer=checkpointer,
        collection=cfg.collection,
        ping_qdrant=_ping_qdrant,
        ping_neo4j=_ping_neo4j,
        ping_ollama=_ping_ollama,
    )
