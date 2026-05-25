"""CLI: agent init-schema | upsert-graph | query | resume."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class LazySessionModel:
    def __init__(self, factory, name):
        self.factory = factory
        self.name = name
        self.session = None
        self.instance = None

    def ensure_loaded(self):
        if self.session is None:
            from retrieve.vram import ModelSession
            self.session = ModelSession(self.factory, name=self.name)
            self.instance = self.session.__enter__()
        return self.instance

    def unload(self):
        if self.session is not None:
            self.session.__exit__(None, None, None)
            self.session = None
            self.instance = None


class LazyEmbedder:
    def __init__(self, lazy_session):
        self.lazy_session = lazy_session

    def embed(self, texts):
        return self.lazy_session.ensure_loaded().embed(texts)

    def unload(self):
        self.lazy_session.unload()


class LazyReranker:
    def __init__(self, lazy_session):
        self.lazy_session = lazy_session

    def score(self, query, passages):
        return self.lazy_session.ensure_loaded().score(query, passages)

    def unload(self):
        self.lazy_session.unload()


def _build_deps(collection: str):
    """Wire production deps. Heavy imports happen here, not at module load."""
    from qdrant_client import QdrantClient

    from embed.qdrant import QdrantConfig
    from graph.client import make_driver

    from .llm import OllamaLLM
    from .nodes import AgentDeps

    cfg = QdrantConfig.from_env()
    qdrant = QdrantClient(host=cfg.host, port=cfg.port)

    # Factories for lazy loading
    def embedder_factory():
        from embed.bge_m3 import BGE_M3Embedder
        return BGE_M3Embedder()

    def reranker_factory():
        from retrieve.reranker import BGE_RerankerV2M3
        return BGE_RerankerV2M3()

    lazy_embed = LazySessionModel(embedder_factory, "bge-m3")
    lazy_rerank = LazySessionModel(reranker_factory, "bge-reranker-v2-m3")

    return AgentDeps(
        embedder=LazyEmbedder(lazy_embed),
        reranker=LazyReranker(lazy_rerank),
        qdrant=qdrant,
        neo4j=make_driver(),
        llm=OllamaLLM(),
        collection=collection or cfg.collection,
    )


def cmd_init_schema(_args) -> int:
    from graph.client import make_driver
    from graph.schema import init_schema

    d = make_driver()
    try:
        n = init_schema(d)
        print(f"applied {n} schema statements")
    finally:
        d.close()
    return 0


def cmd_upsert_graph(args) -> int:
    from graph.client import make_driver
    from graph.upsert import upsert_occurrences_from_chunks

    d = make_driver()
    try:
        n = upsert_occurrences_from_chunks(d, Path(args.in_root))
        print(f"upserted {n} Occurrence nodes")
    finally:
        d.close()
    return 0


def cmd_query(args) -> int:
    from .checkpoint import make_postgres_saver
    from .graph import build_graph
    from .state import initial_state

    deps = _build_deps(args.collection)
    with make_postgres_saver() as cp:
        cp.setup()
        graph = build_graph(deps, checkpointer=cp)
        config = {"configurable": {"thread_id": args.thread}}
        paused = graph.invoke(initial_state(args.query, max_hops=args.max_hops), config=config)
        print("DRAFT:")
        print(paused.get("draft") or "(no draft)")
        print("\nTRACE:")
        for t in paused.get("trace", []):
            print(f"  {t}")
        print(f"\nPaused at HITL. Resume with: agent.run resume {args.thread}")
    return 0


def cmd_resume(args) -> int:
    from .checkpoint import make_postgres_saver
    from .graph import build_graph
    from .nodes import AgentDeps
    from .trace import trace_from_history

    deps = _build_deps(args.collection)
    with make_postgres_saver() as cp:
        graph = build_graph(deps, checkpointer=cp)
        config = {"configurable": {"thread_id": args.thread}}
        if args.draft_file:
            new_draft = Path(args.draft_file).read_text(encoding="utf-8")
            graph.update_state(config, {"draft": new_draft})
        done = graph.invoke(None, config=config)
        print("FINAL:")
        print(done.get("final") or "(no answer)")
        print("\nHISTORY:")
        for step in trace_from_history(graph, config):
            print(f"  {step}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LangGraph agent CLI.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-schema")

    up = sub.add_parser("upsert-graph")
    up.add_argument("--in", dest="in_root", default="data/chunks")

    q = sub.add_parser("query")
    q.add_argument("query", help="Natural-language question.")
    q.add_argument("--thread", required=True)
    q.add_argument("--max-hops", type=int, default=2)
    q.add_argument("--collection", default=None)

    r = sub.add_parser("resume")
    r.add_argument("thread")
    r.add_argument("--draft-file", default=None)
    r.add_argument("--collection", default=None)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    table = {
        "init-schema": cmd_init_schema,
        "upsert-graph": cmd_upsert_graph,
        "query": cmd_query,
        "resume": cmd_resume,
    }
    return table[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
