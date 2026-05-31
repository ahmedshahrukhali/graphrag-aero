"""FastAPI app: /retrieve, /query, /resume, /healthz.

The app wires a single ``BackendDeps`` container into ``app.state`` at
startup, and each handler pulls what it needs from there. We never construct
deps per-request — the lazy model sessions are process-scoped so they survive
across calls (and the agent's synthesize node unloads them before Ollama).

Single-worker by design: uvicorn must be run with ``--workers 1`` so VRAM
isn't contended by parallel inference. Document this in the README.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .deps import BackendDeps, build_default_deps
from .schemas import (
    ComponentHealth,
    HealthResponse,
    QueryPausedResponse,
    QueryRequest,
    ResumeRequest,
    ResumeResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)


# ─── lifespan ────────────────────────────────────────────────────────────────

def _deps_builder() -> Callable[[], BackendDeps]:
    """Indirection so tests can override what `lifespan` builds."""
    return build_default_deps


@asynccontextmanager
async def lifespan(app: FastAPI):
    builder = getattr(app.state, "deps_builder", None) or _deps_builder()
    deps: BackendDeps = builder()
    app.state.deps = deps
    logger.info("backend ready (collection=%s)", deps.collection)
    try:
        yield
    finally:
        # Release VRAM. Best-effort — process exit will clean up either way.
        try:
            deps.agent_deps.embedder.unload()
            deps.agent_deps.reranker.unload()
        except Exception:  # noqa: BLE001
            pass
        # Close the checkpointer's DB connection (Postgres) if one was opened.
        try:
            if deps.closer is not None:
                deps.closer()
        except Exception:  # noqa: BLE001
            pass


def create_app(*, deps_builder: Callable[[], BackendDeps] | None = None,
               otel_exporter: Any | None = None,
               install_otel: bool = True) -> FastAPI:
    """Factory. Tests inject ``deps_builder`` (stubs) and ``otel_exporter``
    (in-memory) and pass ``install_otel=False`` when they want to skip OTel
    entirely.
    """
    app = FastAPI(title="GraphRAG Aero Backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if deps_builder is not None:
        app.state.deps_builder = deps_builder

    if install_otel:
        from .otel import setup_tracing
        setup_tracing(app, exporter=otel_exporter)

    _register_routes(app)
    return app


# ─── routes ──────────────────────────────────────────────────────────────────

def _get_deps(request: Request) -> BackendDeps:
    deps: BackendDeps | None = getattr(request.app.state, "deps", None)
    if deps is None:
        raise HTTPException(status_code=503, detail="backend not initialised")
    return deps


def _register_routes(app: FastAPI) -> None:
    from retrieve.pipeline import retrieve_and_rerank

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(req: RetrieveRequest, request: Request) -> RetrieveResponse:
        if req.ann_k < req.top_k:
            raise HTTPException(400, "ann_k must be >= top_k")
        deps = _get_deps(request)
        ad = deps.agent_deps

        from .otel import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("query.length", len(req.query))
            span.set_attribute("lang", req.lang or "")
            results = retrieve_and_rerank(
                req.query,
                embedder=ad.embedder,
                reranker=ad.reranker,
                client=ad.qdrant,
                collection=deps.collection,
                ann_k=req.ann_k,
                top_k=req.top_k,
                lang=req.lang,
                source=req.source,
            )
            span.set_attribute("results.count", len(results))

        chunks = [
            RetrievedChunk(
                rank=i,
                doc_id=r.record.doc_id,
                source_url=r.record.source_url,
                section_title=r.record.section_title,
                page=r.record.page,
                bbox=list(r.record.bbox),
                page_bboxes=[list(pb) for pb in r.record.page_bboxes],
                corpus=r.record.corpus or None,
                kind=r.record.kind,
                lang=r.record.lang,
                text=r.record.text,
                ann_score=r.ann_score,
                rerank_score=r.rerank_score,
            )
            for i, r in enumerate(results, 1)
        ]
        return RetrieveResponse(query=req.query, results=chunks)

    @app.post("/query", response_model=QueryPausedResponse)
    def query(req: QueryRequest, request: Request) -> QueryPausedResponse:
        from agent.graph import build_graph
        from agent.state import initial_state

        deps = _get_deps(request)
        graph = build_graph(deps.agent_deps, checkpointer=deps.checkpointer)
        config = {"configurable": {"thread_id": req.thread_id}}

        from .otel import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.query") as span:
            span.set_attribute("thread_id", req.thread_id)
            span.set_attribute("max_hops", req.max_hops)
            paused = graph.invoke(
                initial_state(
                    req.query, max_hops=req.max_hops,
                    lang=req.lang, source=req.source,
                ),
                config=config,
            )
            span.set_attribute("draft.present", bool(paused.get("draft")))

        candidates = paused.get("candidates", [])
        return QueryPausedResponse(
            thread_id=req.thread_id,
            draft=paused.get("draft"),
            trace=paused.get("trace", []),
            n_candidates=len(candidates),
            sources=candidates,
        )

    @app.post("/query/stream")
    def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
        """Server-Sent Events variant of /query. Streams retrieve/graph status
        then token-level synthesize output. Skips the LangGraph HITL pause —
        clients that need draft-edit should use /query + /resume instead."""
        from agent.nodes import make_retrieve_node, make_graph_expand_node
        from agent.prompts import SYSTEM_PROMPT, build_user_prompt
        from agent.state import initial_state
        deps = _get_deps(request)
        ad = deps.agent_deps

        def _sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def gen():
            state = dict(initial_state(
                req.query, max_hops=req.max_hops,
                lang=req.lang, source=req.source,
            ))

            yield _sse("status", {"node": "retrieve", "msg": "Retrieving relevant chunks…"})
            state.update(make_retrieve_node(ad)(state))
            cands = state.get("candidates", [])
            best = max((c.get("rerank_score") or 0.0 for c in cands), default=0.0)
            yield _sse("status", {
                "node": "retrieve",
                "msg": f"Retrieved {len(cands)} chunks · best score {best:.2f}",
            })
            yield _sse("sources", {"sources": cands})

            yield _sse("status", {"node": "graph_expand", "msg": "Looking up graph context…"})
            state.update(make_graph_expand_node(ad)(state))
            yield _sse("status", {
                "node": "graph_expand",
                "msg": f"Graph context · {len(state.get('graph_context', []))} edges",
            })

            yield _sse("status", {"node": "synthesize", "msg": "Generating draft…"})
            # VRAM discipline matches synthesize_node
            if hasattr(ad.embedder, "unload"):
                ad.embedder.unload()
            if hasattr(ad.reranker, "unload"):
                ad.reranker.unload()
            prompt = build_user_prompt(
                state["query"], state.get("candidates", []), state.get("graph_context", []),
            )
            pieces: list[str] = []
            for chunk in ad.llm.chat_stream(SYSTEM_PROMPT, prompt):
                pieces.append(chunk)
                yield _sse("token", {"text": chunk})
            draft = "".join(pieces)

            trace = list(state.get("trace", []))
            trace.append({
                "node": "synthesize", "elapsed_ms": 0, "streamed": True,
                "prompt_chars": len(prompt), "draft_chars": len(draft),
            })
            yield _sse("done", {
                "thread_id": req.thread_id,
                "draft": draft,
                "trace": trace,
            })

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/resume/{thread_id}", response_model=ResumeResponse)
    def resume(thread_id: str, body: ResumeRequest, request: Request) -> ResumeResponse:
        from agent.graph import build_graph
        from agent.trace import trace_from_history

        deps = _get_deps(request)
        graph = build_graph(deps.agent_deps, checkpointer=deps.checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        from .otel import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span("agent.resume") as span:
            span.set_attribute("thread_id", thread_id)
            span.set_attribute("edited_draft", body.draft is not None)
            if body.draft is not None:
                graph.update_state(config, {"draft": body.draft})
            done = graph.invoke(None, config=config)

        return ResumeResponse(
            thread_id=thread_id,
            final=done.get("final"),
            trace=done.get("trace", []),
            history=trace_from_history(graph, config),
        )

    @app.get("/graph/{doc_id:path}")
    def graph_lookup(doc_id: str, request: Request) -> dict:
        """Return the knowledge-graph context for a single occurrence / AC document.

        Accepts both the full chunk doc_id (``tsb/a13q0098``) and the bare
        Neo4j occurrence id (``a13q0098``). Strips the corpus prefix before
        querying — Neo4j stores bare ids.
        """
        from graph.query import graph_context_for_occurrences
        deps = _get_deps(request)
        bare_id = doc_id.split("/", 1)[-1] if "/" in doc_id else doc_id
        rows = graph_context_for_occurrences(deps.agent_deps.neo4j, [bare_id])
        if not rows:
            raise HTTPException(status_code=404, detail=f"No graph data for {doc_id!r}")
        return rows[0]

    @app.get("/healthz", response_model=HealthResponse)
    def healthz(request: Request) -> HealthResponse:
        deps = _get_deps(request)

        def check(fn: Callable[[], None]) -> ComponentHealth:
            try:
                fn()
                return ComponentHealth(ok=True)
            except Exception as e:  # noqa: BLE001
                return ComponentHealth(ok=False, detail=str(e)[:200])

        q = check(deps.ping_qdrant)
        n = check(deps.ping_neo4j)
        o = check(deps.ping_ollama)
        return HealthResponse(ok=q.ok and n.ok and o.ok, qdrant=q, neo4j=n, ollama=o)


# Module-level app for ``uvicorn backend.app:app``.
app = create_app(install_otel=bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")))
