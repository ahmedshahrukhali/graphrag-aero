"""Sync httpx client for the FastAPI backend.

Mirrors ``frontend/lib/api.ts`` and ``backend/schemas.py`` 1:1. Gradio
handlers are easier to read sync and the latency budget per call is
dominated by the backend, not by Python threading.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import quote

import httpx


DEFAULT_BACKEND_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 60.0


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


# ─── response dataclasses (mirror backend/schemas.py) ────────────────────────

@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    doc_id: str
    source_url: str | None
    section_title: str
    page: int
    bbox: tuple[float, float, float, float]
    lang: str
    text: str
    ann_score: float
    rerank_score: float | None
    # WS-0: region-level grounding (one (page, x0, top, x1, bottom) per page the
    # chunk touches), corpus tag, and figure discriminator. Defaulted so a
    # response from the pre-re-ingest index still parses.
    page_bboxes: tuple[tuple[float, ...], ...] = ()
    corpus: str | None = None
    kind: str = "text"

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievedChunk":
        return cls(
            rank=d["rank"],
            doc_id=d["doc_id"],
            source_url=d.get("source_url"),
            section_title=d.get("section_title", ""),
            page=d["page"],
            bbox=tuple(d["bbox"]),  # type: ignore[arg-type]
            lang=d["lang"],
            text=d["text"],
            ann_score=float(d["ann_score"]),
            rerank_score=None if d.get("rerank_score") is None else float(d["rerank_score"]),
            page_bboxes=tuple(tuple(float(v) for v in pb) for pb in d.get("page_bboxes", ())),
            corpus=d.get("corpus"),
            kind=d.get("kind", "text"),
        )


@dataclass(frozen=True)
class RetrieveResponse:
    query: str
    results: list[RetrievedChunk]

    @classmethod
    def from_dict(cls, d: dict) -> "RetrieveResponse":
        return cls(
            query=d["query"],
            results=[RetrievedChunk.from_dict(c) for c in d.get("results", [])],
        )


@dataclass(frozen=True)
class QueryPausedResponse:
    thread_id: str
    draft: str | None
    trace: list[dict]
    n_candidates: int

    @classmethod
    def from_dict(cls, d: dict) -> "QueryPausedResponse":
        return cls(
            thread_id=d["thread_id"],
            draft=d.get("draft"),
            trace=list(d.get("trace", [])),
            n_candidates=int(d.get("n_candidates", 0)),
        )


@dataclass(frozen=True)
class ResumeResponse:
    thread_id: str
    final: str | None
    trace: list[dict]
    history: list[dict]

    @classmethod
    def from_dict(cls, d: dict) -> "ResumeResponse":
        return cls(
            thread_id=d["thread_id"],
            final=d.get("final"),
            trace=list(d.get("trace", [])),
            history=list(d.get("history", [])),
        )


@dataclass(frozen=True)
class ComponentHealth:
    ok: bool
    detail: str | None = None


@dataclass(frozen=True)
class HealthResponse:
    ok: bool
    qdrant: ComponentHealth
    neo4j: ComponentHealth
    ollama: ComponentHealth

    @classmethod
    def from_dict(cls, d: dict) -> "HealthResponse":
        def comp(x: dict) -> ComponentHealth:
            return ComponentHealth(ok=bool(x.get("ok")), detail=x.get("detail"))
        return cls(
            ok=bool(d.get("ok")),
            qdrant=comp(d.get("qdrant", {})),
            neo4j=comp(d.get("neo4j", {})),
            ollama=comp(d.get("ollama", {})),
        )


# ─── client ──────────────────────────────────────────────────────────────────

@dataclass
class ApiClient:
    base_url: str = field(default_factory=lambda: os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL))
    timeout: float = DEFAULT_TIMEOUT
    client: httpx.Client | None = None

    def _http(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self.client

    def _request(self, method: str, path: str, json: dict | None = None) -> dict:
        r = self._http().request(method, path, json=json)
        if r.status_code // 100 != 2:
            detail: Any = None
            try:
                detail = r.json()
            except Exception:  # noqa: BLE001
                pass
            raise ApiError(r.status_code, f"{method} {path} → {r.status_code}", detail)
        return r.json()

    def retrieve(
        self,
        query: str,
        *,
        lang: list[str] | None = None,
        source: list[str] | None = None,
        ann_k: int = 50,
        top_k: int = 10,
    ) -> RetrieveResponse:
        body = {"query": query, "lang": lang, "source": source, "ann_k": ann_k, "top_k": top_k}
        return RetrieveResponse.from_dict(self._request("POST", "/retrieve", json=body))

    def query(
        self, query: str, thread_id: str, *, max_hops: int = 2,
        lang: str | None = None, source: str | None = None,
    ) -> QueryPausedResponse:
        body = {
            "query": query, "thread_id": thread_id, "max_hops": max_hops,
            "lang": lang, "source": source,
        }
        return QueryPausedResponse.from_dict(self._request("POST", "/query", json=body))

    def query_stream(
        self, query: str, thread_id: str, *, max_hops: int = 2,
        lang: str | None = None, source: str | None = None,
    ) -> Iterator[dict]:
        """POST /query/stream. Yields parsed SSE events as ``{event, data}``.

        Streams status events for retrieve/graph_expand, then token events
        carrying synthesize chunks, then a final ``done`` event with sources
        and trace. Caller drives partial UI updates from each yield.

        ``lang``/``source`` apply the sidebar Lang/Corpus filters to retrieval;
        ``None`` means "no filter" (the "all" choice).
        """
        body = {
            "query": query, "thread_id": thread_id, "max_hops": max_hops,
            "lang": lang, "source": source,
        }
        # Use a fresh stream-aware client; the default client may have a
        # short timeout that doesn't suit long generations.
        timeout = httpx.Timeout(self.timeout, read=None)
        with httpx.Client(base_url=self.base_url, timeout=timeout) as c:
            with c.stream("POST", "/query/stream", json=body) as r:
                if r.status_code // 100 != 2:
                    try:
                        detail = r.read().decode("utf-8", errors="replace")
                    except Exception:  # noqa: BLE001
                        detail = None
                    raise ApiError(r.status_code, f"POST /query/stream → {r.status_code}", detail)
                event: str | None = None
                data_buf: list[str] = []
                for raw in r.iter_lines():
                    if raw is None:
                        continue
                    if raw == "":
                        if event and data_buf:
                            try:
                                yield {"event": event, "data": json.loads("\n".join(data_buf))}
                            except json.JSONDecodeError:
                                pass
                        event, data_buf = None, []
                        continue
                    if raw.startswith("event:"):
                        event = raw[len("event:"):].strip()
                    elif raw.startswith("data:"):
                        data_buf.append(raw[len("data:"):].lstrip())

    def resume(self, thread_id: str, *, draft: str | None = None) -> ResumeResponse:
        body: dict = {} if draft is None else {"draft": draft}
        path = f"/resume/{quote(thread_id, safe='')}"
        return ResumeResponse.from_dict(self._request("POST", path, json=body))

    def graph_lookup(self, doc_id: str) -> dict:
        """GET /graph/{doc_id} — knowledge-graph context for a document."""
        return self._request("GET", f"/graph/{doc_id}")

    def healthz(self) -> HealthResponse:
        return HealthResponse.from_dict(self._request("GET", "/healthz"))


def make_client(base_url: str | None = None, *, transport: httpx.BaseTransport | None = None) -> ApiClient:
    """Factory. Tests pass an ``httpx.MockTransport`` to intercept calls."""
    base = base_url or os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL)
    if transport is None:
        return ApiClient(base_url=base)
    return ApiClient(base_url=base, client=httpx.Client(base_url=base, transport=transport, timeout=DEFAULT_TIMEOUT))
