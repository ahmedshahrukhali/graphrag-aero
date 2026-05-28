"""HTTP request / response schemas for the backend.

These mirror what the retrieve pipeline and agent already return — kept
intentionally close to those internal shapes so the API isn't a translation
layer with its own bugs to maintain.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─── /retrieve ───────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    lang: Literal["en", "fr"] | None = None
    source: Literal["tsb", "tc"] | None = None
    ann_k: int = Field(50, ge=1, le=500)
    top_k: int = Field(10, ge=1, le=100)


class RetrievedChunk(BaseModel):
    rank: int
    doc_id: str
    source_url: str | None
    section_title: str
    page: int
    bbox: list[float]
    lang: str
    text: str
    ann_score: float
    rerank_score: float | None


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


# ─── /query  +  /resume ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    max_hops: int = Field(2, ge=1, le=5)


class QueryPausedResponse(BaseModel):
    """Returned after the agent runs up to the HITL gate.

    The caller inspects ``draft`` + ``trace``, optionally PUTs an edited draft
    via ``/resume/{thread_id}``, and gets the final answer back. ``sources``
    is the set of chunks the synthesizer was given — surface these in the x-ray
    so what the user sees matches what produced the draft.
    """
    thread_id: str
    draft: str | None
    trace: list[dict]
    n_candidates: int
    sources: list[dict]


class ResumeRequest(BaseModel):
    # Optional human-edited draft. If absent, finalize the model's draft as-is.
    draft: str | None = None


class ResumeResponse(BaseModel):
    thread_id: str
    final: str | None
    trace: list[dict]
    history: list[dict]


# ─── /healthz ────────────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    qdrant: ComponentHealth
    neo4j: ComponentHealth
    ollama: ComponentHealth
