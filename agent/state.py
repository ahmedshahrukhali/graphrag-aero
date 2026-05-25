"""LangGraph state for the multi-hop agent.

State must be JSON-serialisable end-to-end because the PostgresSaver writes
checkpoints as JSON. We use plain TypedDicts + lists/dicts of primitives;
``ScoredChunk`` from ``retrieve.reranker`` becomes a ``ScoredChunkDict``.
"""
from __future__ import annotations

from typing import TypedDict

from retrieve.reranker import ScoredChunk


class ScoredChunkDict(TypedDict):
    doc_id: str
    source_url: str | None
    section_title: str
    page: int
    bbox: list[float]
    chunk_hash: str
    lang: str
    text: str
    ann_score: float
    rerank_score: float | None


def scored_chunk_to_dict(sc: ScoredChunk) -> ScoredChunkDict:
    return ScoredChunkDict(
        doc_id=sc.record.doc_id,
        source_url=sc.record.source_url,
        section_title=sc.record.section_title,
        page=sc.record.page,
        bbox=list(sc.record.bbox),
        chunk_hash=sc.record.chunk_hash,
        lang=sc.record.lang,
        text=sc.record.text,
        ann_score=float(sc.ann_score),
        rerank_score=None if sc.rerank_score is None else float(sc.rerank_score),
    )


class AgentState(TypedDict, total=False):
    """All keys are TypedDict-optional (total=False) because LangGraph nodes
    return partial updates that get merged into state. Initial state must at
    minimum contain ``query``; the others get filled as the graph runs."""

    query: str
    hop: int
    max_hops: int
    candidates: list[ScoredChunkDict]
    graph_context: list[dict]
    draft: str | None
    final: str | None
    trace: list[dict]


def initial_state(query: str, *, max_hops: int = 2) -> AgentState:
    return AgentState(
        query=query,
        hop=0,
        max_hops=max_hops,
        candidates=[],
        graph_context=[],
        draft=None,
        final=None,
        trace=[],
    )
