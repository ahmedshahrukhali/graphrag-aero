"""End-to-end retrieve pipeline: query → BGE-M3 embed → Qdrant ANN → rerank.

Callers (the LangGraph agent in P4, the FastAPI route in P6, the eval harness
in P5) call :func:`retrieve_and_rerank` directly. They're responsible for
providing the loaded models — interactive callers keep them resident; batch
callers wrap the call in :class:`ModelSession` from ``retrieve.vram``.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient

from embed.bge_m3 import DenseEmbedder

from .reranker import CrossEncoderReranker, ScoredChunk, rerank
from .search import dense_search


logger = logging.getLogger(__name__)


DEFAULT_ANN_K = 50
DEFAULT_TOP_K = 10


def retrieve_and_rerank(
    query: str,
    *,
    embedder: DenseEmbedder,
    reranker: CrossEncoderReranker,
    client: QdrantClient,
    collection: str,
    ann_k: int = DEFAULT_ANN_K,
    top_k: int = DEFAULT_TOP_K,
    lang: str | None = None,
    source: str | None = None,
) -> list[ScoredChunk]:
    """Embed ``query``, ANN-search the dense collection, rerank, return top-K.

    ``ann_k`` is how many candidates the reranker sees; ``top_k`` is how many
    we return after reranking. ``ann_k >= top_k`` (caller's responsibility).
    """
    if ann_k < top_k:
        raise ValueError(f"ann_k ({ann_k}) must be ≥ top_k ({top_k})")
    if not query.strip():
        return []

    [q_vec] = embedder.embed([query])
    logger.debug("ANN search: k=%d lang=%s source=%s", ann_k, lang, source)
    candidates = dense_search(
        client, collection, q_vec, k=ann_k, lang=lang, source=source,
    )
    logger.debug("ANN returned %d candidates", len(candidates))
    if not candidates:
        return []
    return rerank(query, candidates, reranker, top_k=top_k)
