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
from .search import dense_search, rrf_fuse, scroll_doc_chunks, sparse_search


logger = logging.getLogger(__name__)


DEFAULT_ANN_K = 50
DEFAULT_TOP_K = 10
DEFAULT_TOP_N_DOCS = 3
DEFAULT_CHAR_BUDGET = 24_000


def retrieve_and_rerank(
    query: str,
    *,
    embedder: DenseEmbedder,
    reranker: CrossEncoderReranker,
    client: QdrantClient,
    collection: str,
    ann_k: int = DEFAULT_ANN_K,
    top_k: int = DEFAULT_TOP_K,
    lang: list[str] | None = None,
    source: list[str] | None = None,
    exclude_hashes: list[str] | None = None,
) -> list[ScoredChunk]:
    """Embed ``query``, ANN-search the dense collection, rerank, return top-K.

    ``ann_k`` is how many candidates the reranker sees; ``top_k`` is how many
    we return after reranking. ``ann_k >= top_k`` (caller's responsibility).
    ``exclude_hashes`` skips specific chunks (used by the feedback loop to avoid
    chunks that produced a previously rejected answer).
    """
    if ann_k < top_k:
        raise ValueError(f"ann_k ({ann_k}) must be ≥ top_k ({top_k})")
    if not query.strip():
        return []

    [q_vec] = embedder.embed([query])
    logger.debug("ANN search: k=%d lang=%s source=%s", ann_k, lang, source)
    candidates = dense_search(
        client, collection, q_vec, k=ann_k, lang=lang, source=source,
        exclude_hashes=exclude_hashes,
    )
    logger.debug("ANN returned %d candidates", len(candidates))
    if not candidates:
        return []
    return rerank(query, candidates, reranker, top_k=top_k)


def hybrid_retrieve_and_rerank(
    query: str,
    *,
    embedder,
    reranker: CrossEncoderReranker,
    client: QdrantClient,
    collection: str,
    ann_k: int = DEFAULT_ANN_K,
    top_k: int = DEFAULT_TOP_K,
    lang: str | None = None,
    source: list[str] | None = None,
    exclude_hashes: list[str] | None = None,
    rrf_k: int = 60,
) -> list[ScoredChunk]:
    """Dense + sparse hybrid retrieval fused with RRF, then reranked.

    ``embedder`` must expose both ``embed(texts)`` and ``embed_sparse(texts)``
    (i.e. ``BGE_M3Embedder``). Falls back silently to dense-only if sparse
    search returns no results (e.g. dense-only collection).
    ``exclude_hashes`` skips specific chunks (used by the feedback loop).
    """
    if ann_k < top_k:
        raise ValueError(f"ann_k ({ann_k}) must be ≥ top_k ({top_k})")
    if not query.strip():
        return []

    [q_dense] = embedder.embed([query])
    [q_sparse] = embedder.embed_sparse([query])

    dense_hits = dense_search(
        client, collection, q_dense, k=ann_k, lang=lang, source=source,
        exclude_hashes=exclude_hashes,
    )
    sparse_hits = sparse_search(
        client, collection, q_sparse, k=ann_k, lang=lang, source=source,
        exclude_hashes=exclude_hashes,
    )

    if not dense_hits and not sparse_hits:
        return []
    if not sparse_hits:
        candidates = dense_hits
    else:
        candidates = rrf_fuse(dense_hits, sparse_hits, k=rrf_k)

    logger.debug(
        "hybrid: %d dense + %d sparse → %d fused",
        len(dense_hits), len(sparse_hits), len(candidates),
    )
    return rerank(query, candidates, reranker, top_k=top_k)


def _top_unique_docs(chunks: list[ScoredChunk], n: int) -> list[str]:
    docs: list[str] = []
    for c in chunks:
        d = c.record.doc_id
        if d not in docs:
            docs.append(d)
        if len(docs) >= n:
            break
    return docs


def anchored_retrieve(
    query: str,
    *,
    embedder: DenseEmbedder,
    reranker: CrossEncoderReranker,
    client: QdrantClient,
    collection: str,
    ann_k: int = DEFAULT_ANN_K,
    top_k: int = DEFAULT_TOP_K,
    top_n_docs: int = DEFAULT_TOP_N_DOCS,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    lang: list[str] | None = None,
    source: list[str] | None = None,
    exclude_hashes: list[str] | None = None,
) -> list[ScoredChunk]:
    """Document-anchored retrieval.

    Chunk-level ANN+rerank reliably identifies the *right documents* (their
    keyword-rich title pages float to the top) but returns those title pages,
    which carry no findings/analysis text — so the LLM has nothing to
    synthesize. Instead we use that signal only to pick the top documents,
    then pull every chunk from those docs, rerank the pool, greedily fill a
    character budget by relevance, and return the selection in reading order
    (doc_id, page) so the LLM sees coherent passages.
    """
    seed = retrieve_and_rerank(
        query, embedder=embedder, reranker=reranker, client=client,
        collection=collection, ann_k=ann_k, top_k=top_k, lang=lang, source=source,
        exclude_hashes=exclude_hashes,
    )
    if not seed:
        return []
    anchor_docs = _top_unique_docs(seed, top_n_docs)
    logger.debug("anchored to %d docs: %s", len(anchor_docs), anchor_docs)

    pool = scroll_doc_chunks(client, collection, anchor_docs)
    # §3: remove chunks explicitly excluded by the feedback loop.
    if exclude_hashes:
        excl = set(exclude_hashes)
        pool = [c for c in pool if c.record.chunk_hash not in excl]
    # doc_id is lang-agnostic by design, so the expansion pool contains
    # EN+FR chunks for any TSB doc whose stem appears in both corpora.
    # Without a lang filter, the synthesiser would receive a bilingual
    # mash. Fall back to the dominant lang of the top seed chunks so
    # "Any" means "stick with whichever language matched best", not "mix".
    effective_lang = lang
    if effective_lang is None:
        effective_lang = seed[0].record.lang
    if effective_lang is not None:
        pool = [c for c in pool if c.record.lang == effective_lang]
    if not pool:
        return seed
    ranked = rerank(query, pool, reranker)  # full pool, no top_k cap

    selected: list[ScoredChunk] = []
    used = 0
    for c in ranked:
        length = len(c.record.text)
        if selected and used + length > char_budget:
            break
        selected.append(c)
        used += length
    logger.debug("anchored selected %d chunks (~%d chars)", len(selected), used)

    selected.sort(key=lambda c: (c.record.doc_id, c.record.page or 0))
    return selected
