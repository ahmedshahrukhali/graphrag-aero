"""Qdrant ANN search for the dense collection + sparse search + RRF fusion.

Hydrates raw Qdrant hits back into the same ``ChunkRecord`` shape that the
ingestion / embed steps emit, wrapped in :class:`ScoredChunk` with the raw
cosine similarity as ``ann_score`` (``rerank_score`` set later by P3's
pipeline).

Filters (lang, source) are pushed into Qdrant payload filters — Python-side
post-filtering would burn ANN candidate budget on rows we'll throw away.

Hybrid retrieval (§1):
  ``sparse_search`` queries the named "sparse" vector (BGE-M3 lexical weights).
  ``rrf_fuse`` merges dense + sparse result lists via Reciprocal Rank Fusion.
"""
from __future__ import annotations

import logging
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from embed.jsonl import ChunkRecord
from embed.qdrant import SPARSE_VECTOR_NAME
from .reranker import ScoredChunk


logger = logging.getLogger(__name__)


def _build_filter(*, lang: str | None, source: str | None) -> qm.Filter | None:
    must: list[qm.FieldCondition] = []
    if lang is not None:
        must.append(qm.FieldCondition(key="lang", match=qm.MatchValue(value=lang)))
    if source is not None:
        # ``doc_id`` is "{source}/{stem}" — filter by prefix on the source segment.
        # Storing source in payload separately would be cleaner but the existing
        # P1 schema doesn't; we encode the prefix as a MatchText (substring).
        must.append(qm.FieldCondition(
            key="doc_id", match=qm.MatchText(text=f"{source}/"),
        ))
    if not must:
        return None
    return qm.Filter(must=must)


def _hydrate(payload: dict) -> ChunkRecord:
    return ChunkRecord.from_dict(payload)


def dense_search(
    client: QdrantClient,
    collection: str,
    query_vector: Sequence[float],
    *,
    k: int = 50,
    lang: str | None = None,
    source: str | None = None,
) -> list[ScoredChunk]:
    """Top-``k`` ANN over ``collection`` with optional payload filters."""
    q_filter = _build_filter(lang=lang, source=source)
    resp = client.query_points(
        collection_name=collection,
        query=list(query_vector),
        query_filter=q_filter,
        limit=k,
        with_payload=True,
    )
    out: list[ScoredChunk] = []
    for pt in resp.points:
        if pt.payload is None:
            logger.warning("hit has no payload, skipping: id=%s", pt.id)
            continue
        out.append(ScoredChunk(
            record=_hydrate(pt.payload),
            ann_score=float(pt.score),
        ))
    return out


def sparse_search(
    client: QdrantClient,
    collection: str,
    sparse_weights: dict[int, float],
    *,
    k: int = 50,
    lang: str | None = None,
    source: str | None = None,
) -> list[ScoredChunk]:
    """Top-``k`` sparse ANN over the named "sparse" vector with optional filters.

    Requires the collection to have a sparse vector named ``SPARSE_VECTOR_NAME``
    (i.e. created with ``ensure_collection(..., with_sparse=True)``).
    Returns an empty list on a dense-only collection rather than raising.
    """
    if not sparse_weights:
        return []
    indices = sorted(sparse_weights.keys())
    values = [sparse_weights[i] for i in indices]
    q_vec = qm.SparseVector(indices=indices, values=values)
    q_filter = _build_filter(lang=lang, source=source)
    try:
        resp = client.query_points(
            collection_name=collection,
            query=q_vec,
            using=SPARSE_VECTOR_NAME,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
        )
    except Exception as exc:
        # Collection may not have a sparse vector — degrade gracefully.
        logger.warning("sparse_search failed (dense-only collection?): %s", exc)
        return []
    out: list[ScoredChunk] = []
    for pt in resp.points:
        if pt.payload is None:
            continue
        out.append(ScoredChunk(record=_hydrate(pt.payload), ann_score=float(pt.score)))
    return out


def rrf_fuse(
    dense_hits: list[ScoredChunk],
    sparse_hits: list[ScoredChunk],
    *,
    k: int = 60,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion of two ranked candidate lists.

    For each unique chunk (keyed by ``chunk_hash``), the RRF score is:
        rrf = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
    where rank is 1-indexed and a chunk absent from a list gets rank = len + 1.
    Returns the merged list sorted by descending RRF score with the fused score
    stored in ``ann_score``.
    """
    dense_rank: dict[str, int] = {
        c.record.chunk_hash: i + 1 for i, c in enumerate(dense_hits)
    }
    sparse_rank: dict[str, int] = {
        c.record.chunk_hash: i + 1 for i, c in enumerate(sparse_hits)
    }
    n_dense = len(dense_hits)
    n_sparse = len(sparse_hits)

    # Collect all unique chunks, preserving first-seen record instance.
    seen: dict[str, ScoredChunk] = {}
    for c in dense_hits + sparse_hits:
        if c.record.chunk_hash not in seen:
            seen[c.record.chunk_hash] = c

    fused: list[ScoredChunk] = []
    for chunk_hash, c in seen.items():
        rd = dense_rank.get(chunk_hash, n_dense + 1)
        rs = sparse_rank.get(chunk_hash, n_sparse + 1)
        score = 1.0 / (k + rd) + 1.0 / (k + rs)
        fused.append(ScoredChunk(record=c.record, ann_score=score))

    fused.sort(key=lambda x: x.ann_score, reverse=True)
    return fused


def scroll_doc_chunks(
    client: QdrantClient,
    collection: str,
    doc_ids: Sequence[str],
    *,
    page_size: int = 256,
) -> list[ScoredChunk]:
    """Fetch every chunk belonging to ``doc_ids`` (no ANN, no scoring).

    Used by anchored retrieval: once ANN+rerank identifies the most relevant
    documents, we pull their full chunk set so the reranker can surface the
    content pages (findings/analysis), not just the keyword-rich title page
    that floats to the top of a chunk-level search.

    ``ann_score`` is set to 1.0 — these chunks were selected by document
    membership, not similarity; the caller reranks them next.
    """
    if not doc_ids:
        return []
    out: list[ScoredChunk] = []
    for doc_id in dict.fromkeys(doc_ids):  # preserve order, drop dups
        flt = qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        ])
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                if pt.payload is None:
                    continue
                out.append(ScoredChunk(record=_hydrate(pt.payload), ann_score=1.0))
            if offset is None:
                break
    return out
