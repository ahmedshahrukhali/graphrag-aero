"""Qdrant ANN search for the dense collection.

Hydrates raw Qdrant hits back into the same ``ChunkRecord`` shape that the
ingestion / embed steps emit, wrapped in :class:`ScoredChunk` with the raw
cosine similarity as ``ann_score`` (``rerank_score`` set later by P3's
pipeline).

Filters (lang, source) are pushed into Qdrant payload filters — Python-side
post-filtering would burn ANN candidate budget on rows we'll throw away.
"""
from __future__ import annotations

import logging
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from embed.jsonl import ChunkRecord
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
