"""Thin Qdrant wrapper: connect, ensure collection, upsert batches.

We use ``qdrant-client`` directly; nothing fancy. The wrapper exists so the CLI
and tests share one definition of "what our collection looks like" (dim,
distance, collection name) instead of duplicating it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .ids import point_id_for
from .jsonl import ChunkRecord


logger = logging.getLogger(__name__)


DENSE_DIM = 1024
DENSE_DISTANCE = qm.Distance.COSINE


@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    collection: str

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        return cls(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            collection=os.environ.get("QDRANT_COLLECTION_DENSE", "aerospace_dense"),
        )


def make_client(cfg: QdrantConfig) -> QdrantClient:
    """Real network client. Tests pass ``QdrantClient(':memory:')`` instead."""
    return QdrantClient(host=cfg.host, port=cfg.port)


def ensure_collection(client: QdrantClient, name: str, *, recreate: bool = False) -> None:
    """Create the dense collection if missing. Idempotent.

    ``recreate=True`` drops any existing collection first — for re-indexing
    runs where the embedding model or vector dim has changed.
    """
    exists = client.collection_exists(collection_name=name)
    if exists and recreate:
        logger.info("dropping existing collection: %s", name)
        client.delete_collection(collection_name=name)
        exists = False
    if not exists:
        logger.info("creating collection %s (dim=%d, distance=%s)",
                    name, DENSE_DIM, DENSE_DISTANCE.value)
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=DENSE_DIM, distance=DENSE_DISTANCE),
        )


def upsert_batch(
    client: QdrantClient,
    collection: str,
    records: Sequence[ChunkRecord],
    vectors: Sequence[Sequence[float]],
) -> int:
    """Upsert ``records`` with their ``vectors`` into ``collection``.

    Point ID = UUID(first 128 bits of chunk_hash), so re-running with the same
    chunks overwrites in place — no duplicates.
    """
    if len(records) != len(vectors):
        raise ValueError(f"len mismatch: {len(records)} records vs {len(vectors)} vectors")
    if not records:
        return 0
    points = [
        qm.PointStruct(
            id=point_id_for(r.chunk_hash),
            vector=list(v),
            payload=r.payload(),
        )
        for r, v in zip(records, vectors)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def count_points(client: QdrantClient, collection: str) -> int:
    """Exact point count — handy for smoke checks and idempotency assertions."""
    return client.count(collection_name=collection, exact=True).count


def chunks(seq: Iterable, n: int) -> Iterable[list]:
    """Yield successive ``n``-sized chunks from ``seq``."""
    buf: list = []
    for item in seq:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
