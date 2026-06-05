"""Thin Qdrant wrapper: connect, ensure collection, upsert batches.

We use ``qdrant-client`` directly; nothing fancy. The wrapper exists so the CLI
and tests share one definition of "what our collection looks like" (dim,
distance, collection name) instead of duplicating it.

Sparse vector support (§1 hybrid retrieval):
  ``ensure_collection(..., with_sparse=True)`` adds a named "sparse" vector
  alongside the unnamed dense vector.  ``upsert_hybrid_batch`` upserts both.
  ``--recreate`` is required when switching a collection from dense-only to hybrid.
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
SPARSE_VECTOR_NAME = "sparse"


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


def ensure_collection(
    client: QdrantClient,
    name: str,
    *,
    recreate: bool = False,
    with_sparse: bool = False,
) -> None:
    """Create the collection if missing. Idempotent.

    ``recreate=True`` drops any existing collection first — required when
    switching between dense-only and hybrid (schema change).

    ``with_sparse=True`` adds a named "sparse" vector alongside the unnamed
    dense vector so that hybrid upserts and queries can use it.
    """
    exists = client.collection_exists(collection_name=name)
    if exists and recreate:
        logger.info("dropping existing collection: %s", name)
        client.delete_collection(collection_name=name)
        exists = False
    if not exists:
        sparse_cfg = (
            {SPARSE_VECTOR_NAME: qm.SparseVectorParams()}
            if with_sparse
            else None
        )
        logger.info(
            "creating collection %s (dim=%d, distance=%s, sparse=%s)",
            name, DENSE_DIM, DENSE_DISTANCE.value, with_sparse,
        )
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=DENSE_DIM, distance=DENSE_DISTANCE),
            sparse_vectors_config=sparse_cfg,
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


def upsert_hybrid_batch(
    client: QdrantClient,
    collection: str,
    records: Sequence[ChunkRecord],
    dense_vectors: Sequence[Sequence[float]],
    sparse_weights: Sequence[dict[int, float]],
) -> int:
    """Upsert ``records`` with dense + sparse vectors into ``collection``.

    The unnamed vector slot holds the dense vector; the named ``"sparse"``
    slot holds the BGE-M3 lexical weights as a ``SparseVector``. Requires the
    collection to have been created with ``with_sparse=True``.
    """
    if not (len(records) == len(dense_vectors) == len(sparse_weights)):
        raise ValueError(
            f"len mismatch: {len(records)} records / {len(dense_vectors)} "
            f"dense / {len(sparse_weights)} sparse"
        )
    if not records:
        return 0
    points = []
    for r, dv, sw in zip(records, dense_vectors, sparse_weights):
        indices = sorted(sw.keys())
        values = [sw[i] for i in indices]
        points.append(qm.PointStruct(
            id=point_id_for(r.chunk_hash),
            vector={
                "": list(dv),
                SPARSE_VECTOR_NAME: qm.SparseVector(indices=indices, values=values),
            },
            payload=r.payload(),
        ))
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
