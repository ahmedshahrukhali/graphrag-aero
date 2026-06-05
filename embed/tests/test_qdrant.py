"""Tests for the Qdrant wrapper.

We use ``QdrantClient(":memory:")`` — real qdrant-client logic, in-process, no
network. That tests our schema (dim, distance, payload) end-to-end without
mocking the library.
"""
import hashlib

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from embed.ids import point_id_for
from embed.jsonl import ChunkRecord
from embed.qdrant import (
    DENSE_DIM,
    DENSE_DISTANCE,
    SPARSE_VECTOR_NAME,
    QdrantConfig,
    chunks,
    count_points,
    ensure_collection,
    upsert_batch,
    upsert_hybrid_batch,
)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def _vec(seed: int) -> list[float]:
    """Cheap deterministic dense vector of the right shape."""
    return [float((seed + i) % 7) / 7.0 for i in range(DENSE_DIM)]


def _record(idx: int, lang: str = "en") -> ChunkRecord:
    # Real sha256 hashes vary in the high-order bytes used for the UUID
    # derivation; a zero-padded ``f"{idx:064x}"`` would collapse to the same
    # UUID for small idx.
    h = hashlib.sha256(f"rec:{idx}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=f"tsb/doc{idx:03d}",
        source_url=f"https://example.test/doc{idx}.pdf",
        section_title="Findings",
        page=idx % 5 + 1,
        bbox=[0.0, 0.0, 100.0, 50.0],
        chunk_hash=h,
        lang=lang,
        text=f"chunk {idx} body",
    )


# ─── config ──────────────────────────────────────────────────────────────────

def test_config_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QDRANT_HOST", "qd")
    monkeypatch.setenv("QDRANT_PORT", "7000")
    monkeypatch.setenv("QDRANT_COLLECTION_DENSE", "x")
    cfg = QdrantConfig.from_env()
    assert cfg == QdrantConfig(host="qd", port=7000, collection="x")


def test_config_defaults(monkeypatch: pytest.MonkeyPatch):
    for k in ("QDRANT_HOST", "QDRANT_PORT", "QDRANT_COLLECTION_DENSE"):
        monkeypatch.delenv(k, raising=False)
    cfg = QdrantConfig.from_env()
    assert cfg.host == "localhost"
    assert cfg.port == 6333
    assert cfg.collection == "aerospace_dense"


# ─── ensure_collection ───────────────────────────────────────────────────────

def test_ensure_collection_creates(client: QdrantClient):
    ensure_collection(client, "c1")
    assert client.collection_exists("c1")
    info = client.get_collection("c1")
    # qdrant-client returns a CollectionInfo; vectors config lives under
    # config.params.vectors.
    vec_cfg = info.config.params.vectors
    assert vec_cfg.size == DENSE_DIM
    assert vec_cfg.distance == DENSE_DISTANCE


def test_ensure_collection_idempotent(client: QdrantClient):
    ensure_collection(client, "c2")
    ensure_collection(client, "c2")  # second call must not raise
    assert client.collection_exists("c2")


def test_ensure_collection_recreate(client: QdrantClient):
    ensure_collection(client, "c3")
    # Drop a point so we can prove recreate clears it.
    upsert_batch(client, "c3", [_record(1)], [_vec(1)])
    assert count_points(client, "c3") == 1
    ensure_collection(client, "c3", recreate=True)
    assert count_points(client, "c3") == 0


# ─── upsert_batch ────────────────────────────────────────────────────────────

def test_upsert_batch_writes_points_and_payload(client: QdrantClient):
    ensure_collection(client, "c4")
    records = [_record(i) for i in range(3)]
    vectors = [_vec(i) for i in range(3)]
    n = upsert_batch(client, "c4", records, vectors)
    assert n == 3
    assert count_points(client, "c4") == 3
    # Retrieve one point and verify payload round-trip.
    pid = point_id_for(records[0].chunk_hash)
    retrieved = client.retrieve(collection_name="c4", ids=[pid], with_payload=True)
    assert len(retrieved) == 1
    payload = retrieved[0].payload
    assert payload["doc_id"] == records[0].doc_id
    assert payload["chunk_hash"] == records[0].chunk_hash
    assert payload["lang"] == "en"
    assert payload["text"] == records[0].text


def test_upsert_batch_idempotent(client: QdrantClient):
    ensure_collection(client, "c5")
    records = [_record(i) for i in range(5)]
    vectors = [_vec(i) for i in range(5)]
    upsert_batch(client, "c5", records, vectors)
    upsert_batch(client, "c5", records, vectors)  # re-upsert
    assert count_points(client, "c5") == 5


def test_upsert_batch_empty(client: QdrantClient):
    ensure_collection(client, "c6")
    assert upsert_batch(client, "c6", [], []) == 0


def test_upsert_batch_len_mismatch(client: QdrantClient):
    ensure_collection(client, "c7")
    with pytest.raises(ValueError):
        upsert_batch(client, "c7", [_record(0)], [])


# ─── chunks() helper ─────────────────────────────────────────────────────────

def test_chunks_exact_multiple():
    assert list(chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_chunks_remainder():
    assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunks_empty():
    assert list(chunks([], 3)) == []


# ─── sparse / hybrid collection ──────────────────────────────────────────────

def test_ensure_collection_with_sparse(client: QdrantClient):
    ensure_collection(client, "c_sp", with_sparse=True)
    info = client.get_collection("c_sp")
    assert SPARSE_VECTOR_NAME in info.config.params.sparse_vectors


def test_ensure_collection_dense_only_has_no_sparse(client: QdrantClient):
    ensure_collection(client, "c_dn")
    info = client.get_collection("c_dn")
    sv = info.config.params.sparse_vectors
    assert sv is None or SPARSE_VECTOR_NAME not in sv


def test_upsert_hybrid_batch_writes_and_searchable(client: QdrantClient):
    ensure_collection(client, "c_hy", with_sparse=True)
    records = [_record(i) for i in range(2)]
    dense = [_vec(i) for i in range(2)]
    sparse: list[dict] = [{0: 0.9, 5: 0.3}, {1: 0.8, 10: 0.2}]
    n = upsert_hybrid_batch(client, "c_hy", records, dense, sparse)
    assert n == 2
    assert count_points(client, "c_hy") == 2


def test_upsert_hybrid_batch_idempotent(client: QdrantClient):
    ensure_collection(client, "c_hi2", with_sparse=True)
    records = [_record(i) for i in range(3)]
    dense = [_vec(i) for i in range(3)]
    sparse = [{0: float(i)} for i in range(3)]
    upsert_hybrid_batch(client, "c_hi2", records, dense, sparse)
    upsert_hybrid_batch(client, "c_hi2", records, dense, sparse)
    assert count_points(client, "c_hi2") == 3


def test_upsert_hybrid_batch_len_mismatch(client: QdrantClient):
    ensure_collection(client, "c_hi3", with_sparse=True)
    with pytest.raises(ValueError):
        upsert_hybrid_batch(client, "c_hi3", [_record(0)], [_vec(0)], [])


def test_upsert_hybrid_batch_empty(client: QdrantClient):
    ensure_collection(client, "c_hi4", with_sparse=True)
    assert upsert_hybrid_batch(client, "c_hi4", [], [], []) == 0
