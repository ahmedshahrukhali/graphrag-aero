"""Tests for dense_search — real qdrant-client in :memory: mode."""
import hashlib

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient

from embed.jsonl import ChunkRecord
from embed.qdrant import DENSE_DIM, ensure_collection, upsert_batch
from retrieve.search import _build_filter, dense_search


COLL = "test_search"


def _record(text: str, *, lang: str = "en", source: str = "tsb", idx: int = 0) -> ChunkRecord:
    h = hashlib.sha256(f"{source}/{idx}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=f"{source}/doc{idx:03d}",
        source_url=f"https://example.test/{source}/{idx}.pdf",
        section_title="",
        page=idx + 1,
        bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=h,
        lang=lang,
        text=text,
    )


def _unit_vec(direction: int) -> list[float]:
    """A 1024-vec pointing along axis ``direction`` (one-hot)."""
    v = [0.0] * DENSE_DIM
    v[direction] = 1.0
    return v


@pytest.fixture
def client() -> QdrantClient:
    c = QdrantClient(":memory:")
    ensure_collection(c, COLL)
    return c


# ─── filter construction ─────────────────────────────────────────────────────

def test_build_filter_none():
    assert _build_filter(lang=None, source=None) is None


def test_build_filter_lang_only():
    f = _build_filter(lang="fr", source=None)
    assert f is not None
    assert len(f.must) == 1
    assert f.must[0].key == "lang"


def test_build_filter_both():
    f = _build_filter(lang="en", source="tsb")
    assert f is not None
    assert {c.key for c in f.must} == {"lang", "doc_id"}


# ─── search end-to-end ───────────────────────────────────────────────────────

def test_search_orders_by_cosine_similarity(client: QdrantClient):
    # Three records on distinct one-hot axes; query points along axis 0.
    records = [_record(f"text-{i}", idx=i) for i in range(3)]
    vectors = [_unit_vec(i) for i in range(3)]
    upsert_batch(client, COLL, records, vectors)
    # Query closer to axis 0 than to 1 or 2.
    q = _unit_vec(0)
    hits = dense_search(client, COLL, q, k=3)
    assert len(hits) == 3
    assert hits[0].record.text == "text-0"
    assert hits[0].ann_score == pytest.approx(1.0, abs=1e-5)


def test_search_k_caps_results(client: QdrantClient):
    records = [_record(f"t{i}", idx=i) for i in range(5)]
    vectors = [_unit_vec(i) for i in range(5)]
    upsert_batch(client, COLL, records, vectors)
    hits = dense_search(client, COLL, _unit_vec(0), k=2)
    assert len(hits) == 2


def test_search_lang_filter(client: QdrantClient):
    records = [
        _record("en1", lang="en", idx=0),
        _record("fr1", lang="fr", idx=1),
        _record("en2", lang="en", idx=2),
    ]
    vectors = [_unit_vec(0), _unit_vec(1), _unit_vec(2)]
    upsert_batch(client, COLL, records, vectors)
    fr_hits = dense_search(client, COLL, _unit_vec(0), k=10, lang="fr")
    assert len(fr_hits) == 1
    assert fr_hits[0].record.lang == "fr"


def test_search_source_filter(client: QdrantClient):
    records = [
        _record("a", source="tsb", idx=0),
        _record("b", source="tc", idx=1),
        _record("c", source="tsb", idx=2),
    ]
    vectors = [_unit_vec(0), _unit_vec(1), _unit_vec(2)]
    upsert_batch(client, COLL, records, vectors)
    tc_hits = dense_search(client, COLL, _unit_vec(0), k=10, source="tc")
    assert len(tc_hits) == 1
    assert tc_hits[0].record.doc_id.startswith("tc/")


def test_search_combined_filter(client: QdrantClient):
    records = [
        _record("a", lang="en", source="tsb", idx=0),
        _record("b", lang="fr", source="tsb", idx=1),
        _record("c", lang="en", source="tc", idx=2),
        _record("d", lang="fr", source="tc", idx=3),
    ]
    vectors = [_unit_vec(i) for i in range(4)]
    upsert_batch(client, COLL, records, vectors)
    hits = dense_search(client, COLL, _unit_vec(0), k=10, lang="fr", source="tsb")
    assert len(hits) == 1
    assert hits[0].record.text == "b"


def test_search_returns_hydrated_record(client: QdrantClient):
    rec = _record("hello", idx=0)
    upsert_batch(client, COLL, [rec], [_unit_vec(0)])
    hits = dense_search(client, COLL, _unit_vec(0), k=1)
    assert isinstance(hits[0].record, ChunkRecord)
    assert hits[0].record.text == "hello"
    assert hits[0].record.chunk_hash == rec.chunk_hash
