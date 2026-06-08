"""Tests for dense_search, sparse_search, rrf_fuse — real qdrant-client :memory:."""
import hashlib

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient

from embed.jsonl import ChunkRecord
from embed.qdrant import DENSE_DIM, ensure_collection, upsert_batch, upsert_hybrid_batch
from retrieve.reranker import ScoredChunk
from retrieve.search import _build_filter, dense_search, rrf_fuse, scroll_doc_chunks, sparse_search


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
    f = _build_filter(lang=["fr"], source=None)
    assert f is not None
    assert len(f.must) == 1
    assert f.must[0].key == "lang"


def test_build_filter_both():
    f = _build_filter(lang=["en"], source=["tsb"])
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
    fr_hits = dense_search(client, COLL, _unit_vec(0), k=10, lang=["fr"])
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
    tc_hits = dense_search(client, COLL, _unit_vec(0), k=10, source=["tc"])
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
    hits = dense_search(client, COLL, _unit_vec(0), k=10, lang=["fr"], source=["tsb"])
    assert len(hits) == 1
    assert hits[0].record.text == "b"


def test_search_returns_hydrated_record(client: QdrantClient):
    rec = _record("hello", idx=0)
    upsert_batch(client, COLL, [rec], [_unit_vec(0)])
    hits = dense_search(client, COLL, _unit_vec(0), k=1)
    assert isinstance(hits[0].record, ChunkRecord)
    assert hits[0].record.text == "hello"
    assert hits[0].record.chunk_hash == rec.chunk_hash


# ─── scroll_doc_chunks ───────────────────────────────────────────────────────

def _doc_rec(doc: str, page: int, text: str) -> ChunkRecord:
    h = hashlib.sha256(f"{doc}:{page}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=doc, source_url=None, section_title="", page=page,
        bbox=[0.0, 0.0, 0.0, 0.0], chunk_hash=h, lang="en", text=text,
    )


def test_scroll_doc_chunks_returns_all_chunks_for_doc(client: QdrantClient):
    recs = [_doc_rec("tsb/doc000", p, f"chunk{p}") for p in range(1, 4)]
    recs.append(_doc_rec("tsb/doc999", 1, "other"))
    upsert_batch(client, COLL, recs, [_unit_vec(i) for i in range(len(recs))])
    out = scroll_doc_chunks(client, COLL, ["tsb/doc000"])
    assert len(out) == 3
    assert {c.record.text for c in out} == {"chunk1", "chunk2", "chunk3"}
    assert all(c.ann_score == 1.0 for c in out)


def test_scroll_doc_chunks_multiple_docs(client: QdrantClient):
    recs = [
        _doc_rec("tsb/doc000", 1, "a"),
        _doc_rec("tsb/doc001", 1, "b"),
        _doc_rec("tsb/doc002", 1, "c"),
    ]
    upsert_batch(client, COLL, recs, [_unit_vec(i) for i in range(3)])
    out = scroll_doc_chunks(client, COLL, ["tsb/doc000", "tsb/doc002"])
    assert {c.record.doc_id for c in out} == {"tsb/doc000", "tsb/doc002"}


def test_scroll_doc_chunks_empty_ids(client: QdrantClient):
    assert scroll_doc_chunks(client, COLL, []) == []


def test_scroll_doc_chunks_paginates(client: QdrantClient):
    # more chunks than page_size — exercise the scroll loop
    recs = [_doc_rec("tsb/doc000", p, f"c{p}") for p in range(1, 11)]
    upsert_batch(client, COLL, recs, [_unit_vec(i % DENSE_DIM) for i in range(10)])
    out = scroll_doc_chunks(client, COLL, ["tsb/doc000"], page_size=3)
    assert len(out) == 10


# ─── sparse_search ───────────────────────────────────────────────────────────

HYBRID_COLL = "test_search_hybrid"


@pytest.fixture
def hybrid_client() -> QdrantClient:
    c = QdrantClient(":memory:")
    ensure_collection(c, HYBRID_COLL, with_sparse=True)
    return c


def test_sparse_search_returns_hits(hybrid_client: QdrantClient):
    rec = _record("CAR 605.38 fuel", idx=0)
    dense = _unit_vec(0)
    # token 5 (e.g. "fuel") has high weight → sparse query on token 5 should score high
    upsert_hybrid_batch(hybrid_client, HYBRID_COLL, [rec], [dense], [{5: 0.9, 10: 0.3}])
    hits = sparse_search(hybrid_client, HYBRID_COLL, [5], [1.0], k=5)
    assert len(hits) == 1
    assert hits[0].record.text == "CAR 605.38 fuel"
    assert hits[0].ann_score > 0


def test_sparse_search_empty_weights_returns_empty(hybrid_client: QdrantClient):
    assert sparse_search(hybrid_client, HYBRID_COLL, [], [], k=5) == []


def test_sparse_search_graceful_on_dense_only_collection(client: QdrantClient):
    # dense-only collection: sparse_search degrades to empty, not an exception
    results = sparse_search(client, COLL, [0], [1.0], k=5)
    assert results == []


# ─── rrf_fuse ────────────────────────────────────────────────────────────────

def _scored(text: str, idx: int, score: float) -> ScoredChunk:
    h = hashlib.sha256(f"rrf:{idx}:{text}".encode()).hexdigest()
    rec = ChunkRecord(
        doc_id=f"tsb/doc{idx:03d}", source_url=None, section_title="",
        page=idx + 1, bbox=[0.0, 0.0, 0.0, 0.0], chunk_hash=h, lang="en", text=text,
    )
    return ScoredChunk(record=rec, ann_score=score)


def test_rrf_fuse_chunk_in_both_ranks_highest():
    # c1 is in dense at rank 1, sparse at rank 2 → highest RRF
    # c2 is in dense at rank 2, sparse at rank 1 → also high
    # c3 is dense-only at rank 3
    c1 = _scored("both-1", 1, 0.9)
    c2 = _scored("both-2", 2, 0.7)
    c3 = _scored("dense-only", 3, 0.5)
    c4 = _scored("both-2-sparse", 2, 0.8)  # same chunk_hash as c2

    # share chunk_hash between c2 and c4
    c4 = ScoredChunk(record=c2.record, ann_score=0.8)

    dense = [c1, c2, c3]
    sparse = [c4, c1]  # c2 rank 1, c1 rank 2

    fused = rrf_fuse(dense, sparse)
    # c1 appears at dense rank 1 + sparse rank 2; c2 appears at dense rank 2 + sparse rank 1
    # RRF(c1) = 1/(60+1) + 1/(60+2); RRF(c2) = 1/(60+2) + 1/(60+1)
    # They tie by symmetry; c3 is dense-only → lower
    assert fused[0].record.chunk_hash in {c1.record.chunk_hash, c2.record.chunk_hash}
    assert fused[-1].record.chunk_hash == c3.record.chunk_hash


def test_rrf_fuse_dense_only_input():
    c1 = _scored("only-dense", 1, 0.9)
    c2 = _scored("only-dense-2", 2, 0.7)
    fused = rrf_fuse([c1, c2], [])
    # With no sparse hits, sparse rank is len(sparse)+1 = 1 for all
    assert fused[0].record.chunk_hash == c1.record.chunk_hash


def test_rrf_fuse_empty_both():
    assert rrf_fuse([], []) == []


def test_rrf_fuse_preserves_all_unique_chunks():
    dense = [_scored(f"d{i}", i, 0.9 - i * 0.1) for i in range(3)]
    sparse = [_scored(f"s{i}", i + 10, 0.8 - i * 0.1) for i in range(3)]
    fused = rrf_fuse(dense, sparse)
    assert len(fused) == 6  # 3 dense-only + 3 sparse-only, no overlap
