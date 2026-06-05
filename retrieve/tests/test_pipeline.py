"""Pipeline test: stub embedder + stub reranker + in-memory Qdrant.

Asserts the orchestration: query → embed → ANN → rerank → top-K, with the
final ordering driven by rerank (not ANN) and filters narrowing correctly.
"""
import hashlib

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient

from embed.jsonl import ChunkRecord
from embed.qdrant import DENSE_DIM, ensure_collection, upsert_batch, upsert_hybrid_batch
from retrieve.pipeline import anchored_retrieve, hybrid_retrieve_and_rerank, retrieve_and_rerank


COLL = "test_pipeline"


def _unit_vec(direction: int) -> list[float]:
    v = [0.0] * DENSE_DIM
    v[direction] = 1.0
    return v


def _record(text: str, *, lang: str = "en", source: str = "tsb", idx: int = 0) -> ChunkRecord:
    h = hashlib.sha256(f"{source}/{idx}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=f"{source}/doc{idx:03d}",
        source_url=None,
        section_title="",
        page=idx + 1,
        bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=h,
        lang=lang,
        text=text,
    )


class StubEmbedder:
    def __init__(self, query_vec_axis: int = 0):
        self._axis = query_vec_axis

    def embed(self, texts):
        # Query always embeds to axis ``_axis``; we use this to control which
        # corpus point is the ANN winner.
        return [_unit_vec(self._axis) for _ in texts]


class StubReranker:
    """Returns scores from a dict keyed by passage text."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.last_query: str | None = None
        self.last_passages: list[str] | None = None

    def score(self, query, passages):
        self.last_query = query
        self.last_passages = list(passages)
        return [self._scores[p] for p in passages]


@pytest.fixture
def client() -> QdrantClient:
    c = QdrantClient(":memory:")
    ensure_collection(c, COLL)
    return c


def test_rerank_drives_final_order(client: QdrantClient):
    # 3 chunks on distinct axes. Query embeds to axis 0 → ANN order: a, b, c.
    # But rerank scores favour c.
    records = [_record(t, idx=i) for i, t in enumerate(["a", "b", "c"])]
    upsert_batch(client, COLL, records, [_unit_vec(i) for i in range(3)])
    out = retrieve_and_rerank(
        "query",
        embedder=StubEmbedder(query_vec_axis=0),
        reranker=StubReranker({"a": 0.1, "b": 0.4, "c": 0.9}),
        client=client, collection=COLL, ann_k=3, top_k=3,
    )
    assert [c.record.text for c in out] == ["c", "b", "a"]


def test_top_k_truncates_post_rerank(client: QdrantClient):
    records = [_record(t, idx=i) for i, t in enumerate(list("abcde"))]
    upsert_batch(client, COLL, records, [_unit_vec(i) for i in range(5)])
    out = retrieve_and_rerank(
        "q",
        embedder=StubEmbedder(0),
        reranker=StubReranker({"a": 1, "b": 5, "c": 3, "d": 2, "e": 4}),
        client=client, collection=COLL, ann_k=5, top_k=2,
    )
    assert [c.record.text for c in out] == ["b", "e"]


def test_ann_k_caps_what_reranker_sees(client: QdrantClient):
    records = [_record(t, idx=i) for i, t in enumerate(list("abcde"))]
    upsert_batch(client, COLL, records, [_unit_vec(i) for i in range(5)])
    rer = StubReranker({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1})
    out = retrieve_and_rerank(
        "q",
        embedder=StubEmbedder(0),
        reranker=rer,
        client=client, collection=COLL, ann_k=3, top_k=3,
    )
    assert len(rer.last_passages or []) == 3
    assert len(out) == 3


def test_lang_filter_narrows_candidates(client: QdrantClient):
    records = [
        _record("en1", lang="en", idx=0),
        _record("fr1", lang="fr", idx=1),
        _record("en2", lang="en", idx=2),
    ]
    upsert_batch(client, COLL, records, [_unit_vec(i) for i in range(3)])
    out = retrieve_and_rerank(
        "q",
        embedder=StubEmbedder(0),
        reranker=StubReranker({"fr1": 0.5}),
        client=client, collection=COLL, ann_k=10, top_k=10, lang="fr",
    )
    assert len(out) == 1
    assert out[0].record.lang == "fr"


def test_empty_query_returns_empty(client: QdrantClient):
    out = retrieve_and_rerank(
        "   ",
        embedder=StubEmbedder(0),
        reranker=StubReranker({}),
        client=client, collection=COLL, ann_k=10, top_k=10,
    )
    assert out == []


def test_ann_k_lt_top_k_raises(client: QdrantClient):
    with pytest.raises(ValueError):
        retrieve_and_rerank(
            "q",
            embedder=StubEmbedder(0),
            reranker=StubReranker({}),
            client=client, collection=COLL, ann_k=3, top_k=10,
        )


def test_no_candidates_returns_empty(client: QdrantClient):
    # Empty collection — search returns 0, pipeline must not crash.
    out = retrieve_and_rerank(
        "q",
        embedder=StubEmbedder(0),
        reranker=StubReranker({}),
        client=client, collection=COLL, ann_k=10, top_k=10,
    )
    assert out == []


# ─── anchored_retrieve ───────────────────────────────────────────────────────

def _doc_rec(doc: str, page: int, text: str, *, lang: str = "en") -> ChunkRecord:
    h = hashlib.sha256(f"{doc}:{page}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=doc, source_url=None, section_title="", page=page,
        bbox=[0.0, 0.0, 0.0, 0.0], chunk_hash=h, lang=lang, text=text,
    )


def test_anchored_surfaces_body_pages_not_just_seed(client: QdrantClient):
    # doc000 wins the seed via its keyword title page, but its body pages —
    # which chunk-level top_k=1 would never return — must appear in the output.
    title = _doc_rec("tsb/doc000", 1, "fuel exhaustion title")
    body_a = _doc_rec("tsb/doc000", 2, "tanks ran dry on approach")
    body_b = _doc_rec("tsb/doc000", 3, "recommend pre-flight fuel check")
    other = _doc_rec("tsb/doc999", 1, "unrelated weather report")
    recs = [title, body_a, body_b, other]
    # title + bodies on axis 0 (seed picks doc000); other on axis 1.
    vecs = [_unit_vec(0), _unit_vec(0), _unit_vec(0), _unit_vec(1)]
    upsert_batch(client, COLL, recs, vecs)
    out = anchored_retrieve(
        "fuel exhaustion",
        embedder=StubEmbedder(0),
        reranker=StubReranker({
            title.text: 0.99, body_a.text: 0.7, body_b.text: 0.8,
            other.text: 0.1,
        }),
        client=client, collection=COLL,
        ann_k=10, top_k=1, top_n_docs=1,
    )
    texts = [c.record.text for c in out]
    assert texts == [title.text, body_a.text, body_b.text]  # reading order
    assert other.text not in texts  # not an anchor doc


def test_anchored_char_budget_caps_selection(client: QdrantClient):
    big = "z" * 4000
    r1 = _doc_rec("tsb/doc000", 1, big + "1")
    r2 = _doc_rec("tsb/doc000", 2, big + "2")
    r3 = _doc_rec("tsb/doc000", 3, big + "3")
    upsert_batch(client, COLL, [r1, r2, r3], [_unit_vec(0)] * 3)
    out = anchored_retrieve(
        "q",
        embedder=StubEmbedder(0),
        reranker=StubReranker({r1.text: 0.5, r2.text: 0.9, r3.text: 0.7}),
        client=client, collection=COLL,
        ann_k=10, top_k=3, top_n_docs=1, char_budget=5000,
    )
    # budget 5000, each ~4001 chars → only the top-reranked (r2) fits
    assert [c.record.text for c in out] == [r2.text]


def test_anchored_lang_filter_excludes_other_language(client: QdrantClient):
    en = _doc_rec("tsb/doc000", 1, "english body", lang="en")
    fr = _doc_rec("tsb/doc000", 2, "corps francais", lang="fr")
    upsert_batch(client, COLL, [en, fr], [_unit_vec(0), _unit_vec(0)])
    out = anchored_retrieve(
        "q",
        embedder=StubEmbedder(0),
        reranker=StubReranker({en.text: 0.9, fr.text: 0.8}),
        client=client, collection=COLL,
        ann_k=10, top_k=2, top_n_docs=1, lang="en",
    )
    assert [c.record.lang for c in out] == ["en"]


def test_anchored_empty_seed_returns_empty(client: QdrantClient):
    out = anchored_retrieve(
        "q",
        embedder=StubEmbedder(0),
        reranker=StubReranker({}),
        client=client, collection=COLL, ann_k=10, top_k=5,
    )
    assert out == []


# ─── hybrid_retrieve_and_rerank ──────────────────────────────────────────────

HYBRID_COLL = "test_pipeline_hybrid"


class StubHybridEmbedder(StubEmbedder):
    """Adds ``embed_sparse`` so hybrid pipeline can use it."""

    def embed_sparse(self, texts):
        return [{i: 0.5} for i, _ in enumerate(texts)]


@pytest.fixture
def hybrid_client() -> QdrantClient:
    c = QdrantClient(":memory:")
    ensure_collection(c, HYBRID_COLL, with_sparse=True)
    return c


def test_hybrid_returns_reranked_results(hybrid_client: QdrantClient):
    records = [_record(t, idx=i) for i, t in enumerate(["a", "b", "c"])]
    dense = [_unit_vec(i) for i in range(3)]
    sparse = [{i: 0.9} for i in range(3)]
    upsert_hybrid_batch(hybrid_client, HYBRID_COLL, records, dense, sparse)
    out = hybrid_retrieve_and_rerank(
        "CAR 605.38",
        embedder=StubHybridEmbedder(query_vec_axis=0),
        reranker=StubReranker({"a": 0.3, "b": 0.9, "c": 0.1}),
        client=hybrid_client, collection=HYBRID_COLL, ann_k=3, top_k=3,
    )
    assert out[0].record.text == "b"


def test_hybrid_falls_back_to_dense_on_empty_sparse(hybrid_client: QdrantClient):
    records = [_record(t, idx=i) for i, t in enumerate(["x", "y"])]
    dense = [_unit_vec(i) for i in range(2)]
    upsert_hybrid_batch(hybrid_client, HYBRID_COLL, records, dense,
                        [{}, {}])  # empty sparse weights → no sparse hits
    out = hybrid_retrieve_and_rerank(
        "q",
        embedder=StubHybridEmbedder(query_vec_axis=0),
        reranker=StubReranker({"x": 0.8, "y": 0.3}),
        client=hybrid_client, collection=HYBRID_COLL, ann_k=5, top_k=2,
    )
    assert len(out) == 2


def test_hybrid_empty_query_returns_empty(hybrid_client: QdrantClient):
    out = hybrid_retrieve_and_rerank(
        "   ",
        embedder=StubHybridEmbedder(0),
        reranker=StubReranker({}),
        client=hybrid_client, collection=HYBRID_COLL, ann_k=5, top_k=5,
    )
    assert out == []
