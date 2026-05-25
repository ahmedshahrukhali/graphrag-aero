"""Tests for rerank() — uses a stub cross-encoder, no FlagEmbedding load."""
from typing import Sequence

import pytest

from embed.jsonl import ChunkRecord
from retrieve.reranker import ScoredChunk, rerank


class StubReranker:
    """Predetermined scores per text; remembers calls."""

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores = scores_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        return [self._scores[p] for p in passages]


def _record(text: str, doc: str = "tsb/x", page: int = 1) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc, source_url=None, section_title="", page=page,
        bbox=[0.0, 0.0, 0.0, 0.0], chunk_hash=f"{hash(text):064x}",
        lang="en", text=text,
    )


def test_rerank_orders_by_rerank_score_desc():
    cands = [
        ScoredChunk(_record("a"), ann_score=0.9),
        ScoredChunk(_record("b"), ann_score=0.8),
        ScoredChunk(_record("c"), ann_score=0.7),
    ]
    # ANN order is a > b > c; rerank flips it.
    stub = StubReranker({"a": 0.1, "b": 0.5, "c": 0.9})
    out = rerank("q", cands, stub)
    assert [c.record.text for c in out] == ["c", "b", "a"]
    assert [c.rerank_score for c in out] == [0.9, 0.5, 0.1]


def test_rerank_preserves_ann_score():
    cands = [
        ScoredChunk(_record("a"), ann_score=0.9),
        ScoredChunk(_record("b"), ann_score=0.4),
    ]
    out = rerank("q", cands, StubReranker({"a": 0.2, "b": 0.7}))
    # b is on top now, but original ann_score must still be 0.4.
    assert out[0].record.text == "b"
    assert out[0].ann_score == 0.4
    assert out[1].ann_score == 0.9


def test_rerank_calls_encoder_once_with_all_passages():
    cands = [
        ScoredChunk(_record("a"), ann_score=0.9),
        ScoredChunk(_record("b"), ann_score=0.8),
        ScoredChunk(_record("c"), ann_score=0.7),
    ]
    stub = StubReranker({"a": 1.0, "b": 2.0, "c": 3.0})
    rerank("q", cands, stub)
    assert len(stub.calls) == 1
    assert stub.calls[0][0] == "q"
    assert stub.calls[0][1] == ["a", "b", "c"]


def test_rerank_top_k_truncates():
    cands = [ScoredChunk(_record(t), ann_score=0.0) for t in "abcde"]
    stub = StubReranker({"a": 1, "b": 5, "c": 3, "d": 2, "e": 4})
    out = rerank("q", cands, stub, top_k=2)
    assert [c.record.text for c in out] == ["b", "e"]


def test_rerank_empty_returns_empty():
    assert rerank("q", [], StubReranker({})) == []


def test_rerank_score_count_mismatch_raises():
    class BadReranker:
        def score(self, q, ps):
            return [0.1]  # wrong length
    with pytest.raises(ValueError):
        rerank("q", [ScoredChunk(_record("a"), 0.0), ScoredChunk(_record("b"), 0.0)], BadReranker())


def test_scored_chunk_final_score_falls_back_to_ann():
    sc = ScoredChunk(_record("a"), ann_score=0.42, rerank_score=None)
    assert sc.final_score == 0.42
    sc2 = ScoredChunk(_record("a"), ann_score=0.42, rerank_score=0.91)
    assert sc2.final_score == 0.91
