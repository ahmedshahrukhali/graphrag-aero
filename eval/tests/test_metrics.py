"""Unit tests for eval/metrics.py."""
from __future__ import annotations

import math
from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    actual = ["doc1", "doc2", "doc3", "doc4"]
    expected = ["doc2", "doc4"]

    assert recall_at_k(actual, expected, k=1) == 0.0
    assert recall_at_k(actual, expected, k=2) == 0.5  # Only doc2
    assert recall_at_k(actual, expected, k=3) == 0.5  # Only doc2
    assert recall_at_k(actual, expected, k=4) == 1.0  # doc2 and doc4
    assert recall_at_k(actual, [], k=2) == 0.0


def test_reciprocal_rank():
    expected = ["doc3"]
    assert reciprocal_rank(["doc1", "doc2", "doc3"], expected) == 1.0 / 3
    assert reciprocal_rank(["doc3", "doc1"], expected) == 1.0
    assert reciprocal_rank(["doc1", "doc2"], expected) == 0.0
    assert reciprocal_rank([], expected) == 0.0


def test_ndcg_at_k():
    actual = ["doc1", "doc2", "doc3"]
    expected = ["doc2"]

    # DCG = 1 / log2(1 + 2) = 1 / log2(3) = ~0.63093
    # IDCG = 1 / log2(0 + 2) = 1.0
    # NDCG = 0.63093 / 1.0 = ~0.63093
    assert math.isclose(ndcg_at_k(actual, expected, k=3), 1.0 / math.log2(3))

    # Target in first position:
    assert ndcg_at_k(["doc2", "doc1"], expected, k=2) == 1.0

    # Target missing:
    assert ndcg_at_k(["doc1", "doc3"], expected, k=2) == 0.0

    # Empty expected:
    assert ndcg_at_k(actual, [], k=3) == 0.0


def test_ndcg_at_k_dedupes_repeated_docids():
    # Chunk-level ranking repeats a relevant doc across ranks. Doc-level nDCG
    # must collapse to unique docs so it never exceeds 1.0.
    actual = ["doc1", "doc2", "doc1", "doc3", "doc1"]
    expected = ["doc1"]
    score = ndcg_at_k(actual, expected, k=5)
    assert score == 1.0  # doc1 is the top unique doc → perfect

    # Relevant doc first appears at rank 2 (0-indexed 1) among unique docs.
    actual2 = ["doc9", "doc2", "doc9", "doc2"]
    expected2 = ["doc2"]
    score2 = ndcg_at_k(actual2, expected2, k=5)
    assert math.isclose(score2, 1.0 / math.log2(3))
    assert score2 <= 1.0
