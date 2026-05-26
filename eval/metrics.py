"""Evaluation metrics for information retrieval.

Includes:
- Recall@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@k)
"""
from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(actual: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Calculate Recall@k.

    Recall@k is the proportion of expected relevant documents that are present
    in the top k retrieved documents.
    """
    if not expected:
        return 0.0
    actual_k = set(actual[:k])
    intersection = actual_k.intersection(set(expected))
    return len(intersection) / len(expected)


def reciprocal_rank(actual: Sequence[str], expected: Sequence[str]) -> float:
    """Calculate Reciprocal Rank (RR).

    RR is 1 / position of the first relevant document in the retrieved list (1-indexed).
    If no expected document is found in actual, returns 0.0.
    """
    expected_set = set(expected)
    for i, doc_id in enumerate(actual, 1):
        if doc_id in expected_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(actual: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (nDCG@k).

    Uses binary relevance (1 if document is in expected, 0 otherwise).

    Relevance is judged at the doc_id level, but the ranked ``actual`` list is
    chunk-level and may repeat a doc_id across ranks. Collapse to unique docs
    (keeping each doc's best/first rank) before scoring; otherwise a single
    relevant doc appearing at several ranks double-counts and DCG can exceed
    IDCG, producing nDCG > 1.
    """
    if not expected:
        return 0.0

    seen: set[str] = set()
    unique_actual: list[str] = []
    for doc_id in actual:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_actual.append(doc_id)

    actual_k = unique_actual[:k]
    expected_set = set(expected)

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, doc_id in enumerate(actual_k):
        if doc_id in expected_set:
            dcg += 1.0 / math.log2(i + 2)

    # Calculate Ideal Discounted Cumulative Gain (IDCG)
    idcg = 0.0
    for i in range(min(k, len(expected))):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
