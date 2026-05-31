"""Pure-helper tests for the Gradio app (no Blocks rendering).

Skipped where gradio isn't installed — ``hf_space.app`` imports it at module
load. Covers the display-ordering (P1) and citation-detection (P3) helpers.
"""
from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from hf_space.app import (
    _cited_keys,
    _parse_citations,
    _query_terms,
    _sources_to_retrieve,
    _thought_from_trace,
)


def _src(doc_id: str, page: int, rerank: float | None) -> dict:
    return {
        "doc_id": doc_id, "source_url": f"https://x/{doc_id}.pdf",
        "section_title": "", "page": page, "bbox": [0, 0, 0, 0],
        "lang": "en", "text": "body", "ann_score": 0.1, "rerank_score": rerank,
    }


def test_sources_sorted_by_rerank_desc_with_rank_reassigned():
    # Backend sends reading order; display must show relevance order.
    out = _sources_to_retrieve(
        [_src("d/a", 7, 0.40), _src("d/b", 1, 0.95), _src("d/c", 3, 0.80)],
        "q",
    )
    assert [c.rerank_score for c in out.results] == [0.95, 0.80, 0.40]
    assert [c.rank for c in out.results] == [1, 2, 3]
    assert out.results[0].doc_id == "d/b"


def test_sources_sort_puts_missing_rerank_last():
    out = _sources_to_retrieve(
        [_src("d/a", 1, None), _src("d/b", 2, 0.5)],
        "q",
    )
    assert out.results[0].rerank_score == 0.5
    assert out.results[1].rerank_score is None


def test_cited_keys_extracts_bare_tags_without_quotes():
    answer = "The copilot let speed decay [tsb/a96p0006 p.2] and recovered [tsb/a08o0029 p. 7]."
    assert _cited_keys(answer) == {("tsb/a96p0006", 2), ("tsb/a08o0029", 7)}


def test_cited_keys_empty_for_uncited_answer():
    assert _cited_keys("I need more detail to answer that.") == set()


def test_cited_keys_tolerates_section_title_in_tag():
    # gemma copies the chunk's section title into the bracket, e.g.
    # [tsb/a03q0109 p.2 §26 JULY 2003]. Observed live; must still resolve.
    answer = (
        "Fuel exhaustion was the cause [tsb/a03q0109 p.2 §26 JULY 2003] and "
        "planning was deficient [tsb/a13q0098 p.56 §10 JUNE 2013]."
    )
    assert _cited_keys(answer) == {("tsb/a03q0109", 2), ("tsb/a13q0098", 56)}


def test_parse_citations_still_captures_quotes_when_present():
    answer = 'The report [tsb/a01 p.4] states that "the engine failed".'
    assert _parse_citations(answer) == {("tsb/a01", 4): "the engine failed"}


def test_parse_citations_captures_quote_after_section_title_tag():
    answer = 'The report [tsb/a01 p.4 §SUMMARY] states that "the engine failed".'
    assert _parse_citations(answer) == {("tsb/a01", 4): "the engine failed"}


def test_query_terms_drops_stopwords_and_short_tokens():
    # "for" is a stopword; "of"/"in" are <3 chars → dropped.
    assert _query_terms("fuel exhaustion forced landing") == (
        "fuel", "exhaustion", "forced", "landing",
    )


def test_query_terms_dedupes_and_lowercases():
    assert _query_terms("Engine ENGINE failure") == ("engine", "failure")


def test_query_terms_handles_french_accents():
    terms = _query_terms("décrochage et collision avec le relief")
    assert "décrochage" in terms
    assert "collision" in terms
    assert "relief" in terms
    assert "et" not in terms  # <3 chars
    assert "avec" not in terms  # FR stopword


def test_query_terms_caps_count():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima"]
    assert len(_query_terms(" ".join(words), max_terms=8)) == 8


def test_query_terms_empty():
    assert _query_terms("") == ()
    assert _query_terms("  the and of  ") == ()


def test_thought_from_trace_renders_each_node():
    trace = [
        {"node": "retrieve", "n_new": 11, "best_rerank": 0.994},
        {"node": "graph_expand", "n_rows": 3},
        {"node": "synthesize", "draft_chars": 1418},
    ]
    md, n = _thought_from_trace(trace)
    assert n == 3
    assert "Retrieved 11 chunks · best score 0.99" in md
    assert "Graph context · 3 edges" in md
    assert "Synthesised the answer" in md


def test_thought_from_trace_empty_falls_back():
    md, n = _thought_from_trace([])
    assert n == 1
    assert "cached" in md.lower()
