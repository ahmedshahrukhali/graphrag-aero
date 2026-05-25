"""Tests for prompt formatting."""
from agent.prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_user_prompt,
    format_citations,
    format_graph_context,
)


def _cand(doc_id: str, page: int, text: str, section: str = "") -> dict:
    return {
        "doc_id": doc_id, "page": page, "section_title": section,
        "text": text, "bbox": [0.0, 0.0, 0.0, 0.0], "chunk_hash": "0" * 64,
        "lang": "en", "source_url": None, "ann_score": 0.5, "rerank_score": 0.7,
    }


def test_format_citations_empty():
    assert format_citations([]) == "(no citations)"


def test_format_citations_numbered():
    out = format_citations([
        _cand("tsb/a01", 3, "alpha", "Findings"),
        _cand("tc/ac01", 7, "beta"),
    ])
    assert "[1]" in out
    assert "[2]" in out
    assert "tsb/a01 p.3" in out
    assert "§Findings" in out
    assert "alpha" in out and "beta" in out


def test_format_citations_truncates_long_text():
    long = "x" * 5000
    out = format_citations([_cand("d", 1, long)], max_chars=100)
    # Should contain truncation marker.
    assert "..." in out
    assert len(out) < 5000


def test_format_graph_context_none():
    assert format_graph_context([]) == "(none)"


def test_format_graph_context_lists_ids():
    rows = [
        {"id": "a01", "lang": "en", "source_url": "u-a"},
        {"id": "b02", "lang": "fr", "source_url": None},
    ]
    out = format_graph_context(rows)
    assert "a01" in out and "b02" in out
    assert "en" in out and "fr" in out


def test_build_user_prompt_includes_everything():
    out = build_user_prompt(
        "what is X?",
        [_cand("tsb/a01", 4, "alpha")],
        [{"id": "a01", "lang": "en", "source_url": "u"}],
    )
    assert "what is X?" in out
    assert "tsb/a01 p.4" in out
    assert "a01" in out


def test_system_prompt_grounds_in_corpus():
    # Sanity: system prompt mentions the actual corpus sources, not generic.
    assert "Transport Canada" in SYSTEM_PROMPT or "TSB" in SYSTEM_PROMPT


def test_user_template_has_required_slots():
    # If we rename slot keys, build_user_prompt breaks; this catches that.
    for slot in ("{query}", "{graph_context}", "{citations}"):
        assert slot in USER_TEMPLATE
