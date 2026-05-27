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


def test_format_graph_context_minimal_fallback():
    # Pre-extraction fallback: {occ_id, occ_url} with no findings
    rows = [{"occ_id": "a01", "occ_url": "u-a", "findings": [], "recommendations": [],
             "direct_regs": [], "acs": []}]
    out = format_graph_context(rows)
    assert "a01" in out


def test_format_graph_context_rich_findings():
    rows = [{
        "occ_id": "a01",
        "occ_url": "u-a",
        "findings": [{"text": "Fuel tanks empty.", "category": "cause", "lang": "en",
                      "source_doc_id": "tsb/a01", "page": 5, "cites_reg": "602.115"}],
        "recommendations": [],
        "direct_regs": ["602.88"],
        "acs": ["700-027"],
    }]
    out = format_graph_context(rows)
    assert "a01" in out
    assert "Fuel tanks empty" in out
    assert "[tsb/a01 p.5]" in out
    assert "CAR 602.115" in out
    assert "602.88" in out
    assert "700-027" in out


def test_format_graph_context_recommendation_with_id():
    rows = [{
        "occ_id": "a01", "occ_url": "u",
        "findings": [],
        "recommendations": [{"id": "A19-01", "text": "Install TAWS.",
                              "lang": "en", "source_doc_id": "tsb/a01", "page": 30}],
        "direct_regs": [], "acs": [],
    }]
    out = format_graph_context(rows)
    assert "A19-01" in out
    assert "Install TAWS" in out
    assert "[tsb/a01 p.30]" in out


def test_build_user_prompt_includes_everything():
    out = build_user_prompt(
        "what is X?",
        [_cand("tsb/a01", 4, "alpha")],
        [{"occ_id": "a01", "occ_url": "u", "findings": [], "recommendations": [],
          "direct_regs": [], "acs": []}],
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
