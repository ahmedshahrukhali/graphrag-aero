"""Pure-helper tests for the Gradio app (no Blocks rendering).

Skipped where gradio isn't installed — ``hf_space.app`` imports it at module
load. Covers the display-ordering (P1) and citation-detection (P3) helpers.
"""
from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from hf_space.app import (
    _adopt_prior,
    _append_turn_links,
    _cited_keys,
    _parse_citations,
    _query_terms,
    _sources_to_retrieve,
    _thought_from_trace,
)


# ── _adopt_prior: carry the transcript into a new turn (multi-turn) ──────────

def test_adopt_prior_none_returns_empty():
    assert _adopt_prior(None) == []


def test_adopt_prior_drops_explicit_none_metadata():
    """Gradio normalises plain turns to metadata=None; carrying that forward
    crashed the turn-rendering logic (.get('metadata', {}).get(...))."""
    chat = [
        {"role": "user", "content": "landing gear", "metadata": None},
        {"role": "assistant", "content": "an answer", "metadata": None},
    ]
    out = _adopt_prior(chat)
    assert all("metadata" not in m for m in out)
    # Every accessor in on_ask does .get("metadata", {}).get(...) — prove it's
    # now safe (the bug was a present-but-None value defeating the {} default).
    assert all(m.get("metadata", {}).get("parent_id") is None for m in out)
    # role/content preserved.
    assert out[0] == {"role": "user", "content": "landing gear"}


def test_adopt_prior_preserves_real_metadata_blocks():
    chat = [
        {"role": "assistant", "content": "steps",
         "metadata": {"title": "🧠 Thought", "id": "turn-0", "status": "done"}},
        {"role": "assistant", "content": "child",
         "metadata": {"title": "Retrieved", "parent_id": "turn-0"}},
    ]
    out = _adopt_prior(chat)
    assert out[0]["metadata"]["id"] == "turn-0"
    assert out[1]["metadata"]["parent_id"] == "turn-0"


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


def _fig_src(doc_id: str, page: int, rerank: float | None, text: str) -> dict:
    d = _src(doc_id, page, rerank)
    d.update(kind="figure", text=text, page_bboxes=[[page, 1.0, 2.0, 3.0, 4.0]])
    return d


def test_kind_propagates_through_sources():
    out = _sources_to_retrieve([_fig_src("d/a", 4, 0.9, "a runway diagram")], "q")
    assert out.results[0].kind == "figure"


def test_kind_defaults_to_text_when_absent():
    out = _sources_to_retrieve([_src("d/a", 1, 0.5)], "q")
    assert out.results[0].kind == "text"


def _capture_render(monkeypatch):
    """Patch render_page_with_bbox to record every call's draw args."""
    from unittest.mock import MagicMock

    from hf_space import app

    calls: list[dict] = []

    def fake_render(source_url, page, bbox, *, doc_id="", draw_bbox,
                    region_bboxes, cited_spans=(), terms, box_images):
        calls.append({
            "draw_bbox": draw_bbox, "box_images": box_images,
            "region_bboxes": region_bboxes, "cited_spans": cited_spans,
            "terms": terms, "doc_id": doc_id,
        })
        return MagicMock()  # _gallery_items calls .save() on the image

    monkeypatch.setattr(app, "render_page_with_bbox", fake_render)
    return app, calls


def test_gallery_marks_figure_as_ai_read_and_boxes_it(monkeypatch):
    app, calls = _capture_render(monkeypatch)
    retrieve = _sources_to_retrieve(
        [_fig_src("tsb/a00a0051", 4, 0.9, "A simplified diagram of Fox Harbour runway")],
        "q",
    )
    # Not cited — a text chunk wouldn't be boxed, but a figure always is.
    items = app._gallery_items(retrieve, draw_bbox=True, cited_dict={})
    assert len(items) == 1
    _img, caption = items[0]
    assert "🖼" in caption and "AI-read figure" in caption
    assert calls[0]["draw_bbox"] is True       # figure region shown even uncited


def test_cited_page_with_unresolvable_quote_still_gets_term_wash(monkeypatch):
    """S41 regression, updated for S42/S43: stored WS-B regions stay OFF (they
    land on header boilerplate — verified S42), so a cited page whose quote
    can't be resolved must still receive the query-term wash so it's never
    left bare."""
    app, calls = _capture_render(monkeypatch)
    src = _src("tsb/a10a0032", 2, 0.9)
    src["page_bboxes"] = [[2, 50.0, 100.0, 300.0, 140.0]]  # stored WS-B region
    src["text"] = "The aircraft departed the runway surface during the landing roll."
    retrieve = _sources_to_retrieve([src], "runway excursion")
    # Cited with an EMPTY quote and a draft that matches nothing in the chunk.
    items = app._gallery_items(
        retrieve, draw_bbox=True, cited_dict={("tsb/a10a0032", 2): ""},
        draft="Completely unrelated synthesized words about helicopters and weather.",
    )
    assert len(items) == 1
    call = calls[0]
    assert call["draw_bbox"] is True
    # Stored regions are off by design (boilerplate); no span resolved either.
    assert call["region_bboxes"] == ()
    assert call["cited_spans"] == ()
    # The query-term wash is present so the page is never left bare.
    assert "runway" in call["terms"]
    assert "excursion" in call["terms"]


def test_cited_quote_is_wired_to_renderer_as_span(monkeypatch):
    """The quote the answer attributed to a page reaches the renderer — this
    is the generation→bbox wiring (S43)."""
    app, calls = _capture_render(monkeypatch)
    src = _src("tsb/a10a0032", 2, 0.9)
    retrieve = _sources_to_retrieve([src], "runway excursion")
    items = app._gallery_items(
        retrieve, draw_bbox=True,
        cited_dict={("tsb/a10a0032", 2): "departed the runway surface"},
        draft="The aircraft departed the runway surface. [tsb/a10a0032 p.2]",
    )
    assert len(items) == 1
    assert calls[0]["cited_spans"] == ("departed the runway surface",)


def test_cited_bare_tag_falls_back_to_best_chunk_sentence_as_span(monkeypatch):
    """A bare citation tag (no quote) anchors via the chunk sentence that best
    matches the draft — that sentence is what gets located on the page."""
    app, calls = _capture_render(monkeypatch)
    src = _src("tsb/a10a0032", 2, 0.9)
    src["text"] = (
        "Weather at the aerodrome was VMC. "
        "The aircraft departed the runway surface during the landing roll. "
        "The occupants were uninjured."
    )
    retrieve = _sources_to_retrieve([src], "runway excursion")
    items = app._gallery_items(
        retrieve, draw_bbox=True, cited_dict={("tsb/a10a0032", 2): ""},
        draft="The aircraft departed the runway surface during the landing roll.",
    )
    assert len(items) == 1
    spans = calls[0]["cited_spans"]
    assert len(spans) == 1
    assert "departed the runway surface" in spans[0]


def test_bbox_toggle_off_passes_no_spans(monkeypatch):
    app, calls = _capture_render(monkeypatch)
    retrieve = _sources_to_retrieve([_src("tsb/a10a0032", 2, 0.9)], "q")
    app._gallery_items(
        retrieve, draw_bbox=False,
        cited_dict={("tsb/a10a0032", 2): "a quote"}, draft="a quote",
    )
    assert calls[0]["cited_spans"] == ()
    assert calls[0]["draw_bbox"] is False


# ── _append_turn_links: per-turn citation hyperlink trail ────────────────────

def _arts(*docs: str) -> dict:
    return {"sources": {"sources": [_src(d, 1, 0.9) for d in docs], "query": "q"}}


def test_turn_links_prefers_cited_docs():
    arts = _append_turn_links(
        _arts("tsb/a", "tsb/b"), "why crash",
        "Cause was fuel [tsb/a p.1].",
    )
    assert arts["turn_links"] == [
        {"q": "why crash", "links": [("tsb/a", "https://x/tsb/a.pdf")]}
    ]


def test_turn_links_falls_back_to_all_docs_when_uncited():
    arts = _append_turn_links(_arts("tsb/a", "tsb/b"), "q2", "No tags here.")
    assert [d for d, _ in arts["turn_links"][0]["links"]] == ["tsb/a", "tsb/b"]


def test_turn_links_appends_across_turns_and_caps():
    arts = _arts("tsb/a")
    for i in range(12):
        arts = _append_turn_links(arts, f"q{i}", "x [tsb/a p.1]")
    assert len(arts["turn_links"]) == 8           # capped
    assert arts["turn_links"][-1]["q"] == "q11"   # newest kept


def test_turn_links_no_sources_leaves_artifacts_unchanged():
    arts = {"sources": {"sources": [], "query": "q"}}
    assert _append_turn_links(dict(arts), "q", "answer") == arts


def test_cited_keys_extracts_bare_tags_without_quotes():
    answer = "The copilot let speed decay [tsb/a96p0006 p.2] and recovered [tsb/a08o0029 p. 7]."
    assert _cited_keys(answer) == {("tsb/a96p0006", 2), ("tsb/a08o0029", 7)}


def test_cited_keys_empty_for_uncited_answer():
    assert _cited_keys("I need more detail to answer that.") == set()


def test_cited_keys_tolerates_section_title_in_tag():
    # the model copies the chunk's section title into the bracket, e.g.
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
