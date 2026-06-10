"""pdf_render tests.

The bbox→pixel math is a pure function and gets unit-tested directly.
The full render path is exercised with a mocked pdfplumber so we don't
need a real PDF on disk or HTTP fetch.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from hf_space import pdf_render
from hf_space.pdf_render import (
    PdfRenderError,
    bbox_to_pixels,
    page_image_bboxes,
    render_page_with_bbox,
    search_page_span,
    search_page_terms,
)


def test_bbox_to_pixels_scales_by_dpi_over_72():
    # 1in × 1in square at the top-left, rendered at 144 DPI → 144 × 144 px.
    px = bbox_to_pixels((0.0, 0.0, 72.0, 72.0), dpi=144)
    assert px == (0, 0, 144, 144)


def test_bbox_to_pixels_default_dpi_100():
    # Default DPI is 100: 72 pt × 72 pt → 100 px × 100 px.
    px = bbox_to_pixels((0.0, 0.0, 72.0, 72.0))
    assert px == (0, 0, 100, 100)


def test_bbox_to_pixels_normalises_reversed_coords():
    # bbox came in with x1 < x0; result should still be (small, small, big, big).
    px = bbox_to_pixels((72.0, 72.0, 0.0, 0.0), dpi=72)
    assert px == (0, 0, 72, 72)


def _white_pil(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), (255, 255, 255))


def test_render_page_with_bbox_happy_path():
    """Verify the orchestration: download → open → pick page → draw → return.

    We mock pdfplumber.open and the http download. We also confirm the
    returned image differs from the bare white background (i.e. the
    rectangle was drawn somewhere).
    """
    pdf_render.render_page_with_bbox.cache_clear()

    fake_page_image = MagicMock()
    fake_page_image.original = _white_pil(800, 1000)

    fake_page = MagicMock()
    fake_page.to_image.return_value = fake_page_image

    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF-1.4\n%%EOF") as dl, \
         patch("pdfplumber.open", return_value=fake_pdf) as pp_open:
        out = render_page_with_bbox(
            "https://example.test/a.pdf", page=1, bbox=(72.0, 144.0, 216.0, 288.0), dpi=72,
        )

    dl.assert_called_once()
    pp_open.assert_called_once()
    fake_page.to_image.assert_called_once_with(resolution=72)

    assert isinstance(out, Image.Image)
    # The drawn rectangle should leave at least some non-white pixels.
    assert any(px != (255, 255, 255) for px in out.getdata())


def test_render_without_bbox_leaves_page_untouched():
    """draw_bbox=False returns the bare rasterised page (no overlay)."""
    pdf_render.render_page_with_bbox.cache_clear()

    fake_page_image = MagicMock()
    fake_page_image.original = _white_pil(800, 1000)
    fake_page = MagicMock()
    fake_page.to_image.return_value = fake_page_image
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF-1.4\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/c.pdf", page=1,
            bbox=(72.0, 144.0, 216.0, 288.0), dpi=72, draw_bbox=False,
        )

    # No overlay drawn — every pixel is still white.
    assert all(px == (255, 255, 255) for px in out.getdata())


def test_search_page_terms_collects_all_hits_across_terms():
    """Every occurrence of every term is collected (not just the first)."""
    fake_page = MagicMock()
    fake_page.search.side_effect = [
        # term "fuel" → two hits
        [{"chars": [{"x0": 10.0, "top": 10.0, "x1": 40.0, "bottom": 22.0}]},
         {"chars": [{"x0": 10.0, "top": 80.0, "x1": 40.0, "bottom": 92.0}]}],
        # term "landing" → one hit (e.g. in the title)
        [{"chars": [{"x0": 60.0, "top": 5.0, "x1": 120.0, "bottom": 18.0}]}],
    ]
    boxes = search_page_terms(fake_page, ("fuel", "landing"))
    assert len(boxes) == 3
    assert (60.0, 5.0, 120.0, 18.0) in boxes
    assert fake_page.search.call_count == 2


def test_search_page_terms_dedupes_identical_boxes():
    fake_page = MagicMock()
    fake_page.search.side_effect = [
        [{"chars": [{"x0": 10.0, "top": 10.0, "x1": 40.0, "bottom": 22.0}]}],
        # "fuels" matches the same span as "fuel" — should not double-count.
        [{"chars": [{"x0": 10.0, "top": 10.0, "x1": 40.0, "bottom": 22.0}]}],
    ]
    boxes = search_page_terms(fake_page, ("fuel", "fuels"))
    assert len(boxes) == 1


def test_search_page_terms_caps_box_count():
    fake_page = MagicMock()
    fake_page.search.return_value = [
        {"chars": [{"x0": float(i), "top": float(i), "x1": float(i) + 5, "bottom": float(i) + 5}]}
        for i in range(100)
    ]
    boxes = search_page_terms(fake_page, ("fuel",), max_boxes=10)
    assert len(boxes) == 10


def test_search_page_terms_swallows_search_errors_per_term():
    fake_page = MagicMock()
    fake_page.search.side_effect = [
        RuntimeError("boom"),
        [{"chars": [{"x0": 1.0, "top": 1.0, "x1": 2.0, "bottom": 2.0}]}],
    ]
    # First term raises, second still contributes its hit.
    boxes = search_page_terms(fake_page, ("bad", "good"))
    assert len(boxes) == 1


def test_search_page_terms_empty_terms_returns_empty():
    fake_page = MagicMock()
    assert search_page_terms(fake_page, ()) == []
    fake_page.search.assert_not_called()


def test_search_page_span_first_hit_only():
    """The cited span is a grounding box — only the FIRST occurrence counts."""
    fake_page = MagicMock()
    fake_page.search.return_value = [
        {"chars": [{"x0": 10.0, "top": 10.0, "x1": 90.0, "bottom": 22.0}]},
        {"chars": [{"x0": 10.0, "top": 400.0, "x1": 90.0, "bottom": 412.0}]},
    ]
    boxes = search_page_span(fake_page, "the aircraft departed the runway")
    assert boxes == [(10.0, 10.0, 90.0, 22.0)]
    assert fake_page.search.call_count == 1


def test_search_page_span_groups_wrapped_quote_per_line():
    """A quote wrapping across two lines yields one tight box per line, not a
    page-wide rectangle."""
    fake_page = MagicMock()
    fake_page.search.return_value = [{
        "chars": [
            {"x0": 200.0, "top": 100.0, "x1": 500.0, "bottom": 112.0},
            {"x0": 50.0, "top": 114.0, "x1": 260.0, "bottom": 126.0},
        ],
    }]
    boxes = search_page_span(fake_page, "speed decayed and the aircraft stalled")
    assert len(boxes) == 2
    assert (200.0, 100.0, 500.0, 112.0) in boxes
    assert (50.0, 114.0, 260.0, 126.0) in boxes


def test_search_page_span_falls_back_to_word_windows():
    """A long quote that doesn't match verbatim retries with its first 8 — then
    last 8 — words (hyphenation / paraphrase drift at one edge)."""
    fake_page = MagicMock()
    fake_page.search.side_effect = [
        [],  # full span misses
        [{"chars": [{"x0": 5.0, "top": 5.0, "x1": 50.0, "bottom": 15.0}]}],  # first-8 window hits
    ]
    span = "one two three four five six seven eight nine ten eleven twelve"
    boxes = search_page_span(fake_page, span)
    assert boxes == [(5.0, 5.0, 50.0, 15.0)]
    assert fake_page.search.call_count == 2


def test_search_page_span_is_punctuation_robust():
    """LLM quote 'engine, fuel' must still match — words joined with \\W+."""
    fake_page = MagicMock()
    fake_page.search.return_value = [
        {"chars": [{"x0": 1.0, "top": 1.0, "x1": 2.0, "bottom": 2.0}]}
    ]
    search_page_span(fake_page, 'the “engine, fuel” system')
    pattern = fake_page.search.call_args[0][0]
    assert "engine" in pattern and "fuel" in pattern
    assert r"\W+" in pattern
    assert "“" not in pattern  # curly quotes never reach the regex


def test_search_page_span_empty_or_no_match_returns_empty():
    fake_page = MagicMock()
    assert search_page_span(fake_page, "") == []
    fake_page.search.assert_not_called()
    fake_page.search.return_value = []
    assert search_page_span(fake_page, "nothing on page") == []


def test_search_page_span_swallows_search_errors():
    fake_page = MagicMock()
    fake_page.search.side_effect = RuntimeError("boom")
    assert search_page_span(fake_page, "any quote at all") == []


def test_page_image_bboxes_extracts_image_rects():
    fake_page = MagicMock()
    fake_page.images = [
        {"x0": 50.0, "top": 60.0, "x1": 250.0, "bottom": 300.0},
        {"x0": 5.0, "top": 5.0, "x1": 6.0, "bottom": 6.0, "extra": "ignored"},
    ]
    boxes = page_image_bboxes(fake_page)
    assert boxes == [(50.0, 60.0, 250.0, 300.0), (5.0, 5.0, 6.0, 6.0)]


def test_page_image_bboxes_no_images_returns_empty():
    fake_page = MagicMock()
    fake_page.images = []
    assert page_image_bboxes(fake_page) == []


def test_page_image_bboxes_skips_malformed_entries():
    fake_page = MagicMock()
    fake_page.images = [
        {"x0": 1.0, "top": 2.0},  # missing x1/bottom → skipped
        {"x0": 10.0, "top": 20.0, "x1": 30.0, "bottom": 40.0},
    ]
    assert page_image_bboxes(fake_page) == [(10.0, 20.0, 30.0, 40.0)]


def test_render_with_box_images_draws_red_outline():
    """box_images=True → a red rectangle is drawn around the figure."""
    pdf_render.render_page_with_bbox.cache_clear()
    fake_page_image = MagicMock()
    fake_page_image.original = _white_pil(800, 1000)
    fake_page = MagicMock()
    fake_page.to_image.return_value = fake_page_image
    fake_page.search.return_value = []
    fake_page.images = [{"x0": 100.0, "top": 100.0, "x1": 400.0, "bottom": 500.0}]
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/img.pdf", page=1,
            bbox=(0.0, 0.0, 1.0, 1.0), dpi=72, box_images=True,
        )

    # A red outline was drawn → some reddish pixels present.
    assert any(r > 180 and g < 90 and b < 90 for (r, g, b) in out.getdata())


def _fake_pdf_one_page():
    """A one-page fake pdf with a white raster and no images."""
    fake_page_image = MagicMock()
    fake_page_image.original = _white_pil(800, 1000)
    fake_page = MagicMock()
    fake_page.to_image.return_value = fake_page_image
    fake_page.images = []
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False
    return fake_pdf, fake_page


def test_render_with_region_bboxes_draws_solid_box_without_search():
    """WS-B: the cited box comes from the stored region rect — drawn directly,
    and page.search is NEVER called for it (the desync path is gone)."""
    pdf_render.render_page_with_bbox.cache_clear()
    fake_pdf, fake_page = _fake_pdf_one_page()

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/region.pdf", page=1,
            bbox=(0.0, 0.0, 0.0, 0.0), dpi=72,
            region_bboxes=((72.0, 144.0, 216.0, 288.0),),
        )

    # Region rect drawn → non-white pixels, and no page-search happened.
    assert any(px != (255, 255, 255) for px in out.getdata())
    fake_page.search.assert_not_called()


def test_render_region_bboxes_multiple_rects_all_drawn():
    """Multiple stored regions (e.g. a cross-page chunk) each get a box."""
    pdf_render.render_page_with_bbox.cache_clear()
    fake_pdf, _ = _fake_pdf_one_page()
    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/multi.pdf", page=1, bbox=(0.0, 0.0, 0.0, 0.0), dpi=72,
            region_bboxes=((10.0, 10.0, 50.0, 30.0), (10.0, 200.0, 60.0, 230.0)),
        )
    # Two amber rects (dpi=72 → 1pt=1px): one at y10-30, one at y200-230.
    px = list(out.getdata())
    w = out.width
    top_band = px[: w * 40]
    low_band = px[w * 195: w * 235]
    assert any(p != (255, 255, 255) for p in top_band)
    assert any(p != (255, 255, 255) for p in low_band)


def test_render_terms_wash_still_works():
    """Term washes remain (the only path that still calls page.search)."""
    pdf_render.render_page_with_bbox.cache_clear()
    fake_pdf, fake_page = _fake_pdf_one_page()
    fake_page.search.return_value = [{"chars": [{"x0": 100.0, "top": 100.0, "x1": 300.0, "bottom": 120.0}]}]

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/terms.pdf", page=1,
            bbox=(0.0, 0.0, 0.0, 0.0), dpi=72, terms=("fuel", "landing"),
        )

    assert any(px != (255, 255, 255) for px in out.getdata())
    assert fake_page.search.called


def test_render_with_cited_spans_draws_solid_box_from_quote_search():
    """S43 generation→bbox wiring: the cited quote is located on the page and
    drawn as the solid CITED box."""
    pdf_render.render_page_with_bbox.cache_clear()
    fake_pdf, fake_page = _fake_pdf_one_page()
    fake_page.search.return_value = [
        {"chars": [{"x0": 100.0, "top": 200.0, "x1": 400.0, "bottom": 215.0}]}
    ]

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/span.pdf", page=1,
            bbox=(0.0, 0.0, 0.0, 0.0), dpi=72,
            cited_spans=("the engine lost power",),
        )

    assert fake_page.search.called
    # Solid box drawn in the quote's band → non-white pixels there.
    px = list(out.getdata())
    w = out.width
    band = px[w * 198: w * 217]
    assert any(p != (255, 255, 255) for p in band)


def test_render_cited_spans_skipped_when_draw_bbox_false():
    pdf_render.render_page_with_bbox.cache_clear()
    fake_pdf, fake_page = _fake_pdf_one_page()

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/span-off.pdf", page=1,
            bbox=(0.0, 0.0, 0.0, 0.0), dpi=72, draw_bbox=False,
            cited_spans=("the engine lost power",),
        )

    fake_page.search.assert_not_called()
    assert all(px == (255, 255, 255) for px in out.getdata())


def test_render_page_out_of_range_raises():
    pdf_render.render_page_with_bbox.cache_clear()

    fake_pdf = MagicMock()
    fake_pdf.pages = []
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        with pytest.raises(PdfRenderError, match="out of range"):
            render_page_with_bbox(
                "https://example.test/b.pdf", page=2, bbox=(0.0, 0.0, 10.0, 10.0),
            )


def test_render_is_cached_per_args():
    pdf_render.render_page_with_bbox.cache_clear()

    fake_page_image = MagicMock()
    fake_page_image.original = _white_pil(100, 100)
    fake_page = MagicMock()
    fake_page.to_image.return_value = fake_page_image
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF") as dl, \
         patch("pdfplumber.open", return_value=fake_pdf):
        render_page_with_bbox("u", 1, (0.0, 0.0, 10.0, 10.0))
        render_page_with_bbox("u", 1, (0.0, 0.0, 10.0, 10.0))

    # The cache means the underlying download fires exactly once.
    assert dl.call_count == 1
