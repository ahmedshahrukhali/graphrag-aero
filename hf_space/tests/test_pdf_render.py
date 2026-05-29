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
    render_page_with_bbox,
    search_page_bbox,
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


def test_search_page_bbox_returns_first_hit():
    """page.search hit → bbox tuple in PDF points."""
    fake_page = MagicMock()
    fake_page.search.return_value = [
        {"x0": 72.0, "top": 144.0, "x1": 216.0, "bottom": 160.0},
        {"x0": 1.0, "top": 1.0, "x1": 2.0, "bottom": 2.0},
    ]
    bbox = search_page_bbox(fake_page, "the quick brown fox jumps")
    assert bbox == (72.0, 144.0, 216.0, 160.0)
    # Only the first N probe words are used to build the pattern.
    fake_page.search.assert_called_once()


def test_search_page_bbox_no_hit_returns_none():
    fake_page = MagicMock()
    fake_page.search.return_value = []
    assert search_page_bbox(fake_page, "nothing here") is None


def test_search_page_bbox_empty_needle_returns_none():
    fake_page = MagicMock()
    assert search_page_bbox(fake_page, "   ") is None
    fake_page.search.assert_not_called()


def test_search_page_bbox_swallows_search_errors():
    fake_page = MagicMock()
    fake_page.search.side_effect = RuntimeError("pdfplumber blew up")
    assert search_page_bbox(fake_page, "some text") is None


def _fake_pdf_with_searchable_page(search_return):
    """A one-page fake pdf whose page.search returns ``search_return``."""
    fake_page_image = MagicMock()
    fake_page_image.original = _white_pil(800, 1000)
    fake_page = MagicMock()
    fake_page.to_image.return_value = fake_page_image
    fake_page.search.return_value = search_return
    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf
    fake_pdf.__exit__.return_value = False
    return fake_pdf, fake_page


def test_render_with_locate_text_hit_draws_box():
    """locate_text found on page → rectangle drawn at the matched span."""
    pdf_render.render_page_with_bbox.cache_clear()
    fake_pdf, fake_page = _fake_pdf_with_searchable_page(
        [{"x0": 72.0, "top": 144.0, "x1": 216.0, "bottom": 288.0}]
    )

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/hit.pdf", page=1,
            bbox=(0.0, 0.0, 1.0, 1.0), dpi=72, locate_text="cited sentence here",
        )

    fake_page.search.assert_called_once()
    # The matched span was boxed → some non-white pixels.
    assert any(px != (255, 255, 255) for px in out.getdata())


def test_render_with_locate_text_miss_leaves_page_bare():
    """locate_text not found → return the bare page, no misleading box."""
    pdf_render.render_page_with_bbox.cache_clear()
    fake_pdf, fake_page = _fake_pdf_with_searchable_page([])

    with patch.object(pdf_render, "_download_pdf", return_value=b"%PDF\n%%EOF"), \
         patch("pdfplumber.open", return_value=fake_pdf):
        out = render_page_with_bbox(
            "https://example.test/miss.pdf", page=1,
            bbox=(72.0, 144.0, 216.0, 288.0), dpi=72, locate_text="not on this page",
        )

    fake_page.search.assert_called_once()
    # No match → no overlay → every pixel still white.
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
