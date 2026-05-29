"""Render a PDF page with the chunk's bbox highlighted.

The Gradio UI in app.py uses this to turn each ``RetrievedChunk`` into a
thumbnail image — the "multimodal output" element of the P8 demo.

Two layers:
- ``bbox_to_pixels`` is a pure function (input: pdfplumber bbox in PDF
  points, rendering DPI; output: integer pixel coords). Trivially
  unit-testable.
- ``render_page_with_bbox`` does the I/O: downloads the PDF via httpx,
  opens with pdfplumber, rasterises the page, draws a rectangle.

Cached with ``functools.lru_cache`` — the same chunk is rendered each
time the user clicks through the citations.
"""
from __future__ import annotations

import io
import logging
import re
from functools import lru_cache
from typing import Tuple

import httpx
from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)

# pdfplumber renders at this DPI when you pass ``resolution=DPI``. PDF
# user-space is 72 points per inch; the rasterised image has
# ``DPI/72`` pixels per point, which is the conversion we apply to bbox
# coordinates so the rectangle lands in the right place.
DEFAULT_DPI = 100
PDF_USER_SPACE_DPI = 72

BBox = Tuple[float, float, float, float]


class PdfRenderError(RuntimeError):
    pass


def bbox_to_pixels(bbox: BBox, *, dpi: int = DEFAULT_DPI) -> tuple[int, int, int, int]:
    """Convert a pdfplumber bbox (PDF points, top-origin) to pixel coords
    at the given render DPI. Returns ``(left, top, right, bottom)``."""
    x0, top, x1, bottom = bbox
    scale = dpi / PDF_USER_SPACE_DPI
    left   = int(round(min(x0, x1) * scale))
    right  = int(round(max(x0, x1) * scale))
    top_px = int(round(min(top, bottom) * scale))
    bot_px = int(round(max(top, bottom) * scale))
    return left, top_px, right, bot_px


def _download_pdf(url: str, *, timeout: float = 30.0) -> bytes:
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise PdfRenderError(f"failed to fetch {url}: {e}") from e
    return r.content


def _draw_bbox(img: Image.Image, pixel_bbox: tuple[int, int, int, int]) -> Image.Image:
    """Draw a translucent rectangle + outline on a copy of ``img``."""
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)
    drw.rectangle(pixel_bbox, fill=(245, 158, 11, 64), outline=(245, 158, 11, 255), width=3)
    return Image.alpha_composite(out, overlay).convert("RGB")


# First N words of the cited quote are enough to localise it on the page;
# searching the full (often multi-line) quote verbatim misses too easily.
_LOCATE_PROBE_WORDS = 12


def search_page_bbox(page, needle: str) -> BBox | None:
    """Locate ``needle`` on a pdfplumber ``page`` and return its bbox in PDF
    points ``(x0, top, x1, bottom)``, or ``None`` if not found.

    Whitespace between words is matched flexibly (``\\s+``) so line wraps in
    the PDF don't defeat the match. Only the first ``_LOCATE_PROBE_WORDS``
    words of ``needle`` are used — enough to pin the location without
    requiring the whole (often wrapped) quote to match verbatim.
    """
    words = [w for w in re.split(r"\s+", needle.strip()) if w]
    if not words:
        return None
    probe = words[:_LOCATE_PROBE_WORDS]
    pattern = r"\s+".join(re.escape(w) for w in probe)
    try:
        hits = page.search(pattern, regex=True, case=False)
    except Exception:  # noqa: BLE001  — pdfplumber search is best-effort
        return None
    if not hits:
        return None
    h = hits[0]
    return (float(h["x0"]), float(h["top"]), float(h["x1"]), float(h["bottom"]))


@lru_cache(maxsize=64)
def render_page_with_bbox(
    pdf_url: str,
    page: int,
    bbox: BBox,
    *,
    dpi: int = DEFAULT_DPI,
    draw_bbox: bool = True,
    locate_text: str | None = None,
) -> Image.Image:
    """Return a PIL Image of ``page`` of the PDF at ``pdf_url``.

    ``page`` is 1-indexed (matches ``RetrievedChunk.page`` from the backend).

    Highlight behaviour:
    - ``draw_bbox=False`` → bare page, no overlay (the UI bbox toggle).
    - ``locate_text`` given → search the page for that text and box the
      *matched span*. If the text isn't found, return the bare page (no
      misleading box). This is the citation-anchored path the UI uses; the
      coarse stored ``bbox`` is intentionally ignored here.
    - ``locate_text=None`` (legacy) → draw the stored ``bbox``.

    All args are part of the LRU cache key.

    Raises :class:`PdfRenderError` on download / parse / out-of-range failures.
    """
    import pdfplumber  # lazy: heavy import

    raw = _download_pdf(pdf_url)
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            if page < 1 or page > len(pdf.pages):
                raise PdfRenderError(f"page {page} out of range (1..{len(pdf.pages)})")
            p = pdf.pages[page - 1]
            page_image = p.to_image(resolution=dpi)
            pil = page_image.original.copy()
            located = search_page_bbox(p, locate_text) if (draw_bbox and locate_text) else None
    except PdfRenderError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PdfRenderError(f"failed to rasterise page {page} of {pdf_url}: {e}") from e

    if not draw_bbox:
        return pil
    if locate_text is not None:
        # Citation-anchored: draw only if we actually located the span.
        if located is None:
            return pil
        return _draw_bbox(pil, bbox_to_pixels(located, dpi=dpi))
    # Legacy path: draw the stored bbox.
    return _draw_bbox(pil, bbox_to_pixels(bbox, dpi=dpi))
