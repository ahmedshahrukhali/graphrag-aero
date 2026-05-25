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
from functools import lru_cache
from typing import Tuple

import httpx
from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)

# pdfplumber renders at this DPI when you pass ``resolution=DPI``. PDF
# user-space is 72 points per inch; the rasterised image has
# ``DPI/72`` pixels per point, which is the conversion we apply to bbox
# coordinates so the rectangle lands in the right place.
DEFAULT_DPI = 120
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


@lru_cache(maxsize=64)
def render_page_with_bbox(
    pdf_url: str,
    page: int,
    bbox: BBox,
    *,
    dpi: int = DEFAULT_DPI,
) -> Image.Image:
    """Return a PIL Image of ``page`` of the PDF at ``pdf_url`` with the
    chunk's bbox highlighted.

    ``page`` is 1-indexed (matches ``RetrievedChunk.page`` from the
    backend). ``bbox`` is the pdfplumber tuple ``(x0, top, x1, bottom)``
    in PDF points.

    Raises :class:`PdfRenderError` on download / parse / out-of-range
    page failures.
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
    except PdfRenderError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PdfRenderError(f"failed to rasterise page {page} of {pdf_url}: {e}") from e

    return _draw_bbox(pil, bbox_to_pixels(bbox, dpi=dpi))
