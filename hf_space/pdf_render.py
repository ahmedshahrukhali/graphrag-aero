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


# Two highlight tiers, drawn over the same amber hue:
#   CITED — the exact passage the answer quoted: solid fill + bold outline.
#   TERM  — every other on-page occurrence of a query term (incl. the title):
#           a lighter wash + thin outline, so the page reads as "captured"
#           without the cited span getting lost in the crowd.
# Each style is (fill_rgba, outline_rgba, outline_width).
_STYLE_CITED = ((245, 158, 11, 64), (245, 158, 11, 255), 3)
_STYLE_TERM = ((245, 158, 11, 34), (245, 158, 11, 140), 1)
# IMAGE — a figure/raster embedded on the page. We have no image *understanding*
# (no VLM in the 8GB budget), so we just mark it: red outline, no fill (don't
# obscure the figure). Signals "this page carries a captured image here".
_STYLE_IMAGE = ((220, 38, 38, 0), (220, 38, 38, 255), 3)

Style = Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], int]


def _draw_boxes(
    img: Image.Image,
    boxes: list[tuple[tuple[int, int, int, int], Style]],
) -> Image.Image:
    """Draw a list of ``(pixel_bbox, style)`` rectangles on a copy of ``img``.

    Boxes are drawn in list order onto one overlay, so callers should pass the
    lighter TERM washes first and the solid CITED box last (drawn on top).
    """
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)
    for pixel_bbox, (fill, outline, width) in boxes:
        drw.rectangle(pixel_bbox, fill=fill, outline=outline, width=width)
    return Image.alpha_composite(out, overlay).convert("RGB")


# WS-B (region-level grounding): the cited box is no longer located by searching
# the page for the answer's quote (``search_page_bbox`` — deleted). It is drawn
# directly from the chunk's stored ``page_bboxes`` (one rect per page, computed
# once at ingest), which is deterministic and never desyncs. See REINGEST_PLAN
# §4.1. The term-wash below still uses ``page.search`` — it's only a coverage
# tint, not the grounding box.

# A page with 40× "fuel" shouldn't turn into confetti — cap the wash boxes.
_MAX_TERM_BOXES = 25


def search_page_terms(
    page, terms: tuple[str, ...], *, max_boxes: int = _MAX_TERM_BOXES
) -> list[BBox]:
    """Return the bbox of every occurrence of each term in ``terms`` on
    ``page`` (case-insensitive), deduped and capped at ``max_boxes``.

    Unlike :func:`search_page_bbox` (which keeps only the first hit of one
    span), this collects *all* hits — that's what lights up the title and the
    repeated mentions, demonstrating retrieval coverage on the page.
    """
    boxes: list[BBox] = []
    seen: set[tuple[int, int, int, int]] = set()
    for term in terms:
        t = term.strip()
        if not t:
            continue
            
        # Make search robust against newlines/extra spaces in PDF text
        parts = [re.escape(w) for w in t.split()]
        robust_pattern = r'\s+'.join(parts)
        
        try:
            hits = page.search(robust_pattern, regex=True, case=False)
        except Exception:  # noqa: BLE001 — pdfplumber search is best-effort
            continue
        for h in hits:
            line_groups = {}
            for c in h.get("chars", []):
                # group chars by approximate top coordinate (2 points tolerance)
                line_y = round(float(c["top"]) / 2) * 2
                if line_y not in line_groups:
                    line_groups[line_y] = []
                line_groups[line_y].append(c)
                
            for chars in line_groups.values():
                x0 = min(float(c["x0"]) for c in chars)
                top = min(float(c["top"]) for c in chars)
                x1 = max(float(c["x1"]) for c in chars)
                bottom = max(float(c["bottom"]) for c in chars)
                
                bb = (x0, top, x1, bottom)
                key = (round(bb[0]), round(bb[1]), round(bb[2]), round(bb[3]))
                if key in seen:
                    continue
                seen.add(key)
                boxes.append(bb)
                if len(boxes) >= max_boxes:
                    return boxes
    return boxes


def page_image_bboxes(page, *, max_boxes: int = _MAX_TERM_BOXES) -> list[BBox]:
    """Bounding boxes (PDF points, top-origin) of raster images on ``page``.

    pdfplumber exposes ``page.images`` with ``x0/top/x1/bottom`` in the same
    coordinate space ``bbox_to_pixels`` expects. Used to red-box figures so the
    demo shows the image was captured even though we don't interpret it.
    """
    out: list[BBox] = []
    for im in (getattr(page, "images", None) or []):
        try:
            out.append((float(im["x0"]), float(im["top"]), float(im["x1"]), float(im["bottom"])))
        except (KeyError, TypeError, ValueError):
            continue
        if len(out) >= max_boxes:
            break
    return out


def _bbox_is_drawable(bbox: BBox) -> bool:
    """A stored bbox worth drawing — nonzero area and not glued to top-left origin."""
    x_valid = (bbox[2] - bbox[0]) > 0
    y_valid = (bbox[3] - bbox[1]) > 0
    not_origin = bbox[0] > 1.0 or bbox[1] > 1.0
    return x_valid and y_valid and not_origin


@lru_cache(maxsize=64)
def render_page_with_bbox(
    pdf_url: str,
    page: int,
    bbox: BBox,
    *,
    dpi: int = DEFAULT_DPI,
    draw_bbox: bool = True,
    region_bboxes: tuple[BBox, ...] = (),
    terms: tuple[str, ...] = (),
    box_images: bool = False,
) -> Image.Image:
    """Return a PIL Image of ``page`` of the PDF at ``pdf_url``.

    ``page`` is 1-indexed (matches ``RetrievedChunk.page`` from the backend).

    Highlight behaviour (two tiers, see ``_STYLE_CITED`` / ``_STYLE_TERM``):
    - ``draw_bbox=False`` → bare page, no overlay (the UI bbox toggle).
    - ``terms`` given → every on-page occurrence of those query terms gets a
      light wash (lights up the document title + repeated mentions). Drawn under
      the cited box. This is the only path that still calls ``page.search``.
    - ``box_images=True`` → red outline around every raster image (figure) on
      the page. We don't interpret the image; this just marks it as captured.
    - **Cited box (WS-B, region-level):** ``region_bboxes`` — the chunk's stored
      regions for *this* page (PDF points, ``(x0, top, x1, bottom)``) — are drawn
      solid on top. These come from ingest (``page_bboxes``); no page search, no
      desync. If ``region_bboxes`` is empty, fall back to the legacy single
      ``bbox`` (when it's drawable and we're not in image-only mode).

    All args are part of the LRU cache key (tuples required for hashing).

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
            term_boxes = search_page_terms(p, terms) if (draw_bbox and terms) else []
            img_boxes = page_image_bboxes(p) if (draw_bbox and box_images) else []
    except PdfRenderError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PdfRenderError(f"failed to rasterise page {page} of {pdf_url}: {e}") from e

    if not draw_bbox:
        return pil

    draw_list: list[tuple[tuple[int, int, int, int], Style]] = []
    # Term washes first (drawn underneath the cited box).
    for tb in term_boxes:
        draw_list.append((bbox_to_pixels(tb, dpi=dpi), _STYLE_TERM))
    # Red image outlines (figures).
    for ib in img_boxes:
        draw_list.append((bbox_to_pixels(ib, dpi=dpi), _STYLE_IMAGE))

    # Cited regions: the stored page_bboxes for this page, drawn solid on top.
    cited = [rb for rb in region_bboxes if _bbox_is_drawable(rb)]
    if not cited and not box_images and _bbox_is_drawable(bbox):
        cited = [bbox]  # legacy single-rect fallback
    for rb in cited:
        draw_list.append((bbox_to_pixels(rb, dpi=dpi), _STYLE_CITED))

    if not draw_list:
        return pil
    return _draw_boxes(pil, draw_list)
