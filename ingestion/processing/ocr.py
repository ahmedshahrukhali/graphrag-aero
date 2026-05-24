"""PaddleOCR fallback for image-only pages.

Imported lazily — the unit tests must run without paddleocr or paddlepaddle
present. The first call to :func:`ocr_page` triggers the import; if paddleocr
isn't installed it raises a clean ``OcrNotAvailable``.

Output shape matches :class:`ingestion.processing.pdf.PageExtract` so the
chunker treats OCR-derived chars uniformly.
"""
from __future__ import annotations

import logging
from typing import Any

from .pdf import Char, PageExtract

logger = logging.getLogger(__name__)


class OcrNotAvailable(RuntimeError):
    """Raised when paddleocr is requested but not importable."""


_ocr_singleton: Any | None = None  # module-level cache; lazy-built


def _get_ocr() -> Any:
    global _ocr_singleton
    if _ocr_singleton is not None:
        return _ocr_singleton
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as e:
        raise OcrNotAvailable(
            "paddleocr is not installed in this image. Install the [ocr] extra."
        ) from e
    # Multilingual model handles EN + FR. Disable angle classifier — TC/TSB
    # PDFs aren't rotated. Initialise once per process.
    _ocr_singleton = PaddleOCR(use_angle_cls=False, lang="latin", show_log=False)
    return _ocr_singleton


def ocr_page(page: Any, page_no: int) -> PageExtract:
    """OCR a pdfplumber page (assumed image-only) and return a :class:`PageExtract`.

    PaddleOCR returns one entry per detected line:
        [ [bbox4points], (text, confidence) ]
    We collapse the polygon to a rectangular bbox and synthesize one Char per
    *line* (not per glyph) — the downstream chunker only uses chars for bbox
    aggregation, so per-line is plenty.
    """
    ocr = _get_ocr()
    # Render the page to an image (pdfplumber API). Resolution=200 balances
    # OCR accuracy with memory; tune later if needed.
    img = page.to_image(resolution=200)
    pil = img.original  # PIL.Image
    result = ocr.ocr(pil, cls=False)
    # PaddleOCR 2.x returns a list of pages; we passed one page so result[0].
    lines = result[0] if result else []
    chars: list[Char] = []
    text_parts: list[str] = []
    for entry in lines:
        polygon, (text, _conf) = entry
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x0, x1 = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        # One Char per line; .text holds the whole line so chunker can locate it.
        chars.append(Char(
            text=text,
            x0=float(x0), x1=float(x1),
            top=float(top), bottom=float(bottom),
            size=float(bottom - top),
            page=page_no,
        ))
        text_parts.append(text)
    page_text = "\n".join(text_parts)
    return PageExtract(page=page_no, text=page_text, chars=chars, image_only=False)
