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


# One PaddleOCR model per language code — built lazily, cached per process.
# Keyed by PaddleOCR's own ``lang`` code ("latin" | "ch" | "chinese_cht"), not
# our corpus lang, because Simplified (CAAC) and Traditional (TTSB) are both
# corpus-lang "zh" yet need different weights.
_ocr_by_lang: dict[str, Any] = {}

# Angle classification corrects skew on scanned pages — the Chinese corpus is
# deliberately scanned, so enable it there. The latin TC/TSB PDFs aren't rotated.
_NEEDS_ANGLE_CLS = frozenset({"ch", "chinese_cht"})


def paddle_lang(lang: str, source: str) -> str:
    """Map (corpus lang, source) → the PaddleOCR ``lang`` code.

    - en / fr            → "latin"  (multilingual Latin model)
    - zh + caac          → "ch"     (Simplified Chinese)
    - zh + ttsb          → "chinese_cht" (Traditional Chinese)
    - zh + anything else → "ch"     (default Simplified)
    """
    if lang == "zh":
        return "chinese_cht" if source == "ttsb" else "ch"
    return "latin"


def _get_ocr(ocr_lang: str = "latin") -> Any:
    cached = _ocr_by_lang.get(ocr_lang)
    if cached is not None:
        return cached
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as e:
        raise OcrNotAvailable(
            "paddleocr is not installed in this image. Install the [ocr] extra."
        ) from e
    model = PaddleOCR(
        use_angle_cls=ocr_lang in _NEEDS_ANGLE_CLS,
        lang=ocr_lang,
        show_log=False,
    )
    _ocr_by_lang[ocr_lang] = model
    return model


OCR_RESOLUTION = 200  # DPI used when rendering pages for PaddleOCR
_PTS_PER_PIXEL = 72.0 / OCR_RESOLUTION  # pixels → PDF points (top-left origin)


def ocr_page(page: Any, page_no: int, ocr_lang: str = "latin") -> PageExtract:
    """OCR a pdfplumber page (assumed image-only) and return a :class:`PageExtract`.

    ``ocr_lang`` is a PaddleOCR ``lang`` code ("latin" | "ch" | "chinese_cht");
    derive it with :func:`paddle_lang` from the doc's (lang, source).

    PaddleOCR returns one entry per detected line:
        [ [bbox4points], (text, confidence) ]
    We collapse the polygon to a rectangular bbox and synthesize one Char per
    *line* (not per glyph) — the downstream chunker only uses chars for bbox
    aggregation, so per-line is plenty.

    Bbox coordinates are stored in **PDF point space** (top-left origin, 72 pts/inch)
    so they are consistent with pdfplumber's text-extraction coordinates and can be
    passed directly to ``hf_space.pdf_render.bbox_to_pixels``.
    """
    ocr = _get_ocr(ocr_lang)
    # Render the page to an image (pdfplumber API). Resolution=200 balances
    # OCR accuracy with memory; tune later if needed.
    img = page.to_image(resolution=OCR_RESOLUTION)
    pil = img.original  # PIL.Image
    # Angle classification is configured on the model; pass cls accordingly so
    # scanned Chinese pages get deskewed.
    result = ocr.ocr(pil, cls=ocr_lang in _NEEDS_ANGLE_CLS)
    # PaddleOCR 2.x returns a list of pages; we passed one page so result[0].
    lines = result[0] if result else []
    chars: list[Char] = []
    text_parts: list[str] = []
    for entry in lines:
        polygon, (text, _conf) = entry
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        # PaddleOCR coords are pixels in the rendered image. Convert to PDF
        # point space so all bboxes across text and OCR pages share a coordinate
        # system that pdf_render.bbox_to_pixels understands.
        x0  = min(xs) * _PTS_PER_PIXEL
        x1  = max(xs) * _PTS_PER_PIXEL
        top = min(ys) * _PTS_PER_PIXEL
        bottom = max(ys) * _PTS_PER_PIXEL
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
