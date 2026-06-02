"""PaddleOCR fallback for image-only pages.

Imported lazily — the unit tests must run without paddleocr or paddlepaddle
present. The first call to :func:`ocr_page` triggers the import; if paddleocr
isn't installed it raises a clean ``OcrNotAvailable``.

Output shape matches :class:`ingestion.processing.pdf.PageExtract` so the
chunker treats OCR-derived chars uniformly.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

from .pdf import Char, PageExtract

logger = logging.getLogger(__name__)


class OcrNotAvailable(RuntimeError):
    """Raised when paddleocr is requested but not importable."""


# One PaddleOCR model per language code — built lazily, cached per process.
# Keyed by PaddleOCR's own ``lang`` code ("latin" | "ch" | "chinese_cht"), not
# our corpus lang, because Simplified (CAAC) and Traditional (TTSB) are both
# corpus-lang "zh" yet need different weights.
_ocr_by_lang: dict[str, Any] = {}

# Textline-orientation classification corrects skew on scanned pages — the
# Chinese corpus is deliberately scanned, so enable it there. The latin TC/TSB
# PDFs aren't rotated. (PaddleOCR 3.x renamed ``use_angle_cls`` → this.)
_NEEDS_ANGLE_CLS = frozenset({"ch", "chinese_cht"})


def _ocr_device() -> str:
    """PaddleOCR 3.x device string, chosen safely.

    Precedence: explicit ``PADDLE_OCR_DEVICE`` env → else auto-detect a usable
    CUDA GPU → else ``cpu``. paddlepaddle-gpu does NOT silently fall back to CPU
    (``device="gpu"`` errors with no GPU), so we detect rather than assume — a
    CPU-only host (or CI) gets ``cpu`` automatically. The ``import paddle`` is
    guarded: in unit tests paddle isn't installed, so we land on ``cpu``."""
    override = os.environ.get("PADDLE_OCR_DEVICE")
    if override:
        return _announce(override, " (PADDLE_OCR_DEVICE override)")
    try:
        import paddle  # type: ignore
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return _announce("gpu", "")
    except Exception as e:  # noqa: BLE001 — probe failure → CPU, but say so
        return _announce("cpu", f" (GPU probe failed: {e})")
    return _announce("cpu", " (no usable GPU detected)")


def _announce(device: str, reason: str) -> str:
    """Surface the OCR device choice. paddleocr/paddlex reconfigures logging on
    import and swallows this module's logger, so we ALSO print to stderr — the
    choice must be visible in unattended WS-F runs (no silent CPU fallback)."""
    msg = f"OCR device: {device}{reason}"
    logger.info(msg)
    print(msg, file=sys.stderr, flush=True)
    return device


def paddle_lang(lang: str, source: str) -> str:
    """Map (corpus lang, source) → the PaddleOCR ``lang`` code.

    - en                 → "en"     (English PP-OCRv5 model)
    - fr                 → "fr"     (French PP-OCRv5 model — handles diacritics)
    - zh + caac          → "ch"     (Simplified Chinese)
    - zh + ttsb          → "chinese_cht" (Traditional Chinese)
    - zh + anything else → "ch"     (default Simplified)

    NB: PaddleOCR 3.x (PP-OCRv5) dropped the 2.x shared ``latin`` model — there
    is no model for ``lang="latin"`` (it raises). Use the per-language ``en``/``fr``
    models instead (verified available in this image; more accurate than the old
    shared model). Unknown langs default to ``en``.
    """
    if lang == "zh":
        return "chinese_cht" if source == "ttsb" else "ch"
    if lang == "fr":
        return "fr"
    return "en"


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
    # PaddleOCR 3.x constructor: ``use_angle_cls``/``show_log`` are gone; textline
    # orientation replaces angle-cls. Disable the doc-orientation + unwarp stages
    # (extra model downloads we don't need for our already-upright page renders).
    model = PaddleOCR(
        lang=ocr_lang,
        use_textline_orientation=ocr_lang in _NEEDS_ANGLE_CLS,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # PP-OCRv5 (3.x default) detects more regions than v2.x, incl. scan
        # artifacts/stamps at low confidence. Drop sub-threshold lines so chunks
        # carry body text, not bleed-through noise.
        text_rec_score_thresh=0.5,
        device=_ocr_device(),
    )
    _ocr_by_lang[ocr_lang] = model
    return model


OCR_RESOLUTION = 200  # DPI used when rendering pages for PaddleOCR
_PTS_PER_PIXEL = 72.0 / OCR_RESOLUTION  # pixels → PDF points (top-left origin)


def ocr_page(page: Any, page_no: int, ocr_lang: str = "latin") -> PageExtract:
    """OCR a pdfplumber page (assumed image-only) and return a :class:`PageExtract`.

    ``ocr_lang`` is a PaddleOCR ``lang`` code ("latin" | "ch" | "chinese_cht");
    derive it with :func:`paddle_lang` from the doc's (lang, source).

    PaddleOCR 3.x ``predict()`` returns a list of one ``OCRResult`` (dict-like)
    per input image. We passed one image, so ``results[0]`` carries parallel
    lists: ``rec_texts`` (one string per detected line) and ``rec_polys`` (the
    line's 4-point polygon in pixel space). We collapse each polygon to a
    rectangle and synthesize one Char *per glyph* (not per line) so the chunker's
    glyph-by-glyph bbox alignment works.

    Bbox coordinates are stored in **PDF point space** (top-left origin, 72 pts/inch)
    so they are consistent with pdfplumber's text-extraction coordinates and can be
    passed directly to ``hf_space.pdf_render.bbox_to_pixels``.
    """
    ocr = _get_ocr(ocr_lang)
    # Render the page to an image (pdfplumber API). Resolution=200 balances
    # OCR accuracy with memory; tune later if needed.
    img = page.to_image(resolution=OCR_RESOLUTION)
    # ``.original`` is a PIL.Image; PaddleOCR wants an ndarray and treats it as
    # BGR (OpenCV convention). Convert RGB→BGR into a contiguous array.
    pil = img.original.convert("RGB")
    arr = np.asarray(pil)[:, :, ::-1].copy()
    results = ocr.predict(arr)
    res = results[0] if results else None
    texts = list(res["rec_texts"]) if res else []
    polys = list(res["rec_polys"]) if res else []
    chars: list[Char] = []
    text_parts: list[str] = []
    for text, polygon in zip(texts, polys):
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        # PaddleOCR coords are pixels in the rendered image. Convert to PDF
        # point space so all bboxes across text and OCR pages share a coordinate
        # system that pdf_render.bbox_to_pixels understands.
        x0  = min(xs) * _PTS_PER_PIXEL
        x1  = max(xs) * _PTS_PER_PIXEL
        top = min(ys) * _PTS_PER_PIXEL
        bottom = max(ys) * _PTS_PER_PIXEL
        # Emit ONE Char per glyph (not per line): the chunker aligns chunk text to
        # char bboxes glyph-by-glyph (``ch == next_char.text``), so a per-line Char
        # whose .text is the whole line never matches and the chunk bbox is lost.
        # Region-level grounding only needs the union, so subdivide the line's
        # width evenly across its glyphs and share the line's y-extent.
        n = len(text)
        for j, glyph in enumerate(text):
            gx0 = x0 + (x1 - x0) * j / n if n else x0
            gx1 = x0 + (x1 - x0) * (j + 1) / n if n else x1
            chars.append(Char(
                text=glyph,
                x0=float(gx0), x1=float(gx1),
                top=float(top), bottom=float(bottom),
                size=float(bottom - top),
                page=page_no,
            ))
        text_parts.append(text)
    page_text = "\n".join(text_parts)
    return PageExtract(page=page_no, text=page_text, chars=chars, image_only=False)
