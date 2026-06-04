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
import queue
import sys
import threading
from concurrent.futures import Future
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


def render_page_to_array(page: Any) -> np.ndarray:
    """Render a pdfplumber page to a BGR uint8 array for PaddleOCR. CPU-only."""
    img = page.to_image(resolution=OCR_RESOLUTION)
    pil = img.original.convert("RGB")
    return np.asarray(pil)[:, :, ::-1].copy()


def _parse_ocr_result(res: Any, page_no: int) -> PageExtract:
    """Convert one PaddleOCR 3.x result dict into a PageExtract."""
    texts = list(res["rec_texts"]) if res else []
    polys = list(res["rec_polys"]) if res else []
    chars: list[Char] = []
    text_parts: list[str] = []
    for text, polygon in zip(texts, polys):
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        x0     = min(xs) * _PTS_PER_PIXEL
        x1     = max(xs) * _PTS_PER_PIXEL
        top    = min(ys) * _PTS_PER_PIXEL
        bottom = max(ys) * _PTS_PER_PIXEL
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
    return PageExtract(page=page_no, text="\n".join(text_parts), chars=chars, image_only=False)


def ocr_page(page: Any, page_no: int, ocr_lang: str = "latin") -> PageExtract:
    """OCR one pdfplumber page synchronously (no batching).

    Prefer :class:`OcrBatchQueue` for multi-worker runs — it groups pages from
    concurrent workers into a single ``predict()`` call, keeping the GPU busy.
    """
    arr = render_page_to_array(page)
    ocr = _get_ocr(ocr_lang)
    results = ocr.predict(arr)
    return _parse_ocr_result(results[0] if results else None, page_no)


class OcrBatchQueue:
    """Routes rendered page arrays to a single GPU thread for batched inference.

    Workers call :meth:`submit` with an already-rendered numpy array (CPU work
    stays in the worker thread).  The OCR thread collects up to ``batch_size``
    arrays per language model and fires them in one ``predict()`` call, keeping
    the GPU busy without concurrent VRAM contention.

    Usage::

        q = OcrBatchQueue(batch_size=8)
        extract = q.submit(arr, page_no, ocr_lang)   # blocks until result ready
        q.shutdown()                                  # after all workers finish
    """

    def __init__(self, batch_size: int = 8, drain_timeout: float = 0.2) -> None:
        self._batch_size = batch_size
        self._drain_timeout = drain_timeout
        self._q: queue.Queue = queue.Queue()
        self._shutdown_flag = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="ocr-batch")
        self._thread.start()

    def submit(self, arr: np.ndarray, page_no: int, ocr_lang: str) -> PageExtract:
        """Submit one rendered page for GPU OCR. Blocks until the result is ready."""
        fut: Future[PageExtract] = Future()
        self._q.put((fut, arr, page_no, ocr_lang))
        return fut.result()

    def shutdown(self, wait: bool = True) -> None:
        """Signal the OCR thread to drain remaining items and exit."""
        self._shutdown_flag.set()
        if wait:
            self._thread.join()

    # ── internal ──────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            items: list = []
            try:
                item = self._q.get(timeout=self._drain_timeout)
                items.append(item)
            except queue.Empty:
                if self._shutdown_flag.is_set() and self._q.empty():
                    break
                continue
            # Drain up to batch_size without blocking so we don't wait forever.
            while len(items) < self._batch_size:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break
            self._process(items)
        # Final drain: flush anything that arrived just before shutdown.
        while not self._q.empty():
            items = []
            while not self._q.empty() and len(items) < self._batch_size:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if items:
                self._process(items)

    def _process(self, items: list) -> None:
        from collections import defaultdict
        by_lang: dict[str, list] = defaultdict(list)
        for fut, arr, page_no, ocr_lang in items:
            by_lang[ocr_lang].append((fut, arr, page_no))

        for ocr_lang, group in by_lang.items():
            arrs = [arr for _, arr, _ in group]
            try:
                ocr = _get_ocr(ocr_lang)
                # PaddleOCR 3.x predict() accepts a list of arrays and returns
                # one result dict per image — same as single-image but batched.
                results = ocr.predict(arrs)
                for (fut, _, page_no), res in zip(group, results):
                    try:
                        fut.set_result(_parse_ocr_result(res, page_no))
                    except Exception as exc:  # noqa: BLE001
                        fut.set_exception(exc)
            except Exception as exc:  # noqa: BLE001
                for fut, _, _ in group:
                    if not fut.done():
                        fut.set_exception(exc)
