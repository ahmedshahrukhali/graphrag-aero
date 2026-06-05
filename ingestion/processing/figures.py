"""WS-C — Figure extraction: detect, crop, and caption figures via Qwen3-VL-8B.

Pipeline per PDF page
---------------------
1. Detect figures — ``page.images`` gives pdfplumber image-dicts with bboxes.
2. Render page to PIL at 150 DPI; crop to each figure's bbox.
3. Caption crop via a ``FigureCaptioner`` (real: ``QwenVLCaptioner``; tests: mock).
4. Emit ``FigureRecord`` objects — the intermediate; used both for Neo4j (:Figure
   nodes) and as ``kind=figure`` chunk JSONL so figures become retrievable.

Key constraints from ``docs/ws_c_qwenvl_findings.md``
------------------------------------------------------
- Feed CROPS, not full pages (full page overflows vision-token budget → empty output).
- Strip the model's ``thinking`` field; use ``response`` only.
- ctx 8192 is ample for a crop.

The VL model is loaded once per run and unloaded after the last doc — VRAM discipline.
It must NOT be loaded concurrently with PaddleOCR or the text generator.
"""
from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Render resolution for figure crops.  150 DPI balances quality vs token count.
_DPI = 150
_SCALE = _DPI / 72.0  # PDF points → pixel scale factor

# Model default — override via env var FIGURE_VL_MODEL.
# Qwen2.5-VL-7B-Instruct is the current public Qwen-VL family ~8B checkpoint.
_DEFAULT_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Prompt used for every figure crop.  Matches the bake-off prompt from ws_c_qwenvl_findings.md.
_CAPTION_PROMPT = (
    "This is a figure from an aviation accident investigation report or advisory circular. "
    "Caption it in 1-2 sentences, then transcribe any visible text, labels, and numbers. "
    "Respond ONLY with:\nCaption: <caption>\nTranscribed: <text>"
)


# ─── FigureRecord ─────────────────────────────────────────────────────────────

@dataclass
class FigureRecord:
    """One figure extracted from a document page."""
    doc_id: str                           # e.g. "tsb/a13q0098"
    page: int                             # 1-indexed page number
    bbox: list[float]                     # [x0, top, x1, bottom] in PDF points
    caption: str                          # VLM-generated caption sentence(s)
    ocr_text: str = ""                    # transcribed labels / text from the figure

    @property
    def figure_id(self) -> str:
        """Stable Neo4j node id: keyed by (doc_id, page, bbox)."""
        bbox_str = ",".join(f"{v:.2f}" for v in self.bbox)
        h = hashlib.sha256(f"{self.doc_id}:{self.page}:{bbox_str}".encode()).hexdigest()[:12]
        return f"{self.doc_id}:fig:{self.page}:{h}"

    @property
    def chunk_hash(self) -> str:
        """Stable chunk hash for figure-as-chunk dedup — keyed by (doc_id, page, bbox)."""
        bbox_str = ",".join(f"{v:.2f}" for v in self.bbox)
        return hashlib.sha256(
            f"figure:{self.doc_id}:{self.page}:{bbox_str}".encode()
        ).hexdigest()


# ─── Captioner protocol ───────────────────────────────────────────────────────

@runtime_checkable
class FigureCaptioner(Protocol):
    """Anything with a ``caption`` method qualifies — makes mocking trivial."""

    def caption(self, crop) -> dict:
        """Return ``{"caption": str, "ocr_text": str}`` for *crop* (PIL Image)."""
        ...


# ─── Real captioner: Qwen-VL via HF transformers ─────────────────────────────

class QwenVLCaptioner:
    """Load Qwen3-VL-8B once; caption crops on demand; unload on context exit.

    Usage::

        with QwenVLCaptioner() as cap:
            result = cap.caption(pil_crop)
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get("FIGURE_VL_MODEL", _DEFAULT_VL_MODEL)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("loading VL model %s (this may take a minute)…", self.model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        logger.info("VL model loaded")

    def _unload(self) -> None:
        import gc
        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VL model unloaded")

    def __enter__(self) -> "QwenVLCaptioner":
        self._load()
        return self

    def __exit__(self, *_exc) -> None:
        self._unload()

    def caption(self, crop) -> dict:
        """Run one VL inference on *crop* (PIL Image). Returns {caption, ocr_text}."""
        if self._model is None or self._processor is None:
            raise RuntimeError("QwenVLCaptioner must be used as a context manager")

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": _CAPTION_PROMPT},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # qwen_vl_utils may not be installed — fall back to passing image directly.
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs = [crop]
            video_inputs = None

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(next(self._model.parameters()).device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Decode only the generated tokens (not the prompt)
        gen_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        raw: str = self._processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return _parse_response(raw)


# ─── Response parsing ─────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    """Extract caption + ocr_text from the model's response.

    Handles two formats:
      - Structured: ``Caption: ... Transcribed: ...``
      - Fallback: entire text treated as caption.
    """
    raw = raw.strip()
    caption = ""
    ocr_text = ""

    if "Caption:" in raw:
        after = raw.split("Caption:", 1)[1]
        if "Transcribed:" in after:
            cap_part, ocr_part = after.split("Transcribed:", 1)
        else:
            cap_part, ocr_part = after, ""
        caption = cap_part.strip()
        ocr_text = ocr_part.strip()
    else:
        caption = raw

    return {"caption": caption, "ocr_text": ocr_text}


# ─── Page rendering + figure detection ───────────────────────────────────────

def _render_page_pil(page):
    """Render a pdfplumber page to a PIL Image at ``_DPI`` DPI."""
    try:
        from PIL import Image as PILImage
        import io
    except ImportError as e:
        raise ImportError("Pillow is required for figure extraction") from e

    img = page.to_image(resolution=_DPI).original
    return img


def _bbox_pdf_to_pixel(bbox: list[float], page_height: float) -> tuple[int, int, int, int]:
    """Convert PDF-space bbox [x0, top, x1, bottom] → pixel (left, upper, right, lower).

    pdfplumber uses top-down coordinates (top=0 at page top), matching PIL.
    """
    x0, top, x1, bottom = bbox
    left = int(x0 * _SCALE)
    upper = int(top * _SCALE)
    right = int(x1 * _SCALE)
    lower = int(bottom * _SCALE)
    return left, upper, right, lower


def _figure_bboxes(page) -> list[list[float]]:
    """Return [x0, top, x1, bottom] bboxes for images on *page* (pdfplumber page object)."""
    bboxes: list[list[float]] = []
    for img in (page.images or []):
        x0 = float(img.get("x0", 0))
        top = float(img.get("top", 0))
        x1 = float(img.get("x1", 0))
        bottom = float(img.get("bottom", 0))
        # Skip tiny or degenerate bboxes (e.g. 1-pixel rule lines).
        if (x1 - x0) < 20 or (bottom - top) < 20:
            continue
        bboxes.append([x0, top, x1, bottom])
    return bboxes


# ─── Main extraction entry points ─────────────────────────────────────────────

def extract_figures_from_page(
    page,
    page_no: int,
    doc_id: str,
    captioner: FigureCaptioner,
) -> list[FigureRecord]:
    """Detect + caption all figures on one pdfplumber page.

    Returns a list of :class:`FigureRecord` — empty if the page has no figures
    or captioning fails for all of them.
    """
    bboxes = _figure_bboxes(page)
    if not bboxes:
        return []

    page_img = _render_page_pil(page)
    page_height = float(page.height)
    records: list[FigureRecord] = []

    for bbox in bboxes:
        try:
            pixel_bbox = _bbox_pdf_to_pixel(bbox, page_height)
            crop = page_img.crop(pixel_bbox)
            result = captioner.caption(crop)
            rec = FigureRecord(
                doc_id=doc_id,
                page=page_no,
                bbox=bbox,
                caption=result.get("caption", ""),
                ocr_text=result.get("ocr_text", ""),
            )
            records.append(rec)
            logger.debug("figure captioned: %s p.%d bbox=%s", doc_id, page_no, bbox)
        except Exception as exc:
            logger.warning(
                "figure caption failed: %s p.%d bbox=%s: %s",
                doc_id, page_no, bbox, exc,
            )
    return records


def extract_figures(
    pdf_path: Path,
    doc_id: str,
    captioner: FigureCaptioner,
) -> list[FigureRecord]:
    """Extract and caption all figures from *pdf_path*. Returns FigureRecord list."""
    import pdfplumber

    records: list[FigureRecord] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                page_records = extract_figures_from_page(page, page_no, doc_id, captioner)
                records.extend(page_records)
    except Exception as exc:
        logger.warning("figure extraction failed for %s: %s", pdf_path, exc)
    logger.info("extracted %d figures from %s", len(records), pdf_path.name)
    return records


# ─── Figure → chunk dict (for JSONL output) ───────────────────────────────────

def figure_to_chunk_dict(
    ref,  # DocRef
    fig: FigureRecord,
    source_url: str | None = None,
) -> dict:
    """Convert a FigureRecord to a chunk dict suitable for JSONL output.

    The figure caption + transcribed text becomes retrievable as a ``kind=figure``
    chunk. The chunk text merges caption and OCR text so the embedder sees both.
    """
    text_parts = [f"[Figure p.{fig.page}] {fig.caption}"]
    if fig.ocr_text:
        text_parts.append(fig.ocr_text)
    text = "\n\n".join(text_parts)

    return {
        "doc_id": ref.doc_id,
        "source_url": source_url or ref.source_url,
        "section_title": "[figure]",
        "page": fig.page,
        "bbox": fig.bbox,
        "page_bboxes": [[float(fig.page)] + fig.bbox],
        "corpus": ref.corpus,
        "kind": "figure",
        "chunk_hash": fig.chunk_hash,
        "lang": ref.lang,
        "text": text,
    }
