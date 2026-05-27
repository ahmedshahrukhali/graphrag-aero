"""Verify stored chunk bboxes against the source PDFs.

Two verification strategies, both run per chunk:

1. **pdfplumber crop text** (primary, always runs): crop the stored bbox on the
   PDF page using the same pdfplumber coordinate system the chunker used, extract
   text, compute character-level similarity vs the stored chunk text.  This is the
   ground-truth check — if the bbox is correct the cropped region should reproduce
   most of the chunk text.

2. **Image crop** (optional, ``--save-crops``): rasterise the page, draw a
   highlighted rectangle over the bbox, save a PNG.  Use for manual visual
   inspection — the rectangle should land on the chunk text.

Usage
-----
    # Quick sample: 50 chunks, all sources
    python -m eval.bbox_eval

    # 200 chunks, TSB only, save crop images for manual inspection
    python -m eval.bbox_eval --n 200 --source tsb --save-crops crops/

    # JSON output for CI / dashboard
    python -m eval.bbox_eval --json > bbox_eval.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── constants ────────────────────────────────────────────────────────────────

EVAL_DPI = 150                 # render resolution for saved crop PNGs
MIN_BBOX_AREA_PTS = 100.0      # skip bboxes smaller than ~10×10 pt (degenerate)
MIN_CHUNK_TEXT_LEN = 30        # skip very short chunks (page separators / headers)
SIMILARITY_THRESHOLD = 0.40    # treat pdfplumber crop similarity ≥ this as a "hit"

# ─── data shapes ─────────────────────────────────────────────────────────────

@dataclass
class BBoxResult:
    doc_id: str
    lang: str
    page: int
    bbox: list[float]
    chunk_text_preview: str      # first 80 chars of stored text
    ocr_text_preview: str        # first 80 chars of OCR result
    similarity: float            # difflib SequenceMatcher ratio
    hit: bool                    # similarity >= SIMILARITY_THRESHOLD
    error: str | None = None     # non-None if crop/OCR failed


@dataclass
class BBoxReport:
    n_sampled: int
    n_ok: int                    # no error
    n_hit: int                   # similarity >= threshold
    mean_similarity: float
    median_similarity: float
    threshold: float
    dpi: int
    results: list[BBoxResult]

# ─── helpers ─────────────────────────────────────────────────────────────────

def _pdf_path(doc_id: str, lang: str, corpus_root: Path) -> Path | None:
    """Reconstruct the PDF path from doc_id and lang.

    doc_id format: "{source}/{stem}"  (e.g. "tsb/a13q0098", "tc/AC_507-001")
    Corpus layout:  corpus_root/{lang}/{source}/{stem}.pdf
    """
    parts = doc_id.split("/", 1)
    if len(parts) != 2:
        return None
    source, stem = parts
    p = corpus_root / lang / source / f"{stem}.pdf"
    return p if p.exists() else None


def _bbox_area(bbox: list[float]) -> float:
    """Area in PDF point² of bbox [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_to_pixels(bbox: list[float], dpi: int) -> tuple[int, int, int, int]:
    """PDF points (top-left origin) → pixel coords at given DPI."""
    x0, top, x1, bottom = bbox
    scale = dpi / 72.0
    return (
        max(0, int(round(min(x0, x1) * scale))),
        max(0, int(round(min(top, bottom) * scale))),
        max(0, int(round(max(x0, x1) * scale))),
        max(0, int(round(max(top, bottom) * scale))),
    )


def _text_similarity(a: str, b: str) -> float:
    """Character-level SequenceMatcher ratio normalised to [0, 1]."""
    a = a.strip().lower()
    b = b.strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _crop_text(pdf_path: Path, page: int, bbox: list[float]) -> str:
    """Use pdfplumber to crop bbox on *page* and extract text.

    This is the ground-truth check: same coordinate system and extraction
    method as the chunker, so a correct bbox will reproduce most of the
    chunk text.  Returns empty string for image-only pages (no selectable
    text in the cropped region).
    """
    import pdfplumber  # type: ignore

    x0, top, x1, bottom = bbox
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page < 1 or page > len(pdf.pages):
            raise ValueError(f"page {page} out of range (1..{len(pdf.pages)})")
        p = pdf.pages[page - 1]
        cropped = p.crop((x0, top, x1, bottom))
        return cropped.extract_text() or ""

# ─── core eval ───────────────────────────────────────────────────────────────

def render_annotated_page(pdf_path: Path, page: int, bbox: list[float], dpi: int) -> Any:
    """Render *page* as a PIL image with the bbox highlighted.

    Returns the full page image with an amber rectangle drawn over the bbox
    region — useful for visual inspection via ``--save-crops``.
    *page* is 1-indexed; *bbox* is [x0, top, x1, bottom] in PDF point space.
    """
    from PIL import Image, ImageDraw  # type: ignore
    import pdfplumber  # type: ignore

    with pdfplumber.open(str(pdf_path)) as pdf:
        if page < 1 or page > len(pdf.pages):
            raise ValueError(f"page {page} out of range (1..{len(pdf.pages)})")
        p = pdf.pages[page - 1]
        page_img = p.to_image(resolution=dpi)
        pil = page_img.original.copy()

    left, top_px, right, bot_px = _bbox_to_pixels(bbox, dpi)
    w, h = pil.size
    left  = max(0, min(left, w))
    right = max(0, min(right, w))
    top_px  = max(0, min(top_px, h))
    bot_px  = max(0, min(bot_px, h))

    out = pil.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)
    drw.rectangle((left, top_px, right, bot_px),
                  fill=(245, 158, 11, 64), outline=(245, 158, 11, 255), width=3)
    return Image.alpha_composite(out, overlay).convert("RGB")


def eval_chunk(
    chunk: dict,
    corpus_root: Path,
    *,
    dpi: int = EVAL_DPI,
    save_crops: Path | None = None,
) -> BBoxResult:
    """Evaluate one chunk: crop its bbox from the source PDF and OCR-compare."""
    doc_id = chunk["doc_id"]
    lang   = chunk["lang"]
    page   = chunk["page"]
    bbox   = chunk["bbox"]
    text   = chunk["text"]

    base = BBoxResult(
        doc_id=doc_id,
        lang=lang,
        page=page,
        bbox=bbox,
        chunk_text_preview=text[:80],
        ocr_text_preview="",
        similarity=0.0,
        hit=False,
    )

    pdf_path = _pdf_path(doc_id, lang, corpus_root)
    if pdf_path is None:
        base.error = f"PDF not found for doc_id={doc_id} lang={lang}"
        return base

    # Primary check: pdfplumber crop text vs stored chunk text.
    try:
        crop_text = _crop_text(pdf_path, page, bbox)
    except Exception as e:
        base.error = f"crop_text failed: {e}"
        return base

    sim = _text_similarity(text, crop_text)
    base.ocr_text_preview = crop_text[:80]
    base.similarity = round(sim, 4)
    base.hit = sim >= SIMILARITY_THRESHOLD

    # Optional: save annotated page image for visual inspection.
    if save_crops is not None:
        try:
            annotated = render_annotated_page(pdf_path, page, bbox, dpi)
            save_crops.mkdir(parents=True, exist_ok=True)
            safe = doc_id.replace("/", "_")
            annotated.save(save_crops / f"{safe}_p{page}.png")
        except Exception as e:
            logger.warning("save_crops failed for %s p%d: %s", doc_id, page, e)

    return base


def load_sample_chunks(
    chunks_root: Path,
    n: int,
    source: str | None,
    *,
    seed: int = 42,
) -> list[dict]:
    """Load *n* random chunks with valid, non-degenerate bboxes."""
    rng = random.Random(seed)
    pattern = "**/*.jsonl"
    all_files = list(chunks_root.glob(pattern))
    rng.shuffle(all_files)

    candidates: list[dict] = []
    for jl_path in all_files:
        # Filter by source if requested.
        parts = jl_path.parts
        if source and source not in parts:
            continue
        try:
            with jl_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    bbox = rec.get("bbox") or []
                    if len(bbox) != 4:
                        continue
                    if _bbox_area(bbox) < MIN_BBOX_AREA_PTS:
                        continue
                    if len(rec.get("text", "")) < MIN_CHUNK_TEXT_LEN:
                        continue
                    if rec.get("page", 0) < 1:
                        continue
                    candidates.append(rec)
        except OSError:
            continue
        if len(candidates) >= n * 10:
            break  # have enough to sample from

    rng.shuffle(candidates)
    return candidates[:n]


# ─── report ──────────────────────────────────────────────────────────────────

def _make_report(results: list[BBoxResult], dpi: int, threshold: float) -> BBoxReport:
    ok = [r for r in results if r.error is None]
    hits = [r for r in ok if r.hit]
    sims = [r.similarity for r in ok]
    sims_sorted = sorted(sims)
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    mid = len(sims_sorted) // 2
    median_sim = sims_sorted[mid] if sims_sorted else 0.0
    return BBoxReport(
        n_sampled=len(results),
        n_ok=len(ok),
        n_hit=len(hits),
        mean_similarity=round(mean_sim, 4),
        median_similarity=round(median_sim, 4),
        threshold=threshold,
        dpi=dpi,
        results=results,
    )


def run_eval(
    chunks_root: Path,
    corpus_root: Path,
    n: int,
    source: str | None,
    *,
    dpi: int = EVAL_DPI,
    save_crops: Path | None = None,
    seed: int = 42,
) -> BBoxReport:
    chunks = load_sample_chunks(chunks_root, n, source, seed=seed)
    if not chunks:
        raise RuntimeError(f"No eligible chunks found under {chunks_root}")
    logger.info("evaluating %d chunks (requested %d, found %d eligible)", len(chunks), n, len(chunks))
    results = []
    for i, chunk in enumerate(chunks):
        r = eval_chunk(chunk, corpus_root, dpi=dpi, save_crops=save_crops)
        status = f"hit={r.hit} sim={r.similarity:.2f}" if r.error is None else f"err={r.error}"
        logger.info("[%d/%d] %s p%d — %s", i + 1, len(chunks), chunk["doc_id"], chunk["page"], status)
        results.append(r)
    return _make_report(results, dpi, SIMILARITY_THRESHOLD)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify chunk bboxes by crop + OCR.")
    p.add_argument("--chunks", type=Path, default=Path("data/chunks"),
                   help="Root of chunk JSONL files.")
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"),
                   help="Root of source PDFs.")
    p.add_argument("--n", type=int, default=50, help="Number of chunks to sample.")
    p.add_argument("--source", choices=["tsb", "tc"], default=None,
                   help="Restrict to one corpus source.")
    p.add_argument("--dpi", type=int, default=EVAL_DPI,
                   help="Render DPI for page rasterisation.")
    p.add_argument("--save-crops", type=Path, default=None, dest="save_crops",
                   help="Directory to save crop PNG images for manual inspection.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true", help="Print full JSON report to stdout.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report = run_eval(
        args.chunks, args.corpus,
        n=args.n,
        source=args.source,
        dpi=args.dpi,
        save_crops=args.save_crops,
        seed=args.seed,
    )

    if args.json:
        # Convert dataclasses to dicts for JSON serialisation.
        out = asdict(report)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        ok_pct = 100.0 * report.n_ok / report.n_sampled if report.n_sampled else 0
        hit_pct = 100.0 * report.n_hit / report.n_ok if report.n_ok else 0
        print(f"\nBBox eval — {report.n_sampled} chunks sampled")
        print(f"  errors          : {report.n_sampled - report.n_ok}")
        print(f"  mean similarity : {report.mean_similarity:.3f}")
        print(f"  median sim      : {report.median_similarity:.3f}")
        print(f"  hits (≥{report.threshold:.2f})    : {report.n_hit}/{report.n_ok} ({hit_pct:.1f}%)")
        print()
        worst = sorted(
            [r for r in report.results if r.error is None],
            key=lambda r: r.similarity,
        )[:5]
        if worst:
            print("Worst 5 by similarity:")
            for r in worst:
                print(f"  {r.doc_id} p{r.page}  sim={r.similarity:.3f}")
                print(f"    chunk : {r.chunk_text_preview!r}")
                print(f"    ocr   : {r.ocr_text_preview!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
