"""CLI: walk ``data/corpus/`` → emit ``data/chunks/{doc_id}.jsonl``.

Idempotent: a doc whose output JSONL mtime is newer than the source PDF is
skipped. Cross-document chunk dedup applies (first occurrence wins).

Run::

    python -m ingestion.processing.run --in data/corpus --out data/chunks
    python -m ingestion.processing.run --in data/corpus --out data/chunks --source tsb --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

from . import pdf as pdf_mod
from .chunk import Chunk, Tokenizer, chunk_pages
from .curation import ADMITTED, Admission, CurationManifest, admit
from .dedup import Dedup
from .doc_id import DocRef, doc_ref_for_path
from .ocr import ocr_page, paddle_lang

logger = logging.getLogger(__name__)

#: How often (in docs processed) to flush the curation manifest to disk.
#: Bounds loss on crash; small enough to be effectively continuous, large
#: enough that the atomic-rename cost stays trivial over a multi-thousand-doc
#: run. Test code patches this to 1 for determinism.
INCREMENTAL_MANIFEST_EVERY: int = 25


# ─── tokenizer factory ───────────────────────────────────────────────────────

def default_tokenizer() -> Tokenizer:
    """BGE-M3 tokenizer; downloads/caches via ``tokenizers`` on first call."""
    from tokenizers import Tokenizer as _Tok
    return _Tok.from_pretrained("BAAI/bge-m3")


# ─── per-doc extraction (text + OCR fallback) ────────────────────────────────

def extract_pages_with_ocr(path: Path, ocr_lang: str = "latin") -> list[pdf_mod.PageExtract]:
    """Open ``path`` once; use pdfplumber per page, OCR-fall-back on image-only pages.

    ``ocr_lang`` is the PaddleOCR ``lang`` code (see :func:`paddle_lang`) — selects
    the Latin vs Simplified vs Traditional Chinese model for the OCR fallback.
    """
    out: list[pdf_mod.PageExtract] = []
    with pdf_mod.open_pdf(path) as pdfdoc:
        for i, page in enumerate(pdfdoc.pages, start=1):
            extracted = pdf_mod.extract_page(page, i)
            if extracted.image_only:
                logger.info("ocr fallback: %s page %d (%s)", path.name, i, ocr_lang)
                try:
                    extracted = ocr_page(page, i, ocr_lang)
                except Exception as e:
                    logger.warning("ocr failed for %s page %d: %s", path.name, i, e)
            out.append(extracted)
    return out


# ─── jsonl writing ───────────────────────────────────────────────────────────

def _chunk_to_record(ref: DocRef, c: Chunk) -> dict:
    return {
        "doc_id": ref.doc_id,
        "source_url": ref.source_url,
        "section_title": c.section_title,
        "page": c.page,
        "bbox": list(c.bbox),
        # WS-0 region-level grounding: one [page, x0, top, x1, bottom] per page.
        "page_bboxes": [list(pb) for pb in c.page_bboxes],
        "corpus": ref.corpus,
        "kind": c.kind,
        "chunk_hash": c.chunk_hash,
        "lang": ref.lang,
        "text": c.text,
    }


def write_jsonl(dest: Path, records: Iterable[dict]) -> int:
    """Atomic write — `<dest>.part` then rename. Returns count written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    n = 0
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")
                n += 1
        tmp.replace(dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return n


# ─── per-doc orchestration ───────────────────────────────────────────────────

def _is_fresh(out_path: Path, src_path: Path) -> bool:
    if not out_path.exists():
        return False
    return out_path.stat().st_mtime >= src_path.stat().st_mtime


def process_doc(
    src: Path,
    out_root: Path,
    tokenizer: Tokenizer,
    dedup: Dedup,
    *,
    force: bool = False,
    curate: bool = False,
    manifest: CurationManifest | None = None,
) -> int:
    """Process one PDF. Returns number of chunks written (0 if skipped/fresh/rejected)."""
    ref = doc_ref_for_path(src)
    # Output mirrors corpus layout but under out_root and with .jsonl: {lang}/{source}/{stem}.jsonl
    dest = out_root / ref.lang / ref.source / f"{ref.stem}.jsonl"
    if not force and _is_fresh(dest, src):
        # Resume-safety: a fresh dest means a prior run wrote it, which (since
        # rejected docs return before write_jsonl below) implies admission.
        # Carry-record without re-extracting so the manifest reflects the full
        # admitted set after resumes, not just docs processed *this* run.
        if curate and manifest is not None:
            manifest.record(ref, ADMITTED)
        logger.info("skip (fresh): %s", dest)
        return 0

    pages = extract_pages_with_ocr(src, paddle_lang(ref.lang, ref.source))
    chunks = chunk_pages(pages, tokenizer)

    if curate:
        result: Admission = admit(ref, chunks)
        if manifest is not None:
            manifest.record(ref, result)
        if not result.admitted:
            logger.info("curate reject (%s): %s", result.reason.value, src.name)
            return 0

    kept = [c for c in chunks if dedup.is_new(c.chunk_hash)]
    records = (_chunk_to_record(ref, c) for c in kept)
    n = write_jsonl(dest, records)
    logger.info(
        "wrote %d chunks (%d before dedup) -> %s",
        n, len(chunks), dest,
    )
    return n


# ─── manifest write (atomic, used both incrementally and at end-of-run) ─────

def _write_manifest(manifest: CurationManifest, path: Path) -> None:
    """Atomic write: `<path>.part` then rename. Crash mid-write never leaves
    a half-JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


# ─── CLI ─────────────────────────────────────────────────────────────────────

def iter_corpus_pdfs(corpus_root: Path, source: str | None) -> list[Path]:
    """All PDFs under ``corpus_root/{en,fr,zh}/{tsb,tc,ttsb,caac}/*.pdf``, deterministic order.

    Missing ``{lang}/{source}`` dirs are skipped, so the EN/FR × tsb/tc and
    ZH × ttsb/caac layouts coexist without enumerating impossible combos.
    """
    sources = ("tsb", "tc", "ttsb", "caac") if source in (None, "all") else (source,)
    paths: list[Path] = []
    for lang in ("en", "fr", "zh"):
        for src in sources:
            d = corpus_root / lang / src
            if not d.is_dir():
                continue
            paths.extend(sorted(d.glob("*.pdf")))
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chunk corpus PDFs into JSONL.")
    p.add_argument("--in", dest="in_root", type=Path, default=Path("data/corpus"))
    p.add_argument("--out", dest="out_root", type=Path, default=Path("data/chunks"))
    p.add_argument("--source", choices=["tsb", "tc", "ttsb", "caac", "all"], default="all")
    p.add_argument("--limit", type=int, default=None,
                   help="Max docs to process — handy for sample runs.")
    p.add_argument("--force", action="store_true",
                   help="Reprocess even if the destination is fresh.")
    p.add_argument("--curate", action="store_true",
                   help="Apply admission criteria (curation.py); write curation_manifest.json.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # pdfminer and PIL are chatty at DEBUG; we don't need their internals.
    for noisy in ("pdfminer", "pdfminer.pdfdocument", "pdfminer.pdfpage",
                  "pdfminer.pdfinterp", "pdfminer.cmapdb", "pdfminer.converter",
                  "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    paths = iter_corpus_pdfs(args.in_root, args.source)
    if args.limit is not None:
        paths = paths[: args.limit]
    logger.info("ingesting %d PDFs from %s", len(paths), args.in_root)

    tokenizer = default_tokenizer()
    dedup = Dedup()
    manifest = CurationManifest() if args.curate else None
    manifest_path = args.out_root / "curation_manifest.json"
    total_chunks = 0
    for i, src in enumerate(paths, start=1):
        try:
            total_chunks += process_doc(
                src, args.out_root, tokenizer, dedup,
                force=args.force, curate=args.curate, manifest=manifest,
            )
        except Exception as e:
            logger.exception("failed: %s: %s", src, e)
        # Flush the manifest periodically so an interrupted overnight run
        # (SIGINT / OOM / power) doesn't lose its tally.
        if manifest is not None and i % INCREMENTAL_MANIFEST_EVERY == 0:
            _write_manifest(manifest, manifest_path)
    logger.info(
        "done: %d chunks across %d docs (%d unique hashes)",
        total_chunks, len(paths), len(dedup),
    )
    if manifest is not None:
        _write_manifest(manifest, manifest_path)
        logger.info("curation manifest -> %s", manifest_path)
        warning = manifest.balance_warning()
        if warning:
            logger.warning(warning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
