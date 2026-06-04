"""CLI: scan ``data/corpus/`` for unopenable PDFs and quarantine them.

The S22 WS-F pilot found 15 TC PDFs that pdfminer rejects with "No /Root
object" (not actually PDFs — likely truncated downloads or HTML error
pages renamed to .pdf). The processing CLI catches them per-doc so a run
survives, but they re-fail on every resume and noise the error log.

This tool moves them once into a quarantine tree that mirrors the corpus
layout, with a CSV manifest, so they can be eyeballed and either re-downloaded
or kept aside permanently.

Default is ``--dry-run`` (read-only). Pass ``--apply`` to actually move.

Run::

    python -m ingestion.maintenance.quarantine_corrupt_pdfs --dry-run
    python -m ingestion.maintenance.quarantine_corrupt_pdfs --apply
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Callable

from ingestion.processing import pdf as pdf_mod
from ingestion.processing.run import iter_corpus_pdfs

logger = logging.getLogger(__name__)


# A PDF that successfully opens but has zero pages is functionally broken too;
# the existing ingest treats it as "no chunks" which curation then rejects as
# EMPTY. We don't quarantine those here — only opens that *raise*.
def _try_open(path: Path) -> Exception | None:
    """Return the exception if ``pdfplumber.open`` raises, else None."""
    try:
        with pdf_mod.open_pdf(path):
            pass
    except Exception as e:  # broad on purpose — pdfminer/pdfplumber raise many
        return e
    return None


def _mirrored_dest(src: Path, corpus_root: Path, quarantine_root: Path) -> Path:
    """Mirror ``corpus/{lang}/{source}/x.pdf`` → ``quarantine/{lang}/{source}/x.pdf``."""
    rel = src.relative_to(corpus_root)
    return quarantine_root / rel


def scan_and_quarantine(
    corpus_root: Path,
    quarantine_root: Path,
    *,
    apply: bool,
    open_probe: Callable[[Path], Exception | None] = _try_open,
) -> list[tuple[Path, Exception]]:
    """Walk ``corpus_root``; return list of (src, error) for unopenable PDFs.

    If ``apply`` is true, move each broken PDF into ``quarantine_root`` and
    append a row to ``quarantine_root/manifest.csv``. Otherwise read-only.
    """
    broken: list[tuple[Path, Exception]] = []
    paths = iter_corpus_pdfs(corpus_root, source="all")
    for src in paths:
        err = open_probe(src)
        if err is None:
            continue
        broken.append((src, err))
        logger.info("broken: %s  (%s: %s)", src, type(err).__name__, err)

    if apply and broken:
        manifest_path = quarantine_root / "manifest.csv"
        new_file = not manifest_path.exists()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(
                    ["original_path", "quarantine_path", "error_class",
                     "error_message", "quarantined_at"]
                )
            for src, err in broken:
                dest = _mirrored_dest(src, corpus_root, quarantine_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dest)
                writer.writerow([
                    str(src), str(dest), type(err).__name__, str(err),
                    _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                ])
    return broken


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quarantine unopenable PDFs from the corpus.")
    p.add_argument("--in", dest="in_root", type=Path, default=Path("data/corpus"))
    p.add_argument("--quarantine", dest="q_root", type=Path,
                   default=Path("data/corpus_quarantine"))
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True,
                       help="Report broken PDFs without moving (default).")
    group.add_argument("--apply", action="store_true",
                       help="Move broken PDFs into the quarantine tree.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("pdfminer", "pdfminer.pdfdocument", "pdfminer.pdfpage",
                  "pdfminer.pdfinterp", "pdfminer.cmapdb", "pdfminer.converter",
                  "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    apply = bool(args.apply)
    broken = scan_and_quarantine(args.in_root, args.q_root, apply=apply)
    logger.info(
        "%s: %d broken PDFs %s",
        "APPLY" if apply else "DRY-RUN",
        len(broken),
        "moved" if apply else "found (re-run with --apply to move)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
