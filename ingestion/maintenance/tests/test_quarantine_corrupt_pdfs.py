"""Tests for the quarantine-corrupt-PDFs CLI.

Avoids the reportlab dep by monkeypatching the open probe; the broken-vs-OK
decision is what matters, not real PDF bytes.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ingestion.maintenance import quarantine_corrupt_pdfs as qcp


def _make_corpus(tmp: Path) -> Path:
    corpus = tmp / "corpus"
    for sub in ("en/tc", "en/tsb", "fr/tc"):
        (corpus / sub).mkdir(parents=True)
    (corpus / "en" / "tc" / "good_a.pdf").write_bytes(b"%PDF-1.7\n")
    (corpus / "en" / "tc" / "broken_b.pdf").write_bytes(b"<html>oops</html>")
    (corpus / "en" / "tsb" / "good_c.pdf").write_bytes(b"%PDF-1.7\n")
    (corpus / "fr" / "tc" / "broken_d.pdf").write_bytes(b"")
    return corpus


def _fake_open_probe(path: Path) -> Exception | None:
    """Pretend pdfplumber.open raises iff the filename starts with 'broken_'."""
    if path.name.startswith("broken_"):
        return RuntimeError("No /Root object")
    return None


def test_dry_run_reports_but_does_not_move(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    quarantine = tmp_path / "quarantine"

    broken = qcp.scan_and_quarantine(
        corpus, quarantine, apply=False, open_probe=_fake_open_probe,
    )
    broken_names = sorted(p.name for p, _ in broken)
    assert broken_names == ["broken_b.pdf", "broken_d.pdf"]
    # Nothing moved.
    assert (corpus / "en" / "tc" / "broken_b.pdf").exists()
    assert (corpus / "fr" / "tc" / "broken_d.pdf").exists()
    assert not quarantine.exists()


def test_apply_moves_and_writes_manifest(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    quarantine = tmp_path / "quarantine"

    broken = qcp.scan_and_quarantine(
        corpus, quarantine, apply=True, open_probe=_fake_open_probe,
    )
    assert len(broken) == 2

    # Good files stayed put.
    assert (corpus / "en" / "tc" / "good_a.pdf").exists()
    assert (corpus / "en" / "tsb" / "good_c.pdf").exists()

    # Broken files moved into the mirrored layout.
    assert not (corpus / "en" / "tc" / "broken_b.pdf").exists()
    assert not (corpus / "fr" / "tc" / "broken_d.pdf").exists()
    assert (quarantine / "en" / "tc" / "broken_b.pdf").exists()
    assert (quarantine / "fr" / "tc" / "broken_d.pdf").exists()

    # Manifest CSV has a header + one row per moved file.
    manifest = quarantine / "manifest.csv"
    rows = list(csv.reader(manifest.open(encoding="utf-8")))
    assert rows[0] == [
        "original_path", "quarantine_path", "error_class",
        "error_message", "quarantined_at",
    ]
    assert len(rows) == 3  # header + 2
    moved_names = sorted(Path(r[1]).name for r in rows[1:])
    assert moved_names == ["broken_b.pdf", "broken_d.pdf"]
    error_classes = {r[2] for r in rows[1:]}
    assert error_classes == {"RuntimeError"}
