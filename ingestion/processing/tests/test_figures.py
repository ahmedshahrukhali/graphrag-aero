"""Tests for ingestion/processing/figures.py — all offline, no model weights."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestion.processing.doc_id import DocRef
from ingestion.processing.figures import (
    FigureCaptioner,
    FigureRecord,
    QwenVLCaptioner,
    _bbox_pdf_to_pixel,
    _figure_bboxes,
    _parse_response,
    extract_figures,
    extract_figures_from_page,
    figure_to_chunk_dict,
)


# ─── FakeCaptioner (implements the FigureCaptioner protocol) ─────────────────

class FakeCaptioner:
    """Deterministic stub — returns a fixed caption/ocr_text."""

    def __init__(self, caption: str = "Test figure caption.", ocr_text: str = "LABEL A"):
        self._caption = caption
        self._ocr_text = ocr_text
        self.call_count = 0

    def caption(self, crop) -> dict:
        self.call_count += 1
        return {"caption": self._caption, "ocr_text": self._ocr_text}


# ─── FigureCaptioner protocol conformance ─────────────────────────────────────

def test_fake_captioner_satisfies_protocol():
    assert isinstance(FakeCaptioner(), FigureCaptioner)


# ─── _parse_response ──────────────────────────────────────────────────────────

def test_parse_response_structured():
    raw = "Caption: Aircraft fuel gauge.\nTranscribed: WING TANKS 130 GAL"
    result = _parse_response(raw)
    assert result["caption"] == "Aircraft fuel gauge."
    assert result["ocr_text"] == "WING TANKS 130 GAL"


def test_parse_response_caption_only():
    raw = "Caption: A diagram showing the engine layout."
    result = _parse_response(raw)
    assert "engine layout" in result["caption"]
    assert result["ocr_text"] == ""


def test_parse_response_unstructured_fallback():
    raw = "Some free text with no structured markers."
    result = _parse_response(raw)
    assert result["caption"] == raw
    assert result["ocr_text"] == ""


def test_parse_response_strips_whitespace():
    raw = "Caption:   Leading space caption.  \nTranscribed:   123 MPH  "
    result = _parse_response(raw)
    assert result["caption"] == "Leading space caption."
    assert result["ocr_text"] == "123 MPH"


# ─── FigureRecord ─────────────────────────────────────────────────────────────

def test_figure_record_figure_id_stable():
    fig = FigureRecord(doc_id="tsb/a13q0098", page=3, bbox=[10.0, 20.0, 200.0, 150.0],
                       caption="fuel gauge")
    fid1 = fig.figure_id
    fid2 = fig.figure_id
    assert fid1 == fid2
    assert fid1.startswith("tsb/a13q0098:fig:3:")


def test_figure_record_figure_id_differs_by_bbox():
    fig1 = FigureRecord(doc_id="tsb/abc", page=1, bbox=[0.0, 0.0, 100.0, 100.0], caption="x")
    fig2 = FigureRecord(doc_id="tsb/abc", page=1, bbox=[0.0, 0.0, 100.0, 200.0], caption="x")
    assert fig1.figure_id != fig2.figure_id


def test_figure_record_chunk_hash_stable_and_different_from_figure_id():
    fig = FigureRecord(doc_id="tsb/abc", page=2, bbox=[1.0, 2.0, 3.0, 4.0], caption="c")
    assert fig.chunk_hash != fig.figure_id
    assert fig.chunk_hash == fig.chunk_hash  # stable


def test_figure_record_different_docs_different_ids():
    fig1 = FigureRecord(doc_id="tsb/a001", page=1, bbox=[0, 0, 100, 100], caption="x")
    fig2 = FigureRecord(doc_id="tsb/a002", page=1, bbox=[0, 0, 100, 100], caption="x")
    assert fig1.figure_id != fig2.figure_id


# ─── _bbox_pdf_to_pixel ───────────────────────────────────────────────────────

def test_bbox_pdf_to_pixel_basic():
    # At 150 DPI, scale factor = 150/72 ≈ 2.083
    left, upper, right, lower = _bbox_pdf_to_pixel([72.0, 72.0, 144.0, 144.0], 612.0)
    assert left == 150
    assert upper == 150
    assert right == 300
    assert lower == 300


def test_bbox_pdf_to_pixel_zero():
    result = _bbox_pdf_to_pixel([0.0, 0.0, 0.0, 0.0], 792.0)
    assert result == (0, 0, 0, 0)


# ─── _figure_bboxes ───────────────────────────────────────────────────────────

def _fake_page(images):
    page = MagicMock()
    page.images = images
    return page


def test_figure_bboxes_returns_valid():
    images = [{"x0": 50.0, "top": 100.0, "x1": 300.0, "bottom": 250.0}]
    page = _fake_page(images)
    bboxes = _figure_bboxes(page)
    assert len(bboxes) == 1
    assert bboxes[0] == [50.0, 100.0, 300.0, 250.0]


def test_figure_bboxes_skips_tiny():
    # Width < 20 pt → skip
    images = [{"x0": 0.0, "top": 0.0, "x1": 10.0, "bottom": 200.0}]
    page = _fake_page(images)
    assert _figure_bboxes(page) == []


def test_figure_bboxes_skips_flat():
    # Height < 20 pt → skip
    images = [{"x0": 0.0, "top": 0.0, "x1": 200.0, "bottom": 10.0}]
    page = _fake_page(images)
    assert _figure_bboxes(page) == []


def test_figure_bboxes_empty_page():
    page = _fake_page([])
    assert _figure_bboxes(page) == []


def test_figure_bboxes_multiple():
    images = [
        {"x0": 10.0, "top": 20.0, "x1": 200.0, "bottom": 180.0},
        {"x0": 210.0, "top": 20.0, "x1": 400.0, "bottom": 180.0},
    ]
    page = _fake_page(images)
    bboxes = _figure_bboxes(page)
    assert len(bboxes) == 2


# ─── extract_figures_from_page ────────────────────────────────────────────────

def _make_pil_image(w=200, h=200):
    """Create a tiny real PIL Image for cropping."""
    from PIL import Image
    return Image.new("RGB", (w, h), color=(128, 64, 32))


def _page_with_image(bbox=None):
    """Build a mock pdfplumber page that has one figure."""
    if bbox is None:
        bbox = {"x0": 50.0, "top": 100.0, "x1": 300.0, "bottom": 250.0}
    page = MagicMock()
    page.images = [bbox]
    page.height = 792.0
    # to_image().original returns a PIL Image
    page_img = _make_pil_image(600, 900)
    page.to_image.return_value.original = page_img
    return page


def test_extract_figures_from_page_calls_captioner():
    cap = FakeCaptioner("A fuel gauge.", "WING TANKS 130 GAL")
    page = _page_with_image()
    records = extract_figures_from_page(page, page_no=3, doc_id="tsb/a13q0098", captioner=cap)
    assert len(records) == 1
    assert cap.call_count == 1
    rec = records[0]
    assert rec.doc_id == "tsb/a13q0098"
    assert rec.page == 3
    assert rec.caption == "A fuel gauge."
    assert rec.ocr_text == "WING TANKS 130 GAL"


def test_extract_figures_from_page_empty_when_no_images():
    cap = FakeCaptioner()
    page = _fake_page([])
    page.height = 792.0
    records = extract_figures_from_page(page, page_no=1, doc_id="tsb/x", captioner=cap)
    assert records == []
    assert cap.call_count == 0


def test_extract_figures_from_page_skips_tiny_images():
    cap = FakeCaptioner()
    page = _page_with_image({"x0": 0.0, "top": 0.0, "x1": 5.0, "bottom": 5.0})
    page.height = 792.0
    records = extract_figures_from_page(page, page_no=1, doc_id="tsb/x", captioner=cap)
    assert records == []
    assert cap.call_count == 0


def test_extract_figures_from_page_captioner_exception_is_skipped(caplog):
    class FailCaptioner:
        def caption(self, crop):
            raise RuntimeError("model error")

    page = _page_with_image()
    records = extract_figures_from_page(page, page_no=1, doc_id="tsb/x", captioner=FailCaptioner())
    assert records == []


def test_extract_figures_from_page_multiple_figures():
    cap = FakeCaptioner("caption", "OCR")
    page = MagicMock()
    page.images = [
        {"x0": 10.0, "top": 10.0, "x1": 200.0, "bottom": 200.0},
        {"x0": 210.0, "top": 10.0, "x1": 400.0, "bottom": 200.0},
    ]
    page.height = 792.0
    page.to_image.return_value.original = _make_pil_image(600, 900)
    records = extract_figures_from_page(page, page_no=2, doc_id="tsb/a001", captioner=cap)
    assert len(records) == 2
    assert cap.call_count == 2


# ─── extract_figures (full PDF) ───────────────────────────────────────────────

def test_extract_figures_handles_missing_file():
    cap = FakeCaptioner()
    records = extract_figures(Path("/nonexistent/file.pdf"), "tsb/xxx", cap)
    assert records == []


def test_extract_figures_via_pdfplumber_mock(tmp_path):
    """Full integration path with pdfplumber mocked out."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake")

    mock_page = _page_with_image()
    mock_page.images = [{"x0": 50.0, "top": 100.0, "x1": 300.0, "bottom": 250.0}]
    mock_page.height = 792.0

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)

    cap = FakeCaptioner("A figure caption.", "TEXT ABC")

    with patch("pdfplumber.open", return_value=mock_pdf):
        records = extract_figures(pdf_path, "tsb/a001", cap)

    assert len(records) == 1
    assert records[0].caption == "A figure caption."
    assert records[0].ocr_text == "TEXT ABC"
    assert records[0].page == 1


# ─── figure_to_chunk_dict ─────────────────────────────────────────────────────

def _make_ref(doc_id="tsb/a13q0098", lang="en", source="tsb"):
    ref = MagicMock(spec=DocRef)
    ref.doc_id = doc_id
    ref.lang = lang
    ref.source = source
    ref.corpus = source
    ref.source_url = f"https://example.test/{doc_id}.pdf"
    return ref


def test_figure_to_chunk_dict_kind_figure():
    ref = _make_ref()
    fig = FigureRecord(doc_id="tsb/a13q0098", page=3,
                       bbox=[10.0, 20.0, 200.0, 150.0],
                       caption="Fuel gauge panel.", ocr_text="WING TANKS 130 GAL")
    chunk = figure_to_chunk_dict(ref, fig)
    assert chunk["kind"] == "figure"
    assert chunk["section_title"] == "[figure]"
    assert chunk["page"] == 3
    assert chunk["bbox"] == [10.0, 20.0, 200.0, 150.0]
    assert chunk["corpus"] == "tsb"
    assert chunk["lang"] == "en"
    assert chunk["doc_id"] == "tsb/a13q0098"


def test_figure_to_chunk_dict_text_contains_caption_and_ocr():
    ref = _make_ref()
    fig = FigureRecord(doc_id="tsb/a13q0098", page=5,
                       bbox=[0.0, 0.0, 100.0, 100.0],
                       caption="Engine schematic.", ocr_text="LH ENG  RH ENG")
    chunk = figure_to_chunk_dict(ref, fig)
    assert "Engine schematic." in chunk["text"]
    assert "LH ENG" in chunk["text"]
    assert "[Figure p.5]" in chunk["text"]


def test_figure_to_chunk_dict_no_ocr_text():
    ref = _make_ref()
    fig = FigureRecord(doc_id="tsb/a13q0098", page=1,
                       bbox=[0.0, 0.0, 100.0, 100.0],
                       caption="Damage photo.", ocr_text="")
    chunk = figure_to_chunk_dict(ref, fig)
    assert "Damage photo." in chunk["text"]
    # No trailing double-newline when ocr_text is empty
    assert not chunk["text"].endswith("\n\n")


def test_figure_to_chunk_dict_page_bboxes_format():
    ref = _make_ref()
    fig = FigureRecord(doc_id="tsb/a13q0098", page=7,
                       bbox=[5.0, 10.0, 200.0, 300.0], caption="c")
    chunk = figure_to_chunk_dict(ref, fig)
    assert chunk["page_bboxes"] == [[7.0, 5.0, 10.0, 200.0, 300.0]]


def test_figure_to_chunk_dict_chunk_hash_is_stable():
    ref = _make_ref()
    fig = FigureRecord(doc_id="tsb/a13q0098", page=3,
                       bbox=[10.0, 20.0, 200.0, 150.0], caption="x")
    c1 = figure_to_chunk_dict(ref, fig)
    c2 = figure_to_chunk_dict(ref, fig)
    assert c1["chunk_hash"] == c2["chunk_hash"]


# ─── Integration with run.extract_figures_for_doc ─────────────────────────────

def test_extract_figures_for_doc_writes_jsonl(tmp_path):
    from ingestion.processing import run as run_mod

    corpus = tmp_path / "corpus" / "en" / "tsb"
    corpus.mkdir(parents=True)
    pdf = corpus / "a001.pdf"
    pdf.write_bytes(b"fake")

    out = tmp_path / "chunks"

    cap = FakeCaptioner("Damage to wing.", "NO TEXT")

    # Patch extract_figures to return one record without real pdfplumber
    fig = FigureRecord(doc_id="tsb/a001", page=1, bbox=[0.0, 0.0, 100.0, 100.0],
                       caption="Damage to wing.", ocr_text="NO TEXT")
    with patch.object(run_mod, "extract_figures", return_value=[fig]):
        n = run_mod.extract_figures_for_doc(pdf, out, cap)

    assert n == 1
    dest = out / "en" / "tsb" / "a001_figures.jsonl"
    assert dest.exists()
    lines = [json.loads(l) for l in dest.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["kind"] == "figure"
    assert "Damage to wing." in lines[0]["text"]


def test_extract_figures_for_doc_skips_fresh(tmp_path):
    from ingestion.processing import run as run_mod
    import time

    corpus = tmp_path / "corpus" / "en" / "tsb"
    corpus.mkdir(parents=True)
    pdf = corpus / "a001.pdf"
    pdf.write_bytes(b"fake")

    out = tmp_path / "chunks" / "en" / "tsb"
    out.mkdir(parents=True)
    dest = out / "a001_figures.jsonl"
    dest.write_text('{"kind":"figure","text":"cached"}\n', encoding="utf-8")

    # Make dest newer than src
    import os
    t = os.path.getmtime(str(pdf)) + 10
    os.utime(str(dest), (t, t))

    cap = FakeCaptioner()
    with patch.object(run_mod, "extract_figures", return_value=[]) as mock_extract:
        n = run_mod.extract_figures_for_doc(pdf, tmp_path / "chunks", cap)

    # extract_figures should not be called for a fresh dest
    mock_extract.assert_not_called()
    assert n == 0


def test_extract_figures_for_doc_no_figures(tmp_path):
    from ingestion.processing import run as run_mod

    corpus = tmp_path / "corpus" / "en" / "tsb"
    corpus.mkdir(parents=True)
    pdf = corpus / "a001.pdf"
    pdf.write_bytes(b"fake")

    out = tmp_path / "chunks"
    cap = FakeCaptioner()

    with patch.object(run_mod, "extract_figures", return_value=[]):
        n = run_mod.extract_figures_for_doc(pdf, out, cap)

    assert n == 0
    # No file written when there are no figures
    dest = out / "en" / "tsb" / "a001_figures.jsonl"
    assert not dest.exists()
