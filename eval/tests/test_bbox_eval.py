"""Tests for eval.bbox_eval — all mocked, no PDF files or PaddleOCR needed."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.bbox_eval import (
    BBoxResult,
    _bbox_area,
    _bbox_to_pixels,
    _text_similarity,
    eval_chunk,
    load_sample_chunks,
    run_eval,
)


# ─── pure helpers ─────────────────────────────────────────────────────────────

def test_bbox_area_normal():
    assert _bbox_area([10.0, 20.0, 110.0, 70.0]) == pytest.approx(5000.0)


def test_bbox_area_degenerate():
    assert _bbox_area([0.0, 0.0, 0.0, 0.0]) == 0.0
    assert _bbox_area([50.0, 50.0, 30.0, 40.0]) == 0.0  # inverted coords → 0


def test_bbox_to_pixels_at_72dpi():
    # At 72 DPI the scale is 1 — PDF pts map 1:1 to pixels.
    assert _bbox_to_pixels([10.0, 20.0, 110.0, 70.0], dpi=72) == (10, 20, 110, 70)


def test_bbox_to_pixels_at_144dpi():
    # At 144 DPI scale = 2.
    assert _bbox_to_pixels([10.0, 20.0, 110.0, 70.0], dpi=144) == (20, 40, 220, 140)


def test_bbox_to_pixels_clamped_to_zero():
    # Negative coords should clamp to 0.
    assert _bbox_to_pixels([-5.0, -10.0, 50.0, 30.0], dpi=72) == (0, 0, 50, 30)


def test_text_similarity_identical():
    assert _text_similarity("hello world", "hello world") == pytest.approx(1.0)


def test_text_similarity_empty():
    assert _text_similarity("", "") == pytest.approx(1.0)
    assert _text_similarity("abc", "") == pytest.approx(0.0)


def test_text_similarity_partial():
    s = _text_similarity("fuel exhaustion", "fuel exhaus")
    assert 0.8 < s < 1.0


def test_text_similarity_case_insensitive():
    assert _text_similarity("Fuel Exhaustion", "fuel exhaustion") == pytest.approx(1.0)


# ─── eval_chunk ───────────────────────────────────────────────────────────────

SAMPLE_CHUNK = {
    "doc_id": "tsb/a13q0098",
    "lang": "en",
    "page": 3,
    "bbox": [86.4, 38.4, 542.2, 755.9],
    "text": "Forced landing following fuel exhaustion Aviation Flycie Inc.",
}


def _make_fake_pil(w=800, h=1100):
    from PIL import Image
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def test_eval_chunk_pdf_not_found(tmp_path):
    """eval_chunk reports error when PDF doesn't exist."""
    result = eval_chunk(SAMPLE_CHUNK, corpus_root=tmp_path)
    assert result.error is not None
    assert "not found" in result.error.lower()
    assert result.similarity == 0.0


def test_eval_chunk_render_error(tmp_path):
    """eval_chunk captures render_crop exceptions gracefully."""
    # Create a dummy PDF directory so _pdf_path finds something.
    pdf_dir = tmp_path / "en" / "tsb"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "a13q0098.pdf").write_bytes(b"%PDF-1.4 dummy")

    with patch("eval.bbox_eval.render_crop", side_effect=ValueError("bad bbox")):
        result = eval_chunk(SAMPLE_CHUNK, corpus_root=tmp_path)
    assert result.error is not None
    assert "render_crop failed" in result.error


def test_eval_chunk_ocr_error(tmp_path):
    """eval_chunk captures OCR exceptions gracefully."""
    pdf_dir = tmp_path / "en" / "tsb"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "a13q0098.pdf").write_bytes(b"%PDF-1.4 dummy")

    fake_crop = _make_fake_pil()
    with patch("eval.bbox_eval.render_crop", return_value=fake_crop):
        with patch("eval.bbox_eval._ocr_image", side_effect=RuntimeError("paddleocr missing")):
            result = eval_chunk(SAMPLE_CHUNK, corpus_root=tmp_path)
    assert result.error is not None
    assert "OCR failed" in result.error


def test_eval_chunk_hit(tmp_path):
    """eval_chunk records a hit when OCR closely matches stored text."""
    pdf_dir = tmp_path / "en" / "tsb"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "a13q0098.pdf").write_bytes(b"%PDF-1.4 dummy")

    fake_crop = _make_fake_pil()
    # OCR returns something close to the chunk text → similarity above threshold.
    fake_ocr = "Forced landing following fuel exhaustion"
    with patch("eval.bbox_eval.render_crop", return_value=fake_crop):
        with patch("eval.bbox_eval._ocr_image", return_value=fake_ocr):
            result = eval_chunk(SAMPLE_CHUNK, corpus_root=tmp_path)
    assert result.error is None
    assert result.hit is True
    assert result.similarity >= 0.40


def test_eval_chunk_miss(tmp_path):
    """eval_chunk records a miss when OCR returns garbage."""
    pdf_dir = tmp_path / "en" / "tsb"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "a13q0098.pdf").write_bytes(b"%PDF-1.4 dummy")

    fake_crop = _make_fake_pil()
    with patch("eval.bbox_eval.render_crop", return_value=fake_crop):
        with patch("eval.bbox_eval._ocr_image", return_value="xxxxxxxxxxx"):
            result = eval_chunk(SAMPLE_CHUNK, corpus_root=tmp_path)
    assert result.error is None
    assert result.hit is False


def test_eval_chunk_saves_crop(tmp_path):
    """eval_chunk saves crop PNG when save_crops is given."""
    pdf_dir = tmp_path / "en" / "tsb"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "a13q0098.pdf").write_bytes(b"%PDF-1.4 dummy")

    crops_dir = tmp_path / "crops"
    fake_crop = _make_fake_pil()
    with patch("eval.bbox_eval.render_crop", return_value=fake_crop):
        with patch("eval.bbox_eval._ocr_image", return_value="fuel"):
            eval_chunk(SAMPLE_CHUNK, corpus_root=tmp_path, save_crops=crops_dir)

    saved = list(crops_dir.glob("*.png"))
    assert len(saved) == 1
    assert "a13q0098" in saved[0].name


# ─── load_sample_chunks ───────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _good_chunk(**overrides) -> dict:
    base = {
        "doc_id": "tsb/a00a0051",
        "lang": "en",
        "page": 2,
        "bbox": [10.0, 20.0, 400.0, 200.0],
        "chunk_hash": "abc123",
        "text": "A" * 80,
    }
    base.update(overrides)
    return base


def test_load_sample_chunks_basic(tmp_path):
    chunks_root = tmp_path / "chunks"
    jl = chunks_root / "en" / "tsb" / "a00a0051.jsonl"
    _write_jsonl(jl, [_good_chunk() for _ in range(10)])

    result = load_sample_chunks(chunks_root, n=5, source=None, seed=0)
    assert len(result) == 5


def test_load_sample_chunks_filters_degenerate(tmp_path):
    chunks_root = tmp_path / "chunks"
    jl = chunks_root / "en" / "tsb" / "a00a0051.jsonl"
    bad_chunks = [
        _good_chunk(bbox=[0.0, 0.0, 0.0, 0.0]),      # zero area
        _good_chunk(bbox=[10.0, 10.0, 15.0, 15.0]),   # area=25 < MIN_BBOX_AREA_PTS(100)
        _good_chunk(text="short"),                      # text too short
        _good_chunk(page=0),                            # invalid page
    ]
    good_chunks = [_good_chunk(doc_id="tsb/aXXX") for _ in range(3)]
    _write_jsonl(jl, bad_chunks + good_chunks)

    result = load_sample_chunks(chunks_root, n=10, source=None, seed=0)
    assert all(r["doc_id"] == "tsb/aXXX" for r in result)
    assert len(result) == 3


def test_load_sample_chunks_source_filter(tmp_path):
    chunks_root = tmp_path / "chunks"
    _write_jsonl(chunks_root / "en" / "tsb" / "tsb_doc.jsonl", [_good_chunk(doc_id="tsb/doc")])
    _write_jsonl(chunks_root / "en" / "tc"  / "tc_doc.jsonl",  [_good_chunk(doc_id="tc/doc")])

    result = load_sample_chunks(chunks_root, n=10, source="tsb", seed=0)
    assert all(r["doc_id"].startswith("tsb/") for r in result)


# ─── run_eval (integration with mocks) ───────────────────────────────────────

def test_run_eval_produces_report(tmp_path):
    chunks_root = tmp_path / "chunks"
    corpus_root = tmp_path / "corpus"
    # Write a chunk file.
    jl = chunks_root / "en" / "tsb" / "a00a0051.jsonl"
    _write_jsonl(jl, [_good_chunk() for _ in range(5)])

    def _fake_eval_chunk(chunk, corpus_root, **kw):
        return BBoxResult(
            doc_id=chunk["doc_id"], lang=chunk["lang"], page=chunk["page"],
            bbox=chunk["bbox"],
            chunk_text_preview=chunk["text"][:80],
            ocr_text_preview="matched text",
            similarity=0.75,
            hit=True,
        )

    with patch("eval.bbox_eval.eval_chunk", side_effect=_fake_eval_chunk):
        report = run_eval(chunks_root, corpus_root, n=3, source=None, seed=0)

    assert report.n_sampled == 3
    assert report.n_ok == 3
    assert report.n_hit == 3
    assert 0.0 < report.mean_similarity <= 1.0
