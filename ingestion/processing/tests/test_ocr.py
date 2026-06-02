"""Per-language OCR routing — offline, with a stubbed PaddleOCR.

No paddleocr/paddlepaddle import happens: we inject a fake module into
``sys.modules`` so ``_get_ocr`` resolves our recorder instead of the real
(heavy, weight-downloading) PaddleOCR.
"""
from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from ingestion.processing import ocr as ocr_mod
from ingestion.processing.ocr import OCR_RESOLUTION, ocr_page, paddle_lang


# ─── paddle_lang: pure (lang, source) → model-code mapping ────────────────────

@pytest.mark.parametrize("lang,source,expected", [
    ("en", "tsb", "en"),             # PP-OCRv5 English (no more shared "latin")
    ("fr", "tc", "fr"),              # PP-OCRv5 French (diacritics)
    ("zh", "caac", "ch"),            # Simplified
    ("zh", "ttsb", "chinese_cht"),   # Traditional
    ("zh", "whatever", "ch"),        # zh default → Simplified
    ("xx", "tsb", "en"),             # unknown lang → English default
])
def test_paddle_lang_mapping(lang, source, expected):
    assert paddle_lang(lang, source) == expected


# ─── fake PaddleOCR + page, installed per-test ────────────────────────────────

class _FakeOCR:
    """Mimics PaddleOCR 3.x: keyword-only constructor + ``predict()`` returning
    a list of one OCRResult-like dict with parallel rec_texts / rec_polys."""

    instances: list["_FakeOCR"] = []

    def __init__(self, *, lang, use_textline_orientation,
                 use_doc_orientation_classify, use_doc_unwarping, device,
                 text_rec_score_thresh):
        self.lang = lang
        self.use_textline_orientation = use_textline_orientation
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.device = device
        self.text_rec_score_thresh = text_rec_score_thresh
        self.predict_calls: list = []
        _FakeOCR.instances.append(self)

    def predict(self, img):
        # ocr_page passes a BGR ndarray (not a PIL.Image) — assert the
        # conversion happened so we don't regress the "PIL → ndarray" fix.
        import numpy as np
        assert isinstance(img, np.ndarray)
        self.predict_calls.append(img)
        # 3.x OCRResult: parallel lists. Poly in pixel space of the 200-DPI render.
        return [{
            "rec_texts": ["中文测试"],
            "rec_polys": [[[10, 20], [110, 20], [110, 40], [10, 40]]],
            "rec_scores": [0.99],
        }]


class _FakePage:
    def to_image(self, resolution):
        assert resolution == OCR_RESOLUTION
        # Real PIL image so ocr_page's .convert("RGB") + np.asarray work.
        return types.SimpleNamespace(original=Image.new("RGB", (8, 8)))


def _fake_paddle_module(*, cuda: bool):
    """A stand-in ``paddle`` so _ocr_device() doesn't import the real (heavy)
    library in tests — keeps the suite hermetic and the device choice deterministic."""
    mod = types.ModuleType("paddle")
    mod.device = types.SimpleNamespace(
        is_compiled_with_cuda=lambda: cuda,
        cuda=types.SimpleNamespace(device_count=lambda: (1 if cuda else 0)),
    )
    return mod


@pytest.fixture
def fake_paddle(monkeypatch):
    _FakeOCR.instances = []
    ocr_mod._ocr_by_lang = {}  # reset the per-language cache
    fake_mod = types.ModuleType("paddleocr")
    fake_mod.PaddleOCR = _FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_mod)
    # Default: no CUDA → device auto-detects to "cpu" (hermetic, no real paddle).
    monkeypatch.setitem(sys.modules, "paddle", _fake_paddle_module(cuda=False))
    monkeypatch.delenv("PADDLE_OCR_DEVICE", raising=False)
    yield
    ocr_mod._ocr_by_lang = {}


# ─── model construction is routed + cached per code ───────────────────────────

def test_latin_model_built_without_angle_cls(fake_paddle):
    # EN/FR (Latin-script) models don't deskew — only the scanned Chinese ones do.
    m = ocr_mod._get_ocr("en")
    assert m.lang == "en"
    assert m.use_textline_orientation is False


def test_chinese_models_enable_angle_cls(fake_paddle):
    ch = ocr_mod._get_ocr("ch")
    cht = ocr_mod._get_ocr("chinese_cht")
    assert ch.lang == "ch" and ch.use_textline_orientation is True
    assert cht.lang == "chinese_cht" and cht.use_textline_orientation is True


def test_models_are_cached_per_code(fake_paddle):
    a = ocr_mod._get_ocr("ch")
    b = ocr_mod._get_ocr("ch")
    c = ocr_mod._get_ocr("chinese_cht")
    assert a is b              # same code reuses the instance
    assert c is not a          # different code builds a distinct model
    assert len(_FakeOCR.instances) == 2


# ─── ocr_page: coords in PDF points, parses 3.x predict() output ──────────────

def test_ocr_page_converts_pixels_to_points(fake_paddle):
    page = _FakePage()
    extract = ocr_page(page, page_no=7, ocr_lang="ch")

    assert extract.page == 7
    assert extract.text == "中文测试"
    # One Char PER GLYPH (4), so the chunker can align chunk text to bboxes.
    assert len(extract.chars) == 4
    # 200-DPI pixels → PDF points: factor 72/200 = 0.36
    f = 72.0 / OCR_RESOLUTION
    first, last = extract.chars[0], extract.chars[-1]
    # Line x-extent [10,110]px subdivided across 4 glyphs: first starts at 10px,
    # last ends at 110px; all glyphs share the line's y-extent.
    assert first.x0 == pytest.approx(10 * f)
    assert last.x1 == pytest.approx(110 * f)
    assert first.x1 == pytest.approx((10 + (110 - 10) / 4) * f)
    for ch in extract.chars:
        assert ch.top == pytest.approx(20 * f)
        assert ch.bottom == pytest.approx(40 * f)
    # predict() was called once with the BGR ndarray.
    assert len(_FakeOCR.instances[0].predict_calls) == 1


def test_ocr_page_device_from_env(fake_paddle, monkeypatch):
    monkeypatch.setenv("PADDLE_OCR_DEVICE", "gpu")  # explicit override wins
    ocr_page(_FakePage(), page_no=1, ocr_lang="en")
    assert _FakeOCR.instances[0].device == "gpu"
    assert _FakeOCR.instances[0].use_textline_orientation is False


def test_device_falls_back_to_cpu_without_gpu(fake_paddle):
    # Fixture stubs paddle with no CUDA → auto-detect must pick CPU (the fallback).
    ocr_page(_FakePage(), page_no=1, ocr_lang="ch")
    assert _FakeOCR.instances[0].device == "cpu"


def test_device_prefers_gpu_when_available(fake_paddle, monkeypatch):
    # Stub paddle reporting a usable CUDA GPU → auto-detect must prefer GPU.
    monkeypatch.setitem(sys.modules, "paddle", _fake_paddle_module(cuda=True))
    ocr_page(_FakePage(), page_no=1, ocr_lang="ch")
    assert _FakeOCR.instances[0].device == "gpu"
