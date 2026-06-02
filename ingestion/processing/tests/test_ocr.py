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
    ("en", "tsb", "latin"),
    ("fr", "tc", "latin"),
    ("zh", "caac", "ch"),            # Simplified
    ("zh", "ttsb", "chinese_cht"),   # Traditional
    ("zh", "whatever", "ch"),        # zh default → Simplified
])
def test_paddle_lang_mapping(lang, source, expected):
    assert paddle_lang(lang, source) == expected


# ─── fake PaddleOCR + page, installed per-test ────────────────────────────────

class _FakeOCR:
    instances: list["_FakeOCR"] = []

    def __init__(self, *, use_angle_cls, lang, show_log):
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.show_log = show_log
        self.cls_calls: list[bool] = []
        _FakeOCR.instances.append(self)

    def ocr(self, img, cls):
        self.cls_calls.append(cls)
        # ocr_page now passes a BGR ndarray (not a PIL.Image) — assert the
        # conversion happened so we don't regress the "PIL → ndarray" fix.
        import numpy as np
        assert isinstance(img, np.ndarray)
        # PaddleOCR returns a list of pages; we passed one page → result[0] is
        # the list of line entries. Polygon is in pixel space of the 200-DPI render.
        return [[
            [[[10, 20], [110, 20], [110, 40], [10, 40]], ("中文测试", 0.99)],
        ]]


class _FakePage:
    def to_image(self, resolution):
        assert resolution == OCR_RESOLUTION
        # Real PIL image so ocr_page's .convert("RGB") + np.asarray work.
        return types.SimpleNamespace(original=Image.new("RGB", (8, 8)))


@pytest.fixture
def fake_paddle(monkeypatch):
    _FakeOCR.instances = []
    ocr_mod._ocr_by_lang = {}  # reset the per-language cache
    fake_mod = types.ModuleType("paddleocr")
    fake_mod.PaddleOCR = _FakeOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_mod)
    yield
    ocr_mod._ocr_by_lang = {}


# ─── model construction is routed + cached per code ───────────────────────────

def test_latin_model_built_without_angle_cls(fake_paddle):
    m = ocr_mod._get_ocr("latin")
    assert m.lang == "latin"
    assert m.use_angle_cls is False


def test_chinese_models_enable_angle_cls(fake_paddle):
    ch = ocr_mod._get_ocr("ch")
    cht = ocr_mod._get_ocr("chinese_cht")
    assert ch.lang == "ch" and ch.use_angle_cls is True
    assert cht.lang == "chinese_cht" and cht.use_angle_cls is True


def test_models_are_cached_per_code(fake_paddle):
    a = ocr_mod._get_ocr("ch")
    b = ocr_mod._get_ocr("ch")
    c = ocr_mod._get_ocr("chinese_cht")
    assert a is b              # same code reuses the instance
    assert c is not a          # different code builds a distinct model
    assert len(_FakeOCR.instances) == 2


# ─── ocr_page: coords in PDF points, cls flag matches the model ───────────────

def test_ocr_page_converts_pixels_to_points_and_passes_cls(fake_paddle):
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
    # the Chinese model deskews → cls=True was passed to .ocr()
    assert _FakeOCR.instances[0].cls_calls == [True]


def test_ocr_page_latin_passes_cls_false(fake_paddle):
    ocr_page(_FakePage(), page_no=1, ocr_lang="latin")
    assert _FakeOCR.instances[0].cls_calls == [False]
