from unittest.mock import MagicMock

from ingestion.processing.pdf import Char, extract_page


def _mock_page(text: str, chars: list[dict], images: list = None):
    p = MagicMock()
    p.extract_text.return_value = text
    p.chars = chars
    p.images = images or []
    return p


def test_extract_page_text_and_chars():
    chars = [
        {"text": "H", "x0": 10, "x1": 18, "top": 50, "bottom": 62, "size": 12.0},
        {"text": "i", "x0": 18, "x1": 22, "top": 50, "bottom": 62, "size": 12.0},
    ]
    p = _mock_page("Hi", chars)
    extract = extract_page(p, 3)
    assert extract.page == 3
    assert extract.text == "Hi"
    assert extract.chars == [
        Char("H", 10.0, 18.0, 50.0, 62.0, 12.0, 3),
        Char("i", 18.0, 22.0, 50.0, 62.0, 12.0, 3),
    ]
    assert extract.image_only is False


def test_extract_page_flags_image_only_when_no_text_but_images_present():
    p = _mock_page("", chars=[], images=[{"x0": 0}])
    extract = extract_page(p, 1)
    assert extract.image_only is True
    assert extract.text == ""


def test_extract_page_not_image_only_when_text_present_even_with_images():
    p = _mock_page("hello", chars=[], images=[{"x0": 0}])
    extract = extract_page(p, 1)
    assert extract.image_only is False


def test_extract_page_not_image_only_when_no_text_and_no_images():
    p = _mock_page("", chars=[], images=[])
    extract = extract_page(p, 1)
    assert extract.image_only is False


def test_coerce_char_defaults_missing_fields():
    # pdfplumber chars usually have all fields; we still cope if some are absent.
    chars = [{"text": "x"}]
    p = _mock_page("x", chars)
    extract = extract_page(p, 1)
    assert extract.chars[0].text == "x"
    assert extract.chars[0].size == 0.0
