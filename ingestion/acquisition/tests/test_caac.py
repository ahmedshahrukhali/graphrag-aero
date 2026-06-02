"""Offline tests for the CAAC (China) seed-manifest scraper."""
from __future__ import annotations

from ingestion.acquisition.caac import (
    DEFAULT_SEED,
    filename_for,
    load_seed,
    load_seed_file,
)

_PDF = "http://www.caac.gov.cn/XXGK/XXGK/GFXWJ/201511/P020151103346484825446.pdf"


def test_filename_for_takes_basename():
    assert filename_for(_PDF) == "P020151103346484825446.pdf"


def test_load_seed_skips_comments_and_blanks():
    text = "\n".join([
        "# a comment",
        "",
        _PDF,
        "   # indented comment",
        "https://www.caac.gov.cn/HDJL/YJZJ/202512/P020251226558514350774.pdf",
    ])
    assert load_seed(text) == [
        _PDF,
        "https://www.caac.gov.cn/HDJL/YJZJ/202512/P020251226558514350774.pdf",
    ]


def test_load_seed_dedupes_by_basename_across_mirrors():
    # Same P-number behind /PHONE/ and /big5/ mirrors → one entry (first wins).
    canonical = "https://www.caac.gov.cn/HDJL/YJZJ/202512/P020251226558514350774.pdf"
    phone = "https://www.caac.gov.cn/PHONE/HDJL/YJZJ/202512/P020251226558514350774.pdf"
    big5 = "https://www.caac.gov.cn/big5/www.caac.gov.cn/PHONE/HDJL/YJZJ/202512/P020251226558514350774.pdf"
    assert load_seed("\n".join([canonical, phone, big5])) == [canonical]


def test_load_seed_rejects_non_caac_and_non_pdf():
    text = "\n".join([
        "https://example.com/x.pdf",                 # off-host
        "https://www.caac.gov.cn/XXGK/page.html",     # not a PDF
        _PDF,                                         # keeper
    ])
    assert load_seed(text) == [_PDF]


def test_load_seed_file_missing_returns_empty(tmp_path):
    assert load_seed_file(tmp_path / "nope.txt") == []


def test_committed_default_seed_parses_nonempty():
    """The committed manifest must stay valid: parses to a deduped CAAC PDF list."""
    urls = load_seed_file(DEFAULT_SEED)
    assert len(urls) >= 15
    assert all(".pdf" in u.lower() and "caac.gov.cn" in u for u in urls)
    # No duplicate basenames survive.
    names = [filename_for(u) for u in urls]
    assert len(names) == len(set(names))
