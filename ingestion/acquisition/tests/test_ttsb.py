"""Offline tests for the TTSB (Taiwan) Traditional-Chinese report scraper."""
from __future__ import annotations

import textwrap

from ingestion.acquisition.ttsb import extract_pdf_urls, filename_for, media_id

_BASE = "https://www.ttsb.gov.tw/1133/1154/1155/1159/Lpsimplelist"


def test_extract_pdf_urls_finds_media_links():
    html = textwrap.dedent("""
        <html><body>
          <a href="/1243/16869/44270/post">detail page</a>
          <a href="/media/9234/安捷b-86002調查報告.pdf">報告</a>
          <a href="/media/8925/jj2258調查報告.pdf">報告</a>
        </body></html>
    """)
    assert extract_pdf_urls(html, _BASE) == [
        "https://www.ttsb.gov.tw/media/8925/jj2258調查報告.pdf",
        "https://www.ttsb.gov.tw/media/9234/安捷b-86002調查報告.pdf",
    ]


def test_extract_pdf_urls_ignores_detail_pages_and_non_media_pdfs():
    html = (
        '<a href="/1243/16869/44270/post">detail</a>'        # not a PDF
        '<a href="/sites/forms/application.pdf">a form</a>'  # PDF but not /media/{id}/
        '<a href="https://example.com/media/1/x.pdf">off-host</a>'
    )
    assert extract_pdf_urls(html, _BASE) == []


def test_extract_pdf_urls_resolves_percent_encoded_relative_links():
    # Real hrefs often percent-encode the Chinese basename.
    html = '<a href="/media/9234/%E5%AE%89%E6%8D%B7.pdf">x</a>'
    assert extract_pdf_urls(html, _BASE) == [
        "https://www.ttsb.gov.tw/media/9234/%E5%AE%89%E6%8D%B7.pdf"
    ]


def test_extract_pdf_urls_deduplicates():
    html = (
        '<a href="/media/9234/a.pdf">1</a>'
        '<a href="https://www.ttsb.gov.tw/media/9234/a.pdf">2</a>'
    )
    assert extract_pdf_urls(html, _BASE) == [
        "https://www.ttsb.gov.tw/media/9234/a.pdf"
    ]


def test_media_id_extracts_numeric_key():
    assert media_id("https://www.ttsb.gov.tw/media/9234/a.pdf") == "9234"
    assert media_id("https://www.ttsb.gov.tw/no-media/a.pdf") is None


def test_filename_for_prefixes_media_id():
    assert filename_for("https://www.ttsb.gov.tw/media/9234/ci611_general.pdf") == (
        "9234_ci611_general.pdf"
    )


def test_filename_for_disambiguates_repeated_basenames():
    # Two reports both named 00_general.pdf — the media id keeps them distinct.
    a = filename_for("https://www.ttsb.gov.tw/media/3059/00_general.pdf")
    b = filename_for("https://www.ttsb.gov.tw/media/4257/00_general.pdf")
    assert a == "3059_00_general.pdf"
    assert b == "4257_00_general.pdf"
    assert a != b


def test_filename_for_decodes_percent_encoded_chinese():
    # %E5%AE%89%E6%8D%B7 == 安捷
    assert filename_for("https://www.ttsb.gov.tw/media/9234/%E5%AE%89%E6%8D%B7.pdf") == (
        "9234_安捷.pdf"
    )
