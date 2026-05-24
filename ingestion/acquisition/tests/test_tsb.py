"""Offline tests for the TSB scraper."""
from __future__ import annotations

import textwrap

from ingestion.acquisition.tsb import build_pdf_url, extract_report_ids


def test_extract_report_ids_from_href():
    html = textwrap.dedent("""
        <html><body>
          <ul>
            <li><a href="/eng/rapports-reports/aviation/2023/a23h0001/a23h0001.html">Report A23H0001</a></li>
            <li><a href="/eng/rapports-reports/aviation/2022/a22f0102/a22f0102.html">Another</a></li>
            <li><a href="/about">Not a report</a></li>
          </ul>
        </body></html>
    """)
    assert extract_report_ids(html) == ["A22F0102", "A23H0001"]


def test_extract_report_ids_from_text_only():
    html = "<p>Report number A21Q0123 is interesting.</p><a href='/x'>x</a>"
    assert extract_report_ids(html) == ["A21Q0123"]


def test_extract_report_ids_normalizes_case_and_dedupes():
    html = (
        '<a href="/eng/.../a23h0001/a23h0001.html">a23h0001</a>'
        '<a href="/eng/.../A23H0001/A23H0001.html">A23H0001 again</a>'
    )
    assert extract_report_ids(html) == ["A23H0001"]


def test_extract_report_ids_ignores_non_matches():
    html = '<a href="/foo">B23H0001</a><a href="/bar">A23H001</a><a href="/baz">A235H0001</a>'
    assert extract_report_ids(html) == []


def test_build_pdf_url_en():
    assert build_pdf_url("A23H0001", "en") == (
        "https://www.bst-tsb.gc.ca/sites/default/files/"
        "rapports-reports/aviation/a23h0001/eng/a23h0001.pdf"
    )


def test_build_pdf_url_fr():
    assert build_pdf_url("A23H0001", "fr") == (
        "https://www.bst-tsb.gc.ca/sites/default/files/"
        "rapports-reports/aviation/a23h0001/fra/a23h0001.pdf"
    )


def test_build_pdf_url_lowercases_id():
    url = build_pdf_url("A23H0001", "en")
    assert "A23H0001" not in url
    assert "a23h0001.pdf" in url
