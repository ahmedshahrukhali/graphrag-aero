"""Offline tests for the TC AC scraper."""
from __future__ import annotations

import textwrap

from ingestion.acquisition.tc import (
    extract_ac_detail_urls,
    extract_pdf_urls,
    filename_for,
)

_BASE = "https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars"
_BASE_FR = "https://tc.canada.ca/fr/aviation/centre-reference/circulaires-information"


def test_extract_pdf_urls_resolves_relative_links():
    html = textwrap.dedent("""
        <html><body>
          <ul>
            <li><a href="https://tc.canada.ca/sites/default/files/2023-06/AC_700-046_issue_03.pdf">AC 700-046</a></li>
            <li><a href="/sites/default/files/2024-01/AC_300-001.pdf">AC 300-001</a></li>
            <li><a href="/news">Not a PDF</a></li>
          </ul>
        </body></html>
    """)
    assert extract_pdf_urls(html, _BASE) == [
        "https://tc.canada.ca/sites/default/files/2023-06/AC_700-046_issue_03.pdf",
        "https://tc.canada.ca/sites/default/files/2024-01/AC_300-001.pdf",
    ]


def test_extract_pdf_urls_ignores_non_tc_hosts():
    html = '<a href="https://example.com/something.pdf">x</a>'
    assert extract_pdf_urls(html, _BASE) == []


def test_extract_pdf_urls_handles_case_insensitive_extension():
    html = '<a href="https://tc.canada.ca/sites/default/files/x.PDF">x</a>'
    assert extract_pdf_urls(html, _BASE) == [
        "https://tc.canada.ca/sites/default/files/x.PDF"
    ]


def test_extract_pdf_urls_deduplicates():
    html = (
        '<a href="https://tc.canada.ca/sites/default/files/a.pdf">1</a>'
        '<a href="https://tc.canada.ca/sites/default/files/a.pdf">2</a>'
    )
    assert extract_pdf_urls(html, _BASE) == [
        "https://tc.canada.ca/sites/default/files/a.pdf"
    ]


def test_extract_pdf_urls_accepts_subdomains_of_tc():
    html = '<a href="https://www.tc.canada.ca/sites/default/files/a.pdf">x</a>'
    assert extract_pdf_urls(html, _BASE) == [
        "https://www.tc.canada.ca/sites/default/files/a.pdf"
    ]


def test_extract_ac_detail_urls_finds_children_of_index():
    html = textwrap.dedent("""
        <html><body>
          <a href="/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-100-001">100-001</a>
          <a href="/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-dan-001">DAN-001</a>
          <a href="/en/aviation/reference-centre/advisory-circulars#100-series">anchor</a>
          <a href="/en/aviation/reference-centre/advisory-circulars">self</a>
          <a href="/en/aviation/other-page">sibling, not child</a>
          <a href="https://tc.canada.ca/sites/default/files/AC.pdf">a PDF on the index, ignore</a>
          <a href="https://example.com/en/aviation/reference-centre/advisory-circulars/x">off-host</a>
        </body></html>
    """)
    assert extract_ac_detail_urls(html, _BASE) == [
        "https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-dan-001",
        "https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-100-001",
    ]


def test_extract_ac_detail_urls_works_for_fr_index():
    html = (
        '<a href="/fr/aviation/centre-reference/circulaires-information/circulaire-information-ci-ndeg-100-001">x</a>'
        '<a href="/fr/aviation/centre-reference/circulaires-information#serie-100">anchor</a>'
    )
    assert extract_ac_detail_urls(html, _BASE_FR) == [
        "https://tc.canada.ca/fr/aviation/centre-reference/circulaires-information/circulaire-information-ci-ndeg-100-001",
    ]


def test_extract_ac_detail_urls_strips_query_and_fragment():
    html = (
        '<a href="/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-100-001?x=1">a</a>'
        '<a href="/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-100-001#top">b</a>'
    )
    # Both collapse to the canonical detail URL.
    assert extract_ac_detail_urls(html, _BASE) == [
        "https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-100-001",
    ]


def test_filename_for_takes_last_segment():
    assert filename_for(
        "https://tc.canada.ca/sites/default/files/2024-01/AC_300-001.pdf"
    ) == "AC_300-001.pdf"


def test_filename_for_ignores_query():
    assert filename_for(
        "https://tc.canada.ca/sites/default/files/foo.pdf?x=1"
    ) == "foo.pdf"


def test_filename_for_decodes_percent_encoded_utf8():
    # FR ACs sometimes have percent-encoded titles. We want the readable name.
    assert filename_for(
        "https://tc.canada.ca/sites/default/files/2022-09/"
        "CI_102-001_-_EXPLOITATION_D%E2%80%99A%C3%89RONEFS.pdf"
    ) == "CI_102-001_-_EXPLOITATION_D’AÉRONEFS.pdf"
