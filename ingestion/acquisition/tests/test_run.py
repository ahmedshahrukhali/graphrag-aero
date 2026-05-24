"""End-to-end test of the run orchestrator with mocked HTTP."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.acquisition import run


class _StreamingResp:
    def __init__(self, chunks, status_error=None):
        self._chunks = chunks
        self._status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error

    def iter_content(self, chunk_size=1024):
        for c in self._chunks:
            yield c


def test_run_tsb_downloads_en_and_fr(tmp_path: Path):
    index_html = (
        '<a href="/eng/rapports-reports/aviation/2023/a23h0001/a23h0001.html">A23H0001</a>'
    )

    text_resp = MagicMock()
    text_resp.text = index_html
    text_resp.raise_for_status = lambda: None

    session = MagicMock()
    session.get.side_effect = [
        text_resp,                              # index fetch
        _StreamingResp([b"EN-pdf-bytes"]),      # EN download
        _StreamingResp([b"FR-pdf-bytes"]),      # FR download
    ]

    with patch("ingestion.acquisition.run.make_session", return_value=session):
        results = run.run_tsb(tmp_path, session, limit=None)

    en = tmp_path / "en" / "tsb" / "a23h0001.pdf"
    fr = tmp_path / "fr" / "tsb" / "a23h0001.pdf"
    assert en.read_bytes() == b"EN-pdf-bytes"
    assert fr.read_bytes() == b"FR-pdf-bytes"
    assert [r.downloaded for r in results] == [True, True]


def test_run_tsb_continues_past_404(tmp_path: Path):
    import requests

    index_html = (
        '<a href="/eng/a23h0001.html">A23H0001</a>'
        '<a href="/eng/a22f0001.html">A22F0001</a>'
    )

    text_resp = MagicMock()
    text_resp.text = index_html
    text_resp.raise_for_status = lambda: None

    # IDs come out sorted: [A22F0001, A23H0001]; each tried EN then FR.
    a22_en = _StreamingResp([b"a22-en"])
    a22_fr = _StreamingResp([], status_error=requests.HTTPError("404 fr missing"))
    a23_en = _StreamingResp([], status_error=requests.HTTPError("404 en missing"))
    a23_fr = _StreamingResp([b"a23-fr"])

    session = MagicMock()
    session.get.side_effect = [text_resp, a22_en, a22_fr, a23_en, a23_fr]

    results = run.run_tsb(tmp_path, session, limit=None)

    downloaded = [r for r in results if r.downloaded]
    assert len(downloaded) == 2
    assert (tmp_path / "en" / "tsb" / "a22f0001.pdf").exists()
    assert (tmp_path / "fr" / "tsb" / "a23h0001.pdf").exists()
    # The two failures should have left no leftover files.
    assert not (tmp_path / "fr" / "tsb" / "a22f0001.pdf").exists()
    assert not (tmp_path / "en" / "tsb" / "a23h0001.pdf").exists()


def test_run_tsb_respects_limit(tmp_path: Path):
    index_html = (
        '<a href="/eng/a23h0001.html">A23H0001</a>'
        '<a href="/eng/a22f0001.html">A22F0001</a>'
        '<a href="/eng/a21q0001.html">A21Q0001</a>'
    )
    text_resp = MagicMock()
    text_resp.text = index_html
    text_resp.raise_for_status = lambda: None

    session = MagicMock()
    session.get.side_effect = [
        text_resp,
        _StreamingResp([b"a"]),  # first id, EN
        _StreamingResp([b"b"]),  # first id, FR
    ]

    results = run.run_tsb(tmp_path, session, limit=1)

    assert len(results) == 2  # 1 id × 2 langs
    # index call + 2 downloads
    assert session.get.call_count == 3


def _text_resp(body: str):
    r = MagicMock()
    r.text = body
    r.raise_for_status = lambda: None
    return r


def test_run_tc_two_step_en_and_fr(tmp_path: Path):
    en_index = (
        '<a href="/en/aviation/reference-centre/advisory-circulars/'
        'advisory-circular-ac-no-100-001">AC100-001</a>'
    )
    fr_index = (
        '<a href="/fr/aviation/centre-reference/circulaires-information/'
        'circulaire-information-ci-ndeg-100-001">CI100-001</a>'
    )
    en_detail = (
        '<a href="https://tc.canada.ca/sites/default/files/2021-06/'
        'AC_100-001_e08_20210622.pdf">PDF</a>'
    )
    fr_detail = (
        '<a href="https://tc.canada.ca/sites/default/files/2021-06/'
        'AC_100-001_f08_20210622.pdf">PDF</a>'
    )

    session = MagicMock()
    session.get.side_effect = [
        _text_resp(en_index),                  # EN index
        _text_resp(en_detail),                 # EN detail page
        _StreamingResp([b"en-pdf"]),           # EN PDF download
        _text_resp(fr_index),                  # FR index
        _text_resp(fr_detail),                 # FR detail page
        _StreamingResp([b"fr-pdf"]),           # FR PDF download
    ]

    results = run.run_tc(tmp_path, session, limit=None)

    assert (tmp_path / "en" / "tc" / "AC_100-001_e08_20210622.pdf").read_bytes() == b"en-pdf"
    assert (tmp_path / "fr" / "tc" / "AC_100-001_f08_20210622.pdf").read_bytes() == b"fr-pdf"
    assert [r.downloaded for r in results] == [True, True]


def test_run_tc_skips_detail_page_with_no_pdf(tmp_path: Path):
    # Two detail pages; the first has no PDF, the second has one. Crawl must continue.
    en_index = (
        '<a href="/en/aviation/reference-centre/advisory-circulars/'
        'advisory-circular-ac-no-aaa">A</a>'
        '<a href="/en/aviation/reference-centre/advisory-circulars/'
        'advisory-circular-ac-no-bbb">B</a>'
    )
    fr_index = ""  # empty FR index — nothing to do for FR
    detail_no_pdf = "<p>page without any PDF link</p>"
    detail_with_pdf = (
        '<a href="https://tc.canada.ca/sites/default/files/AC_bbb.pdf">PDF</a>'
    )

    session = MagicMock()
    session.get.side_effect = [
        _text_resp(en_index),
        _text_resp(detail_no_pdf),
        _text_resp(detail_with_pdf),
        _StreamingResp([b"bbb-bytes"]),
        _text_resp(fr_index),
    ]

    results = run.run_tc(tmp_path, session, limit=None)

    assert (tmp_path / "en" / "tc" / "AC_bbb.pdf").read_bytes() == b"bbb-bytes"
    assert [r.downloaded for r in results] == [True]


def test_run_tc_continues_past_detail_404(tmp_path: Path):
    import requests as _requests

    en_index = (
        '<a href="/en/aviation/reference-centre/advisory-circulars/'
        'advisory-circular-ac-no-aaa">A</a>'
        '<a href="/en/aviation/reference-centre/advisory-circulars/'
        'advisory-circular-ac-no-bbb">B</a>'
    )
    fr_index = ""
    err_resp = MagicMock()
    err_resp.raise_for_status = MagicMock(
        side_effect=_requests.HTTPError("404 detail missing")
    )
    detail_with_pdf = (
        '<a href="https://tc.canada.ca/sites/default/files/AC_bbb.pdf">PDF</a>'
    )

    session = MagicMock()
    session.get.side_effect = [
        _text_resp(en_index),
        err_resp,                                # first detail page errors
        _text_resp(detail_with_pdf),             # second detail page ok
        _StreamingResp([b"bbb-bytes"]),
        _text_resp(fr_index),
    ]

    results = run.run_tc(tmp_path, session, limit=None)
    assert [r.downloaded for r in results] == [True]
    assert (tmp_path / "en" / "tc" / "AC_bbb.pdf").read_bytes() == b"bbb-bytes"


def test_run_tc_respects_limit(tmp_path: Path):
    # 3 EN detail pages, limit=1 -> only the first is fetched.
    en_index = (
        '<a href="/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-aaa">A</a>'
        '<a href="/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-bbb">B</a>'
        '<a href="/en/aviation/reference-centre/advisory-circulars/advisory-circular-ac-no-ccc">C</a>'
    )
    fr_index = ""
    detail = '<a href="https://tc.canada.ca/sites/default/files/AC_aaa.pdf">PDF</a>'

    session = MagicMock()
    session.get.side_effect = [
        _text_resp(en_index),
        _text_resp(detail),
        _StreamingResp([b"aaa"]),
        _text_resp(fr_index),
    ]

    results = run.run_tc(tmp_path, session, limit=1)
    assert [r.downloaded for r in results] == [True]
    # EN: index + 1 detail + 1 download = 3 calls. FR: index = 1 call. Total 4.
    assert session.get.call_count == 4
