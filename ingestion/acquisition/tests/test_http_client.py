"""Offline tests for the polite HTTP client."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.acquisition.http_client import (
    USER_AGENT,
    download,
    fetch_text,
    make_session,
    sha256_of,
)


class _StreamingResp:
    """Stand-in for ``requests.Response`` returned by ``session.get(stream=True)``."""

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


def test_make_session_sets_user_agent():
    s = make_session()
    assert "graphrag-aero" in s.headers["User-Agent"]
    assert s.headers["User-Agent"] == USER_AGENT


def test_download_writes_file_and_returns_sha(tmp_path: Path):
    payload = b"hello pdf body"
    session = MagicMock()
    session.get.return_value = _StreamingResp([payload])
    sleeps: list[float] = []
    dest = tmp_path / "a" / "b.pdf"

    result = download(session, "http://x/test.pdf", dest, sleep=sleeps.append)

    assert dest.read_bytes() == payload
    assert result.downloaded is True
    assert result.size_bytes == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert sleeps == [1.0]   # one polite delay before the request


def test_download_streams_multiple_chunks(tmp_path: Path):
    chunks = [b"chunk-one-", b"chunk-two-", b"chunk-three"]
    session = MagicMock()
    session.get.return_value = _StreamingResp(chunks)
    dest = tmp_path / "z.pdf"

    result = download(session, "http://x/z.pdf", dest, sleep=lambda s: None)

    assert dest.read_bytes() == b"".join(chunks)
    assert result.size_bytes == sum(len(c) for c in chunks)
    assert result.sha256 == hashlib.sha256(b"".join(chunks)).hexdigest()


def test_download_skips_existing(tmp_path: Path):
    dest = tmp_path / "x.pdf"
    dest.write_bytes(b"already here")
    session = MagicMock()

    result = download(session, "http://x/x.pdf", dest, sleep=lambda s: None)

    assert result.downloaded is False
    assert result.size_bytes == len(b"already here")
    assert result.sha256 == sha256_of(dest)
    session.get.assert_not_called()


def test_download_redownloads_zero_byte_files(tmp_path: Path):
    dest = tmp_path / "empty.pdf"
    dest.touch()  # zero-byte stub
    session = MagicMock()
    session.get.return_value = _StreamingResp([b"real content"])

    result = download(session, "http://x/empty.pdf", dest, sleep=lambda s: None)

    assert result.downloaded is True
    assert dest.read_bytes() == b"real content"


def test_download_cleans_tmp_on_error(tmp_path: Path):
    session = MagicMock()
    session.get.return_value = _StreamingResp([], status_error=RuntimeError("boom"))
    dest = tmp_path / "y.pdf"

    with pytest.raises(RuntimeError):
        download(session, "http://x/y.pdf", dest, sleep=lambda s: None)

    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_creates_parent_dirs(tmp_path: Path):
    session = MagicMock()
    session.get.return_value = _StreamingResp([b"x"])
    dest = tmp_path / "deep" / "nested" / "dir" / "f.pdf"

    download(session, "http://x/f.pdf", dest, sleep=lambda s: None)

    assert dest.exists()
    assert dest.parent.is_dir()


def test_fetch_text_passes_through_and_sleeps():
    session = MagicMock()
    fake_resp = MagicMock()
    fake_resp.text = "<html>hi</html>"
    fake_resp.raise_for_status = lambda: None
    session.get.return_value = fake_resp
    sleeps: list[float] = []

    text = fetch_text(session, "http://x/index", sleep=sleeps.append, rate_limit_s=0.5)

    assert text == "<html>hi</html>"
    assert sleeps == [0.5]
    session.get.assert_called_once()


def test_fetch_text_raises_on_http_error():
    import requests

    session = MagicMock()
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock(side_effect=requests.HTTPError("404"))
    session.get.return_value = fake_resp

    with pytest.raises(requests.HTTPError):
        fetch_text(session, "http://x/404", sleep=lambda s: None)


def test_sha256_of_matches_hashlib(tmp_path: Path):
    p = tmp_path / "f.bin"
    payload = b"the quick brown fox" * 1000
    p.write_bytes(payload)

    assert sha256_of(p) == hashlib.sha256(payload).hexdigest()
