"""Polite HTTP utilities: rate-limited fetch + streaming download with SHA-256."""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

USER_AGENT = (
    "graphrag-aero/0.1 (data acquisition for academic research; "
    "contact via repo issues)"
)


@dataclass
class DownloadResult:
    url: str
    path: Path
    downloaded: bool   # False = file already existed on disk, skipped
    sha256: str
    size_bytes: int


def make_session(user_agent: str = USER_AGENT, retries: int = 3) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_text(
    session: requests.Session,
    url: str,
    *,
    rate_limit_s: float = 1.0,
    timeout_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    sleep(rate_limit_s)
    r = session.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def download(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    rate_limit_s: float = 1.0,
    timeout_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Stream-download ``url`` to ``dest``. Skip if ``dest`` already exists non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("skip (exists): %s", dest)
        return DownloadResult(
            url=url,
            path=dest,
            downloaded=False,
            sha256=sha256_of(dest),
            size_bytes=dest.stat().st_size,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    sleep(rate_limit_s)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    size = 0
    try:
        with session.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
        tmp.replace(dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise

    logger.info("downloaded: %s (%d bytes)", dest, size)
    return DownloadResult(
        url=url,
        path=dest,
        downloaded=True,
        sha256=h.hexdigest(),
        size_bytes=size,
    )
