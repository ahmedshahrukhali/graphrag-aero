"""Taiwan Transportation Safety Board (TTSB) aviation report scraper.

Distinct from the Canada ``tsb.py`` — this is Taiwan's TTSB (formerly the
Aviation Safety Council, ASC), the source of our **Traditional Chinese** corpus.

Unlike TC (index → detail → PDF), the TTSB Chinese listing pages expose the
report PDFs *directly* as ``/media/{id}/{name}.pdf`` links alongside the per-case
detail (``/post``) links — verified live 2026-06. So acquisition is a single-step
crawl: fetch a listing page, harvest its ``/media/.../*.pdf`` links, download.

The numeric ``{id}`` in ``/media/{id}/`` is the stable unique key: report PDF
basenames repeat across cases (ASC-era files like ``00_general.pdf`` /
``ci611_general.pdf``), so the local filename prefixes the media id to stay
collision-free and to preserve provenance.

Older ASC-era reports tend to be **scanned** (image-only pages) — exactly what we
want to exercise the Chinese OCR path.
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

# Traditional-Chinese listing pages that link report PDFs directly. The English
# tree (/english/...) serves English PDFs — we deliberately use the Chinese tree.
INDEX_URLS = (
    "https://www.ttsb.gov.tw/1133/1154/1155/1159/Lpsimplelist",  # 調查完成事故 Completed
    "https://www.ttsb.gov.tw/1133/1154/1155/1157/Lpsimplelist",  # 重大調查案件 Major
)

_TTSB_HOST_SUFFIX = "ttsb.gov.tw"
_MEDIA_ID_RE = re.compile(r"/media/(\d+)/")


def extract_pdf_urls(html: str, base_url: str) -> list[str]:
    """Return absolute, deduplicated ``/media/{id}/*.pdf`` URLs on ttsb.gov.tw.

    Restricting to the ``/media/{digits}/`` prefix keeps us to report assets and
    skips unrelated PDFs (forms, logos).
    """
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        abs_url = urljoin(base_url, a["href"])
        parsed = urlparse(abs_url)
        if not parsed.netloc.endswith(_TTSB_HOST_SUFFIX):
            continue
        if not parsed.path.lower().endswith(".pdf"):
            continue
        if not _MEDIA_ID_RE.search(parsed.path):
            continue
        # Drop fragments/queries — keep the canonical asset URL.
        urls.add(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
    return sorted(urls)


def media_id(url: str) -> str | None:
    """The numeric id in ``/media/{id}/`` — the report's stable unique key."""
    m = _MEDIA_ID_RE.search(urlparse(url).path)
    return m.group(1) if m else None


def filename_for(url: str) -> str:
    """Collision-free local filename: ``{media_id}_{decoded basename}``.

    Basenames repeat across reports; the media id disambiguates and preserves the
    source pointer. Percent-encoded UTF-8 (Traditional Chinese) is decoded so the
    on-disk name is readable.
    """
    path = unquote(urlparse(url).path)
    basename = path.rsplit("/", 1)[-1]
    mid = media_id(url)
    return f"{mid}_{basename}" if mid else basename
