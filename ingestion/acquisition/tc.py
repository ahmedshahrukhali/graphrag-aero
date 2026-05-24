"""Transport Canada Advisory Circulars scraper.

The AC index page links to *detail* pages, one per AC. Each detail page links
to one (or more) PDF under ``https://tc.canada.ca/sites/default/files/...``.
So acquisition is a two-step crawl: index -> detail URLs -> PDFs.

Filenames in the PDF URL already encode series + issue + revision + date.
"""
from __future__ import annotations

from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

INDEX_URL_EN = "https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars"
INDEX_URL_FR = "https://tc.canada.ca/fr/aviation/centre-reference/circulaires-information"

_TC_HOST_SUFFIX = "tc.canada.ca"


def extract_ac_detail_urls(html: str, index_url: str) -> list[str]:
    """Return absolute, deduplicated AC-detail-page URLs linked from an index page.

    A detail page is any same-host link whose path lives one level below the
    index path (i.e., starts with ``<index_path>/``) and isn't itself a PDF.
    This is language-agnostic: it works for both
    ``/en/.../advisory-circulars/advisory-circular-ac-*`` and
    ``/fr/.../circulaires-information/circulaire-information-ci-*``.
    """
    base_path = urlparse(index_url).path.rstrip("/") + "/"
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        abs_url = urljoin(index_url, a["href"])
        parsed = urlparse(abs_url)
        if not parsed.netloc.endswith(_TC_HOST_SUFFIX):
            continue
        if not parsed.path.startswith(base_path):
            continue
        if parsed.path.lower().endswith(".pdf"):
            continue
        # Strip fragments/queries — we only want the canonical detail URL.
        urls.add(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")
    return sorted(urls)


def extract_pdf_urls(html: str, base_url: str) -> list[str]:
    """Return absolute, deduplicated PDF URLs on tc.canada.ca linked from this page."""
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        abs_url = urljoin(base_url, a["href"])
        parsed = urlparse(abs_url)
        if not parsed.netloc.endswith(_TC_HOST_SUFFIX):
            continue
        if not parsed.path.lower().endswith(".pdf"):
            continue
        urls.add(abs_url)
    return sorted(urls)


def filename_for(url: str) -> str:
    """Last path segment of ``url``, URL-decoded, safe to use as a local filename.

    TC sometimes percent-encodes UTF-8 in the path (e.g. French apostrophes,
    accented capitals). We decode so on-disk names match the canonical title.
    """
    return unquote(urlparse(url).path).rsplit("/", 1)[-1]
