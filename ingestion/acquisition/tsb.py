"""TSB aviation investigation report scraper.

The TSB report index links to per-report HTML pages whose URLs (and visible text)
include the occurrence ID, formatted as ``A<YY><region letter><4 digits>`` (e.g.
``A23H0001``). We harvest IDs from the index HTML and rebuild the PDF URL from
the documented pattern:

    https://www.bst-tsb.gc.ca/sites/default/files/rapports-reports/aviation/{id}/{lang}/{id}.pdf

with ``{lang}`` = ``eng`` or ``fra``.
"""
from __future__ import annotations

import re
from typing import Literal

from bs4 import BeautifulSoup

INDEX_URL_EN = "https://www.tsb.gc.ca/eng/rapports-reports/aviation/index.html"
INDEX_URL_FR = "https://www.tsb.gc.ca/fra/rapports-reports/aviation/index.html"

# Occurrence IDs: A<2-digit year><region letter A-Z><4 digits>. Case-insensitive.
REPORT_ID_RE = re.compile(r"\b([Aa]\d{2}[A-Za-z]\d{4})\b")

Lang = Literal["en", "fr"]


def extract_report_ids(html: str) -> list[str]:
    """Return all TSB occurrence IDs referenced in ``html``, uppercased and sorted."""
    soup = BeautifulSoup(html, "html.parser")
    ids: set[str] = set()
    for a in soup.find_all("a", href=True):
        for m in REPORT_ID_RE.finditer(a["href"]):
            ids.add(m.group(1).upper())
    for m in REPORT_ID_RE.finditer(soup.get_text(" ", strip=True)):
        ids.add(m.group(1).upper())
    return sorted(ids)


def build_pdf_url(report_id: str, lang: Lang) -> str:
    rid = report_id.lower()
    lang_path = "eng" if lang == "en" else "fra"
    return (
        "https://www.bst-tsb.gc.ca/sites/default/files/"
        f"rapports-reports/aviation/{rid}/{lang_path}/{rid}.pdf"
    )
