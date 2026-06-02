"""CAAC (Civil Aviation Administration of China) Simplified-Chinese AC scraper.

CAAC's on-site document index is a JS/JSONP widget (TRS WAS5) that can't be
scraped, but the PDFs themselves download fine. So instead of crawling an index
we read a **committed seed manifest** of direct PDF URLs (``caac_seed.txt``,
harvested once via search engine — see that file's header) and download each.

Reproducible (no live-search dependency at run time) and idempotent (the
streaming ``download`` skips files already on disk). Simplified Chinese; older
ACs are scanned → they exercise the ``ch`` PaddleOCR path in ``processing/``.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

# Seed lives with the module (committed) — data/corpus/ is gitignored, so the
# manifest can't live under the corpus tree it feeds.
DEFAULT_SEED = Path(__file__).with_name("caac_seed.txt")

_CAAC_HOST_SUFFIX = "caac.gov.cn"


def filename_for(url: str) -> str:
    """Local filename = the PDF's basename (the opaque ``P{digits}.pdf``).

    The P-number encodes date + sequence and is unique per document, so it needs
    no disambiguating prefix. Percent-encoding is decoded for readability.
    """
    return unquote(urlparse(url).path).rsplit("/", 1)[-1]


def load_seed(text: str) -> list[str]:
    """Parse a seed manifest into a deduplicated list of CAAC PDF URLs.

    Skips blanks and ``#`` comments; keeps only ``caac.gov.cn`` ``*.pdf`` URLs;
    dedupes by basename (mirror URLs — ``/PHONE/``, ``/big5/`` — share the same
    P-number and collapse to one download). Order is preserved (first wins).
    """
    urls: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = urlparse(line)
        if not parsed.netloc.endswith(_CAAC_HOST_SUFFIX):
            continue
        if not parsed.path.lower().endswith(".pdf"):
            continue
        name = filename_for(line)
        if name in seen:
            continue
        seen.add(name)
        urls.append(line)
    return urls


def seed_url_map(urls: list[str]) -> dict[str, str]:
    """Return ``{basename → url}`` for a deduplicated list of seed URLs."""
    return {filename_for(u): u for u in urls}


def load_seed_file(path: Path = DEFAULT_SEED) -> list[str]:
    """Read + parse the seed manifest at ``path`` (empty list if it's missing)."""
    if not path.exists():
        return []
    return load_seed(path.read_text(encoding="utf-8"))
