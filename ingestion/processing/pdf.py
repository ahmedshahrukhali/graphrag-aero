"""pdfplumber wrapper.

For each page yields a :class:`PageExtract` carrying the extracted text, the
per-character list (used downstream for bbox aggregation), the page number
(1-indexed), and a flag indicating whether the page is image-only (no
extractable text but does contain images) — those go through the OCR fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pdfplumber


# A subset of pdfplumber's char dict we actually use downstream.
# pdfplumber gives us many more fields; we only need positional + font.
@dataclass
class Char:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float       # mean font size; used for the section-title heuristic
    page: int


@dataclass
class PageExtract:
    page: int                       # 1-indexed page number
    text: str                       # pdfplumber extract_text() — may be empty
    chars: list[Char] = field(default_factory=list)
    image_only: bool = False        # text empty + images present


def _coerce_char(d: dict[str, Any], page_no: int) -> Char:
    return Char(
        text=str(d.get("text", "")),
        x0=float(d.get("x0", 0.0)),
        x1=float(d.get("x1", 0.0)),
        top=float(d.get("top", 0.0)),
        bottom=float(d.get("bottom", 0.0)),
        size=float(d.get("size", 0.0)),
        page=page_no,
    )


def extract_page(page: Any, page_no: int) -> PageExtract:
    """Extract one pdfplumber page into our :class:`PageExtract` shape."""
    text = page.extract_text() or ""
    chars = [_coerce_char(c, page_no) for c in (page.chars or [])]
    image_only = (not text.strip()) and bool(page.images)
    return PageExtract(page=page_no, text=text, chars=chars, image_only=image_only)


@contextmanager
def open_pdf(path: Path):
    """Thin wrapper so callers (and tests) can monkeypatch the open call."""
    with pdfplumber.open(str(path)) as pdf:
        yield pdf


def iter_pages(path: Path) -> Iterator[PageExtract]:
    """Yield :class:`PageExtract` for each page in ``path``, in order."""
    with open_pdf(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            yield extract_page(page, i)
