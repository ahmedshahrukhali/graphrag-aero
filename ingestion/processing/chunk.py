"""Fixed-size 512-token chunker with 128-token overlap.

Tokenizer: BGE-M3 (XLM-RoBERTa) — boundaries match what the embedder will see.

For each chunk we carry:
    - text             (the chunk's literal text)
    - page             (the dominant page the chunk's chars come from)
    - bbox             (union of contributing chars' boxes on that page)
    - section_title    (most recent header-style line seen ≤ chunk's start)
    - chunk_hash       (sha256 of normalized text, for cross-doc dedup)

Section title detection uses content patterns:
  TSB section headings (EN + FR) and numbered TC AC section lines 
  (e.g., "1.2 Title") are recognised using deterministic regex patterns.
  This avoids false positives from large cover/date boilerplate text.
"""
from __future__ import annotations

import bisect
import re
import statistics
from dataclasses import dataclass
from typing import Protocol

from .dedup import chunk_hash
from .pdf import Char, PageExtract


WINDOW_TOKENS = 512
OVERLAP_TOKENS = 128
HEADER_SIZE_RATIO = 1.2
HEADER_MAX_CHARS = 120

# Content-based section header patterns — fallback when font-size heuristic misses.
# TSB finding/risk/safety headings (EN + FR).
_TSB_SECTION_RE = re.compile(
    r"findings\s+as\s+to\s+causes?(?:\s+and\s+contributing\s+factors)?"
    r"|findings\s+as\s+to\s+risk"
    r"|safety\s+action\b"
    r"|faits\s+établis\s+quant\s+aux\s+causes?"
    r"|faits\s+établis\s+quant\s+aux\s+risques?"
    r"|mesures\s+de\s+sécurité\b",
    re.I,
)
# Numbered TC AC sections: "1.2 Title", "3.1.1 Scope", "1. Introduction".
_NUMBERED_SECTION_RE = re.compile(
    r"^\d{1,2}(?:\.\d{1,2}){0,2}\.?\s{1,4}[A-Z][A-Za-zÀ-ÿ]"
)
PAGE_SEP = "\n\n"   # inserted between pages in the joined stream
# Minimum bbox area in PDF pt² before we fall back to the full page-text extent.
# Cross-page chunks often land only a page-number line on the dominant page,
# giving a bbox like [72, 41, 302, 54] (area ≈ 2.8k pt²) that is useless for
# highlighting.  Below this threshold we widen to all chars on that page.
MIN_USABLE_BBOX_AREA = 5000.0


# ─── tokenizer protocol ──────────────────────────────────────────────────────
# We accept any object with `encode(text)` returning an object that exposes
# `ids: list[int]` and `offsets: list[tuple[int,int]]` (char offsets into the
# input). That matches HuggingFace's ``tokenizers.Tokenizer`` AND lets us fake
# it in tests without pulling in the real library.

class _Encoding(Protocol):
    ids: list[int]
    offsets: list[tuple[int, int]]


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> _Encoding: ...


# ─── data shapes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    section_title: str
    chunk_hash: str
    # Region-level grounding (WS-0): one rect per page the chunk touches, as
    # (page, x0, top, x1, bottom). ``page``/``bbox`` above stay as the dominant
    # page + its rect for backward compatibility; ``page_bboxes`` is the full,
    # deterministic grounding payload rendered directly at the UI (no re-search).
    page_bboxes: tuple[tuple[int, float, float, float, float], ...] = ()
    # Discriminator for the figure tier (§4.2). Text chunks are "text"; a
    # figure-blurb chunk minted by figures.py will carry "figure".
    kind: str = "text"


# ─── joining pages ───────────────────────────────────────────────────────────

@dataclass
class _Joined:
    text: str                                   # full corpus text, pages glued with PAGE_SEP
    # For each character index in ``text``: which page it belongs to, and its
    # bbox if known. Pages' separator chars get page=0 / bbox=None.
    char_pages: list[int]
    char_bbox: list[tuple[float, float, float, float] | None]
    # Sorted (offset, title) pairs — section title in force from that char onward.
    header_offsets: list[int]
    header_titles: list[str]


def _join_pages(pages: list[PageExtract]) -> _Joined:
    parts: list[str] = []
    char_pages: list[int] = []
    char_bbox: list[tuple[float, float, float, float] | None] = []
    headers: list[tuple[int, str]] = []

    headers: list[tuple[int, str]] = []

    cursor = 0
    for i, page in enumerate(pages):
        if i > 0:
            parts.append(PAGE_SEP)
            for ch in PAGE_SEP:
                char_pages.append(0)
                char_bbox.append(None)
            cursor += len(PAGE_SEP)

        page_text = page.text
        page_start = cursor

        # Map each pdfplumber char to its position in the page's text.
        # pdfplumber's per-page char order generally aligns with extract_text(),
        # but not always character-for-character (whitespace + newlines may be
        # inserted by the layout engine). We do a best-effort align by
        # consuming chars from the chars list as we walk page_text; non-matching
        # text positions get bbox=None.
        char_iter = iter(page.chars)
        next_char: Char | None = next(char_iter, None)
        for ch in page_text:
            parts.append(ch)
            char_pages.append(page.page)
            if next_char is not None and ch == next_char.text:
                char_bbox.append((next_char.x0, next_char.top, next_char.x1, next_char.bottom))
                next_char = next(char_iter, None)
            else:
                char_bbox.append(None)
            cursor += 1

        # Header detection: content patterns.
        line_start_offset = page_start
        for line in page_text.split("\n"):
            line_end_offset = line_start_offset + len(line)
            stripped = line.strip()
            if stripped and len(stripped) <= HEADER_MAX_CHARS:
                is_header = False
                if _TSB_SECTION_RE.search(stripped) or _NUMBERED_SECTION_RE.match(stripped):
                    is_header = True
                if is_header:
                    headers.append((line_start_offset, stripped))
            line_start_offset = line_end_offset + 1  # account for the "\n"

    headers.sort(key=lambda x: x[0])
    return _Joined(
        text="".join(parts),
        char_pages=char_pages,
        char_bbox=char_bbox,
        header_offsets=[h[0] for h in headers],
        header_titles=[h[1] for h in headers],
    )


def _section_at(joined: _Joined, offset: int) -> str:
    """Most recent header title at-or-before ``offset``; empty string if none."""
    if not joined.header_offsets:
        return ""
    i = bisect.bisect_right(joined.header_offsets, offset) - 1
    if i < 0:
        return ""
    return joined.header_titles[i]


def _page_union_bbox(
    joined: _Joined, page: int, start: int, end: int
) -> tuple[float, float, float, float] | None:
    """Union bbox of ``page``'s chars within [start, end), with full-page fallback.

    Returns ``(x0, y0, x1, y1)`` in PDF points, or ``None`` if ``page`` has no
    positioned chars at all. If the windowed union is below
    ``MIN_USABLE_BBOX_AREA`` (e.g. a cross-page chunk that landed only a
    page-number line here), falls back to the extent of ALL chars on the page so
    the rendered region rect is never microscopic.
    """
    x0 = float("inf"); y0 = float("inf")
    x1 = float("-inf"); y1 = float("-inf")
    have = False
    for i in range(start, min(end, len(joined.char_pages))):
        if joined.char_pages[i] != page:
            continue
        bb = joined.char_bbox[i]
        if bb is None:
            continue
        have = True
        if bb[0] < x0: x0 = bb[0]
        if bb[1] < y0: y0 = bb[1]
        if bb[2] > x1: x1 = bb[2]
        if bb[3] > y1: y1 = bb[3]
    if have and (x1 - x0) * (y1 - y0) >= MIN_USABLE_BBOX_AREA:
        return (x0, y0, x1, y1)

    # Degenerate (no positioned chars in window, or area too small): fall back to
    # the extent of ALL chars on this page.
    fx0 = float("inf"); fy0 = float("inf")
    fx1 = float("-inf"); fy1 = float("-inf")
    for i in range(len(joined.char_pages)):
        if joined.char_pages[i] != page:
            continue
        bb = joined.char_bbox[i]
        if bb is None:
            continue
        if bb[0] < fx0: fx0 = bb[0]
        if bb[1] < fy0: fy0 = bb[1]
        if bb[2] > fx1: fx1 = bb[2]
        if bb[3] > fy1: fy1 = bb[3]
    if fx0 < float("inf"):
        return (fx0, fy0, fx1, fy1)
    return (x0, y0, x1, y1) if have else None


def _pages_in_range(joined: _Joined, start: int, end: int) -> list[int]:
    """Pages touched by [start, end), in reading order (ascending page number)."""
    pages: set[int] = set()
    for i in range(start, min(end, len(joined.char_pages))):
        p = joined.char_pages[i]
        if p:
            pages.add(p)
    return sorted(pages)


def _page_bboxes_for_range(
    joined: _Joined, start: int, end: int
) -> tuple[tuple[int, float, float, float, float], ...]:
    """Region-level grounding (WS-0): one ``(page, x0, y0, x1, y1)`` per page the
    chunk touches, in reading order. This is the entire grounding payload —
    bounded, deterministic, computed once at ingest, rendered directly."""
    out: list[tuple[int, float, float, float, float]] = []
    for page in _pages_in_range(joined, start, end):
        bb = _page_union_bbox(joined, page, start, end)
        if bb is not None:
            out.append((page, bb[0], bb[1], bb[2], bb[3]))
    return tuple(out)


def _bbox_for_range(joined: _Joined, start: int, end: int) -> tuple[int, tuple[float, float, float, float]]:
    """Pick the dominant page in [start, end) and the union bbox on that page.

    Returns ``(page, bbox)`` with ``bbox = (x0, y0, x1, y1)``. If no chars in
    the range had positional info, returns ``(page, (0, 0, 0, 0))``. Kept for
    backward compatibility — ``page``/``bbox`` are the dominant-page fields;
    full region grounding lives in :func:`_page_bboxes_for_range`.
    """
    page_counts: dict[int, int] = {}
    for i in range(start, min(end, len(joined.char_pages))):
        p = joined.char_pages[i]
        if p:
            page_counts[p] = page_counts.get(p, 0) + 1
    if not page_counts:
        return 0, (0.0, 0.0, 0.0, 0.0)
    dominant_page = max(page_counts.items(), key=lambda kv: kv[1])[0]
    bb = _page_union_bbox(joined, dominant_page, start, end)
    return dominant_page, bb if bb is not None else (0.0, 0.0, 0.0, 0.0)


# ─── public API ──────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[PageExtract],
    tokenizer: Tokenizer,
    *,
    window: int = WINDOW_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """Tokenize joined page text and emit overlapping chunks with metadata."""
    if not pages:
        return []
    joined = _join_pages(pages)
    if not joined.text.strip():
        return []

    # add_special_tokens=False: BGE-M3 wraps inputs with <s>/</s> by default,
    # giving zero-width (0,0) offsets that corrupt the char-window math. We
    # window pure content tokens. The embed step will add specials back, so
    # final encoded chunks are 512 + 2 = 514 tokens — within model limits.
    enc = tokenizer.encode(joined.text, add_special_tokens=False)
    offsets = enc.offsets
    n = len(offsets)
    if n == 0:
        return []
    step = max(1, window - overlap)

    chunks: list[Chunk] = []
    start_tok = 0
    while start_tok < n:
        end_tok = min(start_tok + window, n)
        char_start = offsets[start_tok][0]
        # offsets[end_tok-1][1] is the (exclusive) char end of the last token.
        char_end = offsets[end_tok - 1][1]
        text = joined.text[char_start:char_end]
        if text.strip():
            page, bbox = _bbox_for_range(joined, char_start, char_end)
            page_bboxes = _page_bboxes_for_range(joined, char_start, char_end)
            section = _section_at(joined, char_start)
            chunks.append(Chunk(
                text=text,
                page=page,
                bbox=bbox,
                section_title=section,
                chunk_hash=chunk_hash(text),
                page_bboxes=page_bboxes,
            ))
        if end_tok == n:
            break
        start_tok += step
    return chunks
