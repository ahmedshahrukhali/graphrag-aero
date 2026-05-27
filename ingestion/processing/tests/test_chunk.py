"""Tests for the chunker. The real BGE-M3 tokenizer is replaced with a small
whitespace tokenizer that exposes the same ``encode(text) -> .ids/.offsets``
shape, so tests stay offline.
"""
from dataclasses import dataclass

from ingestion.processing.chunk import MIN_USABLE_BBOX_AREA, Chunk, chunk_pages
from ingestion.processing.dedup import chunk_hash
from ingestion.processing.pdf import Char, PageExtract


@dataclass
class _Enc:
    ids: list[int]
    offsets: list[tuple[int, int]]


class _WhitespaceTokenizer:
    """One token per whitespace-delimited word; offsets are char spans."""

    def encode(self, text: str, add_special_tokens: bool = True) -> _Enc:
        # We don't emit special tokens; the kwarg is accepted for protocol parity.
        _ = add_special_tokens
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            while i < n and not text[i].isspace():
                i += 1
            offsets.append((start, i))
            ids.append(len(ids))
        return _Enc(ids=ids, offsets=offsets)


def _chars_for(text: str, page: int, *, x0: float = 0.0, y0: float = 0.0,
               char_w: float = 5.0, char_h: float = 10.0,
               size: float = 12.0) -> list[Char]:
    """Lay out chars left-to-right on one line. Newlines reset and advance y."""
    out: list[Char] = []
    cx, cy = x0, y0
    for ch in text:
        if ch == "\n":
            cx = x0
            cy += char_h + 2
            continue
        out.append(Char(text=ch, x0=cx, x1=cx + char_w, top=cy, bottom=cy + char_h,
                        size=size, page=page))
        cx += char_w
    return out


def test_empty_pages_yields_no_chunks():
    assert chunk_pages([], _WhitespaceTokenizer()) == []


def test_whitespace_only_pages_yield_no_chunks():
    page = PageExtract(page=1, text="   \n  ", chars=[])
    assert chunk_pages([page], _WhitespaceTokenizer()) == []


def test_single_short_page_one_chunk():
    text = "Findings the engine failed at altitude"
    page = PageExtract(page=1, text=text, chars=_chars_for(text, page=1))
    chunks = chunk_pages([page], _WhitespaceTokenizer())
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text == text
    assert c.page == 1
    assert c.chunk_hash == chunk_hash(text)
    # bbox is the union of all char bboxes on page 1.
    assert c.bbox[0] == 0.0
    assert c.bbox[2] > 0.0


def test_overlapping_windows_have_expected_token_overlap():
    # 10 tokens, window=4, overlap=2 -> step=2 -> windows at 0,2,4,6 (=4 chunks).
    text = "a b c d e f g h i j"
    page = PageExtract(page=1, text=text, chars=_chars_for(text, page=1))
    chunks = chunk_pages([page], _WhitespaceTokenizer(), window=4, overlap=2)
    texts = [c.text for c in chunks]
    assert texts == ["a b c d", "c d e f", "e f g h", "g h i j"]


def test_section_title_propagates_to_following_chunks():
    # Page 1: a big-font "HEADER" line, then body. Smaller window so we get >1 chunk.
    body = "the engine quit"
    text = "HEADER\n" + body
    chars = (
        _chars_for("HEADER", page=1, size=30.0)
        + _chars_for("\n" + body, page=1, size=10.0, y0=12)
    )
    page = PageExtract(page=1, text=text, chars=chars)
    chunks = chunk_pages([page], _WhitespaceTokenizer(), window=2, overlap=0)
    # First chunk should already be inside the body (HEADER token comes first).
    # All chunks AFTER the header should carry section_title == "HEADER".
    assert any(c.section_title == "HEADER" for c in chunks)


def test_chunk_picks_dominant_page_for_bbox():
    # Page 1: 1 token. Page 2: 3 tokens. A chunk spanning both prefers page 2.
    p1_text = "alpha"
    p2_text = "beta gamma delta"
    p1 = PageExtract(page=1, text=p1_text, chars=_chars_for(p1_text, page=1))
    p2 = PageExtract(page=2, text=p2_text, chars=_chars_for(p2_text, page=2, y0=100))
    # Window covers all 4 tokens.
    chunks = chunk_pages([p1, p2], _WhitespaceTokenizer(), window=8, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page == 2
    # bbox should reflect page-2 chars (y0 ~ 100 vs page-1 ~ 0).
    assert chunks[0].bbox[1] >= 100.0


def test_chunk_hash_is_stable_for_same_text():
    text = "Findings the engine failed at altitude"
    page = PageExtract(page=1, text=text, chars=_chars_for(text, page=1))
    a = chunk_pages([page], _WhitespaceTokenizer())[0].chunk_hash
    b = chunk_pages([page], _WhitespaceTokenizer())[0].chunk_hash
    assert a == b


def test_last_chunk_smaller_than_window_still_emitted():
    text = "a b c d e"  # 5 tokens
    page = PageExtract(page=1, text=text, chars=_chars_for(text, page=1))
    chunks = chunk_pages([page], _WhitespaceTokenizer(), window=4, overlap=1)
    # Step = 3. Windows start at 0, 3 -> emit [a b c d], [d e]. Both are kept.
    assert [c.text for c in chunks] == ["a b c d", "d e"]


def test_bbox_fallback_expands_tiny_chunk_bbox():
    """Cross-page chunk: dominant page has only a few chars in the window (tiny bbox).
    The fallback must expand to the full page extent so highlights aren't microscopic.
    """
    # Two words: "ab" at (0,0)..(10,10) and "cd" far away at (500,490)..(510,500).
    # With window=1 each word is its own chunk — both have area=100 pt² < MIN_USABLE_BBOX_AREA.
    # Fallback: use all chars on the page → extent = (0, 0, 510, 500).
    chars = [
        Char(text="a", x0=0, x1=5, top=0, bottom=10, size=12, page=1),
        Char(text="b", x0=5, x1=10, top=0, bottom=10, size=12, page=1),
        Char(text="c", x0=500, x1=505, top=490, bottom=500, size=12, page=1),
        Char(text="d", x0=505, x1=510, top=490, bottom=500, size=12, page=1),
    ]
    page = PageExtract(page=1, text="ab cd", chars=chars)
    chunks = chunk_pages([page], _WhitespaceTokenizer(), window=1, overlap=0)
    assert len(chunks) == 2
    for c in chunks:
        area = (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1])
        assert area >= MIN_USABLE_BBOX_AREA, (
            f"Expected fallback to full page extent (area>={MIN_USABLE_BBOX_AREA}), got bbox={c.bbox}"
        )
        # Full page extent must cover both word positions.
        assert c.bbox[0] <= 0.0 and c.bbox[2] >= 510.0
        assert c.bbox[1] <= 0.0 and c.bbox[3] >= 500.0
