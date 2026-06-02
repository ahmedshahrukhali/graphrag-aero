"""Curation admission rules for the curated re-ingest (REINGEST_PLAN §3).

Rules are **FROZEN** at version 1. Change a constant → bump VERSION and update §3.

Closed-set reject reasons
--------------------------
empty           — no chunks extracted from the document
sub_threshold   — total extracted text < MIN_DOC_CHARS characters
cover_only      — every chunk is cover-page / boilerplate noise
lang_misdetect  — declared ZH doc whose text is >80 % ASCII letters
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .chunk import Chunk
from .doc_id import DocRef

# ── FROZEN thresholds (version 1) ────────────────────────────────────────────

VERSION = 1

#: Minimum total extracted characters for a document to be admitted.
MIN_DOC_CHARS: int = 200

#: A chunk whose stripped text is shorter than this is a boilerplate candidate.
BOILERPLATE_CHUNK_CHARS: int = 80

#: A ZH doc is lang-misdetected when this fraction of its non-space chars are
#: ASCII letters.  0.80 = 80 %.
ZH_ASCII_LETTER_THRESHOLD: float = 0.80

#: Informational balance target: ZH-admitted : EN_TC-admitted should stay inside
#: this band so the cross-lingual overlap demo is honest.
BALANCE_RATIO_BAND: tuple[float, float] = (0.5, 2.0)

# ── Patterns ─────────────────────────────────────────────────────────────────

# "- 2 -", "- 7", "– 12 –" — running page-number lines.
_PAGE_MARKER_RE = re.compile(r"^\s*[-–]\s*\d+\s*[-–]?\s*$")

# Date-only lines ("26 JULY 2003", "2003-07-26", "MARCH 2024").
_MONTHS = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)
_DATE_ONLY_RE = re.compile(
    rf"^\s*\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}\s*$"
    rf"|^\s*\d{{4}}-\d{{2}}-\d{{2}}\s*$"
    rf"|^\s*(?:{_MONTHS})\s+\d{{4}}\s*$",
    re.I,
)


# ── Admission result ─────────────────────────────────────────────────────────

class RejectReason(str, Enum):
    EMPTY = "empty"
    SUB_THRESHOLD = "sub_threshold"
    COVER_ONLY = "cover_only"
    LANG_MISDETECT = "lang_misdetect"


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: RejectReason | None = None  # None when admitted


_ADMITTED = Admission(admitted=True)


# ── Per-chunk helper ─────────────────────────────────────────────────────────

def is_boilerplate_chunk(chunk: Chunk) -> bool:
    """True when the chunk carries only header/footer noise, not body content."""
    text = chunk.text.strip()
    if len(text) < BOILERPLATE_CHUNK_CHARS:
        return True
    if _PAGE_MARKER_RE.match(text):
        return True
    if _DATE_ONLY_RE.match(text):
        return True
    return False


# ── Admission predicate ───────────────────────────────────────────────────────

def admit(ref: DocRef, chunks: list[Chunk]) -> Admission:
    """Return an :class:`Admission` for the document at ``ref``.

    Rules are tested in priority order; the first failure short-circuits.
    """
    if not chunks:
        return Admission(False, RejectReason.EMPTY)

    total_chars = sum(len(c.text) for c in chunks)
    if total_chars < MIN_DOC_CHARS:
        return Admission(False, RejectReason.SUB_THRESHOLD)

    if all(is_boilerplate_chunk(c) for c in chunks):
        return Admission(False, RejectReason.COVER_ONLY)

    if ref.lang == "zh":
        all_text = "".join(c.text for c in chunks)
        non_space = [ch for ch in all_text if not ch.isspace()]
        if non_space:
            ascii_frac = sum(1 for ch in non_space if ch.isascii() and ch.isalpha()) / len(non_space)
            if ascii_frac > ZH_ASCII_LETTER_THRESHOLD:
                return Admission(False, RejectReason.LANG_MISDETECT)

    return _ADMITTED


# ── Manifest accumulator ──────────────────────────────────────────────────────

@dataclass
class CurationManifest:
    """Running tally for one curated ingest run."""

    admitted: int = 0
    rejected: int = 0
    reject_reasons: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    by_corpus: dict[str, dict] = field(default_factory=dict)
    by_lang: dict[str, dict] = field(default_factory=dict)

    def _bucket(self, d: dict, key: str) -> dict:
        if key not in d:
            d[key] = {"admitted": 0, "rejected": 0}
        return d[key]

    def record(self, ref: DocRef, result: Admission) -> None:
        """Register one document's admission result."""
        slot = "admitted" if result.admitted else "rejected"
        if result.admitted:
            self.admitted += 1
        else:
            self.rejected += 1
            self.reject_reasons[result.reason.value] += 1  # type: ignore[index]
        self._bucket(self.by_corpus, ref.corpus)[slot] += 1
        self._bucket(self.by_lang, ref.lang)[slot] += 1

    def balance_warning(self) -> str | None:
        """Non-None if the ZH:EN_TC admitted ratio is outside the target band."""
        zh = self.by_lang.get("zh", {}).get("admitted", 0)
        en_tc = sum(
            self.by_lang.get(lang, {}).get("admitted", 0) for lang in ("en", "fr")
        )
        if en_tc == 0 or zh == 0:
            return None
        ratio = zh / en_tc
        lo, hi = BALANCE_RATIO_BAND
        if not (lo <= ratio <= hi):
            return (
                f"ZH:EN_TC admitted ratio {ratio:.2f} outside target band "
                f"[{lo:.2f}, {hi:.2f}] — adjust corpus size."
            )
        return None

    def to_dict(self) -> dict:
        d: dict = {
            "curation_version": VERSION,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "total": self.admitted + self.rejected,
            "reject_reasons": dict(self.reject_reasons),
            "by_corpus": dict(self.by_corpus),
            "by_lang": dict(self.by_lang),
        }
        warning = self.balance_warning()
        if warning:
            d["balance_warning"] = warning
        return d
