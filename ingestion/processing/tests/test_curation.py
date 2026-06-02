"""Offline tests for ingestion/processing/curation.py (REINGEST_PLAN §3 frozen rules)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.processing.chunk import Chunk
from ingestion.processing.curation import (
    BOILERPLATE_CHUNK_CHARS,
    MIN_DOC_CHARS,
    ZH_ASCII_LETTER_THRESHOLD,
    Admission,
    CurationManifest,
    RejectReason,
    admit,
    is_boilerplate_chunk,
)
from ingestion.processing.doc_id import DocRef


# ── Helpers ──────────────────────────────────────────────────────────────────

_DUMMY_PATH = Path("data/corpus/en/tsb/a00a0001.pdf")
_ZERO_BBOX: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _ref(lang: str = "en", source: str = "tsb", corpus: str = "tsb") -> DocRef:
    return DocRef(
        doc_id=f"{source}/dummy",
        source=source,
        stem="dummy",
        lang=lang,
        source_url=None,
        path=_DUMMY_PATH,
        corpus=corpus,
    )


def _chunk(text: str) -> Chunk:
    from ingestion.processing.dedup import chunk_hash as _hash
    return Chunk(
        text=text,
        page=1,
        bbox=_ZERO_BBOX,
        section_title="",
        chunk_hash=_hash(text),
    )


def _good_chunks(n: int = 3) -> list[Chunk]:
    """Produce n chunks each with enough substantive text to be admitted."""
    return [_chunk(f"substantive aviation safety content chunk {i} " * 10) for i in range(n)]


# ── is_boilerplate_chunk ──────────────────────────────────────────────────────

def test_boilerplate_short_chunk():
    assert is_boilerplate_chunk(_chunk("short"))


def test_boilerplate_page_marker():
    assert is_boilerplate_chunk(_chunk("- 2 -"))
    assert is_boilerplate_chunk(_chunk("– 12 –"))
    assert is_boilerplate_chunk(_chunk("- 7"))


def test_boilerplate_date_only():
    assert is_boilerplate_chunk(_chunk("26 JULY 2003"))
    assert is_boilerplate_chunk(_chunk("2003-07-26"))
    assert is_boilerplate_chunk(_chunk("March 2024"))


def test_not_boilerplate_substantive():
    text = "a" * (BOILERPLATE_CHUNK_CHARS + 10)
    assert not is_boilerplate_chunk(_chunk(text))


# ── admit — EMPTY ─────────────────────────────────────────────────────────────

def test_admit_empty_chunks_rejected():
    result = admit(_ref(), [])
    assert not result.admitted
    assert result.reason == RejectReason.EMPTY


# ── admit — SUB_THRESHOLD ─────────────────────────────────────────────────────

def test_admit_sub_threshold_rejected():
    # Single chunk with fewer chars than MIN_DOC_CHARS.
    tiny = _chunk("x" * (MIN_DOC_CHARS - 1))
    result = admit(_ref(), [tiny])
    assert not result.admitted
    assert result.reason == RejectReason.SUB_THRESHOLD


def test_admit_exactly_at_threshold_passes():
    # Total text == MIN_DOC_CHARS should pass the sub-threshold gate.
    chunk = _chunk("x" * MIN_DOC_CHARS)
    # Still needs to not be boilerplate — make it long enough.
    assert admit(_ref(), [chunk]).admitted


# ── admit — COVER_ONLY ────────────────────────────────────────────────────────

def test_admit_cover_only_rejected():
    # Many boilerplate chunks whose *total* text exceeds MIN_DOC_CHARS so
    # SUB_THRESHOLD doesn't fire first, but every individual chunk is a page marker.
    n = MIN_DOC_CHARS // 5 + 5   # enough chunks so sum >= MIN_DOC_CHARS
    boilerplate = [_chunk(f"- {i} -") for i in range(n)]
    result = admit(_ref(), boilerplate)
    assert not result.admitted
    assert result.reason == RejectReason.COVER_ONLY


def test_admit_one_good_chunk_passes_cover_only():
    # One substantive chunk is enough to pass cover_only.
    chunks = [_chunk("- 2 -"), _chunk("a" * (BOILERPLATE_CHUNK_CHARS + 50) * 3)]
    result = admit(_ref(), chunks)
    # May still fail sub_threshold — that's checked before cover_only.
    # For this test make the good chunk supply enough chars.
    total = sum(len(c.text) for c in chunks)
    if total >= MIN_DOC_CHARS:
        assert result.reason != RejectReason.COVER_ONLY


# ── admit — LANG_MISDETECT ────────────────────────────────────────────────────

def test_admit_zh_mostly_ascii_rejected():
    # A ZH doc whose text is entirely Latin — lang-misdetect.
    zh_ref = _ref(lang="zh", source="ttsb", corpus="ttsb")
    ascii_chunks = _good_chunks(3)  # text is all ASCII
    result = admit(zh_ref, ascii_chunks)
    assert not result.admitted
    assert result.reason == RejectReason.LANG_MISDETECT


def test_admit_zh_chinese_text_admitted():
    zh_ref = _ref(lang="zh", source="ttsb", corpus="ttsb")
    # Mostly Chinese characters — should not trigger lang-misdetect.
    chinese_text = "本報告敘述台灣交通安全委員會調查航空事故之結果。" * 20
    chunks = [_chunk(chinese_text)]
    result = admit(zh_ref, chunks)
    assert result.admitted


def test_admit_en_doc_not_checked_for_lang():
    # EN docs are never subject to the lang-misdetect check.
    en_ref = _ref(lang="en", source="tsb")
    result = admit(en_ref, _good_chunks())
    assert result.admitted


# ── admit — good doc passes ───────────────────────────────────────────────────

def test_admit_good_en_doc():
    result = admit(_ref(), _good_chunks())
    assert result.admitted
    assert result.reason is None


# ── CurationManifest ─────────────────────────────────────────────────────────

def test_manifest_records_admitted():
    m = CurationManifest()
    m.record(_ref(lang="en", source="tsb", corpus="tsb"), Admission(True))
    assert m.admitted == 1
    assert m.rejected == 0
    assert m.by_corpus["tsb"]["admitted"] == 1
    assert m.by_lang["en"]["admitted"] == 1


def test_manifest_records_rejected():
    m = CurationManifest()
    m.record(_ref(lang="zh", source="caac", corpus="caac"),
             Admission(False, RejectReason.LANG_MISDETECT))
    assert m.rejected == 1
    assert m.reject_reasons["lang_misdetect"] == 1
    assert m.by_corpus["caac"]["rejected"] == 1
    assert m.by_lang["zh"]["rejected"] == 1


def test_manifest_to_dict_has_version():
    m = CurationManifest()
    d = m.to_dict()
    assert d["curation_version"] >= 1
    assert "admitted" in d and "rejected" in d and "total" in d


def test_manifest_balance_warning_none_when_balanced():
    m = CurationManifest()
    # 3 EN/TC + 3 ZH → ratio 1.0 — inside [0.5, 2.0].
    for _ in range(3):
        m.record(_ref(lang="en", corpus="tsb"), Admission(True))
    for _ in range(3):
        m.record(_ref(lang="zh", corpus="caac"), Admission(True))
    assert m.balance_warning() is None


def test_manifest_balance_warning_when_imbalanced():
    m = CurationManifest()
    # 10 EN + 1 ZH → ratio 0.1 — outside [0.5, 2.0].
    for _ in range(10):
        m.record(_ref(lang="en", corpus="tsb"), Admission(True))
    m.record(_ref(lang="zh", corpus="caac"), Admission(True))
    assert m.balance_warning() is not None


def test_manifest_counts_add_up():
    m = CurationManifest()
    m.record(_ref(), Admission(True))
    m.record(_ref(), Admission(False, RejectReason.EMPTY))
    assert m.to_dict()["total"] == m.admitted + m.rejected == 2
