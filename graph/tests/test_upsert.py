"""Tests for graph/upsert.py — FakeDriver captures UNWIND batches."""
import json
from pathlib import Path

import pytest

from embed.jsonl import ChunkRecord
from graph.extract import ExtractedEntities, NoopExtractor
from graph.upsert import (
    _ac_id_from_doc_id,
    _dedup_rows,
    _occurrence_row,
    upsert_acs_from_chunks,
    upsert_entities_from_chunks,
    upsert_occurrences_from_chunks,
)


class FakeSession:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def run(self, cypher, **params):
        self.log.append((cypher, dict(params)))
        return iter(())


class FakeDriver:
    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def session(self, **kwargs):
        return FakeSession(self.runs)

    def close(self):
        pass


def _rec(doc_id: str, *, lang: str = "en", chunk_hash: str = "0" * 64,
         text: str = "x", page: int = 1) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        source_url=f"https://example.test/{doc_id}.pdf",
        section_title="", page=page, bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=chunk_hash, lang=lang, text=text,
    )


def _write_chunks(root: Path, doc_id: str, lang: str, n: int,
                  text: str = "x"):
    src, stem = doc_id.split("/", 1)
    path = root / lang / src / f"{stem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            r = {
                "doc_id": doc_id,
                "source_url": f"https://example.test/{doc_id}.pdf",
                "section_title": "",
                "page": i + 1,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "chunk_hash": f"{hash(doc_id + str(i)):064x}",
                "lang": lang,
                "text": text,
            }
            f.write(json.dumps(r) + "\n")


# ─── _occurrence_row / _dedup_rows / _ac_id_from_doc_id ──────────────────────

def test_occurrence_row_tsb():
    row = _occurrence_row(_rec("tsb/a00a0051", lang="en"))
    assert row == {
        "id": "a00a0051",
        "source_url": "https://example.test/tsb/a00a0051.pdf",
        "lang": "en",
    }


def test_occurrence_row_tc_is_skipped():
    assert _occurrence_row(_rec("tc/dan-001-e_0")) is None


def test_dedup_rows():
    rows = [
        {"id": "a", "source_url": "u", "lang": "en"},
        {"id": "a", "source_url": "u", "lang": "en"},
        {"id": "b", "source_url": "u", "lang": "fr"},
    ]
    assert len(_dedup_rows(rows)) == 2


def test_ac_id_from_doc_id():
    assert _ac_id_from_doc_id("tc/AC_702-001_ISSUE-1") == "702-001"
    assert _ac_id_from_doc_id("tc/AC_507-002_-_ISSUE_02") == "507-002"
    assert _ac_id_from_doc_id("tc/ac_300_008_issue_03") == "300-008"
    assert _ac_id_from_doc_id("tc/no_ac_here") is None


# ─── upsert_occurrences_from_chunks ──────────────────────────────────────────

def test_upsert_occurrences_walks_chunks_dedupes_and_skips_tc(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tsb/a01", "en", 3)
    _write_chunks(root, "tsb/b02", "fr", 2)
    _write_chunks(root, "tc/ac01", "en", 4)

    d = FakeDriver()
    n = upsert_occurrences_from_chunks(d, root)
    assert n == 2
    all_ids = {row["id"] for _, p in d.runs for row in p["rows"]}
    assert all_ids == {"a01", "b02"}


def test_upsert_occurrences_batches(tmp_path):
    root = tmp_path / "chunks"
    for i in range(7):
        _write_chunks(root, f"tsb/x{i:03d}", "en", 1)
    d = FakeDriver()
    n = upsert_occurrences_from_chunks(d, root, batch_size=3)
    assert n == 7
    assert len(d.runs) == 3


def test_upsert_occurrences_empty_root(tmp_path):
    d = FakeDriver()
    assert upsert_occurrences_from_chunks(d, tmp_path / "empty") == 0
    assert d.runs == []


# ─── upsert_acs_from_chunks ───────────────────────────────────────────────────

def test_upsert_acs_from_tc_corpus(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tc/AC_702-001_ISSUE-1", "en", 2)
    _write_chunks(root, "tc/AC_507-002_-_ISSUE_02", "en", 1)
    _write_chunks(root, "tsb/a01", "en", 1)  # skipped — not TC

    d = FakeDriver()
    n = upsert_acs_from_chunks(d, root)
    assert n == 2
    ac_ids = {row["id"] for _, p in d.runs for row in p["rows"]}
    assert ac_ids == {"702-001", "507-002"}


def test_upsert_acs_skips_unrecognised_tc_names(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tc/no_ac_number_here", "en", 1)
    d = FakeDriver()
    n = upsert_acs_from_chunks(d, root)
    assert n == 0


# ─── upsert_entities_from_chunks ─────────────────────────────────────────────

class RegFixtureExtractor:
    """Returns pre-set entities for specific doc_ids; NoopExtractor otherwise."""

    def __init__(self, table: dict[str, ExtractedEntities]):
        self._table = table
        self._noop = NoopExtractor()

    def extract(self, chunk: ChunkRecord) -> ExtractedEntities:
        return self._table.get(chunk.doc_id, self._noop.extract(chunk))


def test_entity_upsert_creates_finding_and_regulation(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tsb/a01", "en", 1)

    ents = ExtractedEntities({
        "findings": [{"text": "Fuel tanks empty on arrival.", "category": "cause", "lang": "en"}],
        "regulations": ["602.115"],
    })
    d = FakeDriver()
    counts = upsert_entities_from_chunks(d, root, RegFixtureExtractor({"tsb/a01": ents}))

    assert counts["findings"] == 1
    assert counts["regulations"] >= 1
    assert counts["chunks"] >= 1

    cyphers = {c for c, _ in d.runs}
    assert any("Finding" in c for c in cyphers)
    assert any("Regulation" in c for c in cyphers)


def test_entity_upsert_links_finding_to_regulation(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tsb/a01", "en", 1)

    ents = ExtractedEntities({
        "findings": [{"text": "Pilot violated CAR.", "category": "cause", "lang": "en"}],
        "regulations": ["602.88"],
    })
    d = FakeDriver()
    upsert_entities_from_chunks(d, root, RegFixtureExtractor({"tsb/a01": ents}))

    link_rows = [p["rows"] for c, p in d.runs if "CITES" in c]
    assert link_rows, "expected a CITES link run"
    link_ids = {r.get("reg_id") for rows in link_rows for r in rows}
    assert "602.88" in link_ids


def test_entity_upsert_tc_chunks_create_ac_citations(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tc/AC_702-001_ISSUE-1", "en", 1)

    ents = ExtractedEntities({"advisory_circulars": ["702-001"], "regulations": ["602.115"]})
    d = FakeDriver()
    counts = upsert_entities_from_chunks(
        d, root, RegFixtureExtractor({"tc/AC_702-001_ISSUE-1": ents}))
    # TC chunks should create AC + Regulation nodes
    assert counts["acs"] >= 1
    assert counts["regulations"] >= 1
    # Findings should not be created for TC chunks
    assert counts["findings"] == 0


def test_entity_upsert_noop_extractor_produces_zero_entities(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tsb/a01", "en", 5)
    d = FakeDriver()
    counts = upsert_entities_from_chunks(d, root, NoopExtractor())
    assert counts["findings"] == 0
    assert counts["recommendations"] == 0
    assert counts["regulations"] == 0


def test_entity_upsert_recommendation_with_tsb_id(tmp_path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tsb/a01", "en", 1)

    ents = ExtractedEntities({
        "recommendations": [{"id": "A19-01", "text": "Install TAWS.", "lang": "en"}],
    })
    d = FakeDriver()
    counts = upsert_entities_from_chunks(d, root, RegFixtureExtractor({"tsb/a01": ents}))
    assert counts["recommendations"] == 1
    # Verify rec id is the TSB id, not a hash
    rec_rows = [p["rows"] for c, p in d.runs if "Recommendation" in c and "MERGE" in c]
    assert any(r["id"] == "A19-01" for rows in rec_rows for r in rows)
