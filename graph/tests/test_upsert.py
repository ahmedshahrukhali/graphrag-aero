"""Tests for occurrence upsert — FakeDriver captures UNWIND batches."""
import json
from pathlib import Path

import pytest

from graph.upsert import (
    UPSERT_OCCURRENCE_CYPHER,
    _dedup_rows,
    _occurrence_row,
    upsert_occurrences_from_chunks,
)
from embed.jsonl import ChunkRecord


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


def _rec(doc_id: str, *, lang: str = "en", chunk_hash: str = "0" * 64) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        source_url=f"https://example.test/{doc_id}.pdf",
        section_title="", page=1, bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=chunk_hash, lang=lang, text="x",
    )


# ─── row helpers ─────────────────────────────────────────────────────────────

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
    deduped = _dedup_rows(rows)
    assert len(deduped) == 2
    assert {r["id"] for r in deduped} == {"a", "b"}


# ─── end-to-end with fake driver + tmp chunks dir ────────────────────────────

def _write_chunks(root: Path, doc_id: str, lang: str, n: int):
    """Write ``n`` chunk records for ``doc_id`` under root."""
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
                "chunk_hash": f"{i:064x}",
                "lang": lang,
                "text": f"chunk-{i}",
            }
            f.write(json.dumps(r) + "\n")


def test_upsert_walks_chunks_dedupes_and_skips_tc(tmp_path: Path):
    root = tmp_path / "chunks"
    _write_chunks(root, "tsb/a01", "en", 3)   # one occurrence (3 chunks)
    _write_chunks(root, "tsb/b02", "fr", 2)   # another occurrence
    _write_chunks(root, "tc/ac01", "en", 4)   # skipped (TC)

    d = FakeDriver()
    n = upsert_occurrences_from_chunks(d, root)
    assert n == 2

    # One run() call with both occurrences in the rows batch.
    assert len(d.runs) == 1
    cypher, params = d.runs[0]
    assert cypher == UPSERT_OCCURRENCE_CYPHER
    ids = {row["id"] for row in params["rows"]}
    assert ids == {"a01", "b02"}


def test_upsert_batches(tmp_path: Path):
    root = tmp_path / "chunks"
    for i in range(7):
        _write_chunks(root, f"tsb/x{i:03d}", "en", 1)

    d = FakeDriver()
    n = upsert_occurrences_from_chunks(d, root, batch_size=3)
    assert n == 7
    # ceil(7 / 3) = 3 batches
    assert len(d.runs) == 3
    assert sum(len(p["rows"]) for _, p in d.runs) == 7


def test_upsert_empty_root(tmp_path: Path):
    d = FakeDriver()
    n = upsert_occurrences_from_chunks(d, tmp_path / "empty")
    assert n == 0
    assert d.runs == []
