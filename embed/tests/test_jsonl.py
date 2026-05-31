"""Tests for chunk JSONL iteration + filtering."""
import json
from pathlib import Path

import pytest

from embed.jsonl import ChunkRecord, iter_chunk_files, iter_records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_record(doc_id: str, idx: int, lang: str) -> dict:
    return {
        "doc_id": doc_id,
        "source_url": f"https://example.test/{doc_id}.pdf",
        "section_title": "Findings",
        "page": idx,
        "bbox": [0.0, 0.0, 100.0, 50.0],
        "chunk_hash": f"{idx:064x}",
        "lang": lang,
        "text": f"chunk {idx} of {doc_id}",
    }


@pytest.fixture
def chunks_root(tmp_path: Path) -> Path:
    root = tmp_path / "chunks"
    _write_jsonl(root / "en" / "tsb" / "a01.jsonl",
                 [_make_record("tsb/a01", i, "en") for i in range(3)])
    _write_jsonl(root / "en" / "tc" / "ac01.jsonl",
                 [_make_record("tc/ac01", i, "en") for i in range(2)])
    _write_jsonl(root / "fr" / "tsb" / "a02.jsonl",
                 [_make_record("tsb/a02", i, "fr") for i in range(4)])
    return root


def test_iter_files_all(chunks_root: Path):
    files = iter_chunk_files(chunks_root)
    assert len(files) == 3
    assert [p.name for p in files] == ["a01.jsonl", "ac01.jsonl", "a02.jsonl"]


def test_iter_files_filter_source(chunks_root: Path):
    files = iter_chunk_files(chunks_root, source="tsb")
    assert [p.parent.name for p in files] == ["tsb", "tsb"]


def test_iter_files_filter_lang(chunks_root: Path):
    files = iter_chunk_files(chunks_root, lang="fr")
    assert len(files) == 1
    assert files[0].parent.parent.name == "fr"


def test_iter_records_all(chunks_root: Path):
    recs = list(iter_records(chunks_root))
    # 3 + 2 + 4
    assert len(recs) == 9
    assert isinstance(recs[0], ChunkRecord)


def test_iter_records_limit(chunks_root: Path):
    recs = list(iter_records(chunks_root, limit=4))
    assert len(recs) == 4


def test_iter_records_filter_combo(chunks_root: Path):
    recs = list(iter_records(chunks_root, source="tsb", lang="en"))
    assert len(recs) == 3
    assert all(r.lang == "en" for r in recs)
    assert all(r.doc_id == "tsb/a01" for r in recs)


def test_record_payload_roundtrip():
    src = _make_record("tsb/a99", 7, "en")
    rec = ChunkRecord.from_dict(src)
    payload = rec.payload()
    for k in ("doc_id", "source_url", "section_title", "page", "chunk_hash", "lang", "text"):
        assert payload[k] == src[k]
    assert payload["bbox"] == src["bbox"]


def test_ws0_fields_roundtrip():
    """page_bboxes / corpus / kind survive from_dict → payload unchanged."""
    src = _make_record("caac/ac21-44", 3, "zh")
    src["page_bboxes"] = [[3.0, 10.0, 20.0, 110.0, 70.0], [4.0, 5.0, 5.0, 200.0, 90.0]]
    src["corpus"] = "caac"
    src["kind"] = "figure"
    payload = ChunkRecord.from_dict(src).payload()
    assert payload["page_bboxes"] == src["page_bboxes"]
    assert payload["corpus"] == "caac"
    assert payload["kind"] == "figure"


def test_ws0_backward_compat_derives_from_legacy_payload():
    """A pre-re-ingest payload (no page_bboxes/corpus/kind) still hydrates:
    page_bboxes is derived from (page, bbox), corpus from the doc_id prefix,
    kind defaults to text."""
    src = _make_record("tsb/a01", 5, "en")  # no WS-0 keys
    rec = ChunkRecord.from_dict(src)
    assert rec.corpus == "tsb"
    assert rec.kind == "text"
    # Single derived rect: [page, *bbox].
    assert rec.page_bboxes == [[5.0, 0.0, 0.0, 100.0, 50.0]]


def test_ws0_backward_compat_empty_bbox_yields_no_region():
    """A legacy payload with a degenerate zero bbox derives no region rect."""
    src = _make_record("tsb/a01", 5, "en")
    src["bbox"] = [0.0, 0.0, 0.0, 0.0]
    rec = ChunkRecord.from_dict(src)
    assert rec.page_bboxes == []


def test_unknown_source_raises(chunks_root: Path):
    with pytest.raises(ValueError):
        iter_chunk_files(chunks_root, source="nope")


def test_missing_subdir_ok(tmp_path: Path):
    # Empty root → no files, no error.
    assert iter_chunk_files(tmp_path / "empty") == []
    assert list(iter_records(tmp_path / "empty")) == []
