"""Tests for the extract Protocol + NoopExtractor."""
from embed.jsonl import ChunkRecord
from graph.extract import ExtractedEntities, NoopExtractor, extract_all


def _rec(text: str = "x") -> ChunkRecord:
    return ChunkRecord(
        doc_id="tsb/a01", source_url=None, section_title="", page=1,
        bbox=[0.0, 0.0, 0.0, 0.0], chunk_hash="0" * 64, lang="en", text=text,
    )


def test_noop_returns_empty_bag():
    ent = NoopExtractor().extract(_rec())
    assert ent == ExtractedEntities()
    assert isinstance(ent, dict)


def test_extract_all_runs_one_per_chunk():
    out = extract_all(NoopExtractor(), [_rec("a"), _rec("b"), _rec("c")])
    assert len(out) == 3
    assert all(isinstance(e, ExtractedEntities) for e in out)
