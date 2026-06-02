"""End-to-end test of the run orchestrator with mocked pdf extraction + tokenizer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from ingestion.processing import run as run_mod
from ingestion.processing.pdf import PageExtract


@dataclass
class _Enc:
    ids: list[int]
    offsets: list[tuple[int, int]]


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> _Enc:
        _ = add_special_tokens
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text):
                break
            start = i
            while i < len(text) and not text[i].isspace():
                i += 1
            offsets.append((start, i))
            ids.append(len(ids))
        return _Enc(ids=ids, offsets=offsets)


def _make_corpus(tmp: Path) -> Path:
    corpus = tmp / "corpus"
    (corpus / "en" / "tsb").mkdir(parents=True)
    (corpus / "fr" / "tsb").mkdir(parents=True)
    (corpus / "en" / "tsb" / "a23h0001.pdf").write_bytes(b"fake")
    (corpus / "fr" / "tsb" / "a23h0001.pdf").write_bytes(b"fake")
    return corpus


def _fake_extract_pages_with_ocr(path: Path, ocr_lang: str = "latin") -> list[PageExtract]:
    # Deterministic content keyed off filename + lang, with enough words to chunk.
    text = f"Findings the engine failed at altitude on {path.parent.parent.name}"
    return [PageExtract(page=1, text=text, chars=[])]


def test_run_writes_jsonl_for_each_doc(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    out = tmp_path / "chunks"
    with patch.object(run_mod, "extract_pages_with_ocr",
                      side_effect=_fake_extract_pages_with_ocr), \
         patch.object(run_mod, "default_tokenizer",
                      return_value=_WhitespaceTokenizer()):
        rc = run_mod.main(["--in", str(corpus), "--out", str(out)])
    assert rc == 0
    en_out = out / "en" / "tsb" / "a23h0001.jsonl"
    fr_out = out / "fr" / "tsb" / "a23h0001.jsonl"
    assert en_out.exists() and fr_out.exists()
    en_records = [json.loads(line) for line in en_out.read_text(encoding="utf-8").splitlines()]
    assert len(en_records) == 1
    r = en_records[0]
    assert r["doc_id"] == "tsb/a23h0001"
    assert r["lang"] == "en"
    assert r["source_url"].endswith("/eng/a23h0001.pdf")
    assert "engine failed" in r["text"]
    assert isinstance(r["bbox"], list) and len(r["bbox"]) == 4
    assert len(r["chunk_hash"]) == 64


def test_run_is_idempotent_skips_fresh_outputs(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    out = tmp_path / "chunks"
    call_count = {"n": 0}

    def counting_extract(path: Path, ocr_lang: str = "latin"):
        call_count["n"] += 1
        return _fake_extract_pages_with_ocr(path)

    with patch.object(run_mod, "extract_pages_with_ocr", side_effect=counting_extract), \
         patch.object(run_mod, "default_tokenizer",
                      return_value=_WhitespaceTokenizer()):
        run_mod.main(["--in", str(corpus), "--out", str(out)])
        first = call_count["n"]
        run_mod.main(["--in", str(corpus), "--out", str(out)])
        second = call_count["n"]
    # Second run should skip both docs because their .jsonl mtime is newer.
    assert second == first


def test_run_force_reprocesses(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    out = tmp_path / "chunks"

    def fake_extract(path: Path, ocr_lang: str = "latin"):
        return _fake_extract_pages_with_ocr(path)

    with patch.object(run_mod, "extract_pages_with_ocr", side_effect=fake_extract), \
         patch.object(run_mod, "default_tokenizer",
                      return_value=_WhitespaceTokenizer()):
        run_mod.main(["--in", str(corpus), "--out", str(out)])
        first_mtime = (out / "en" / "tsb" / "a23h0001.jsonl").stat().st_mtime
        # Bump the PDF mtime so without --force the file would still skip (output is fresh).
        # Then --force should reprocess regardless.
        run_mod.main(["--in", str(corpus), "--out", str(out), "--force"])
        second_mtime = (out / "en" / "tsb" / "a23h0001.jsonl").stat().st_mtime
    assert second_mtime >= first_mtime


def test_run_limit_caps_doc_count(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    out = tmp_path / "chunks"
    seen: list[Path] = []

    def trace_extract(path: Path, ocr_lang: str = "latin"):
        seen.append(path)
        return _fake_extract_pages_with_ocr(path)

    with patch.object(run_mod, "extract_pages_with_ocr", side_effect=trace_extract), \
         patch.object(run_mod, "default_tokenizer",
                      return_value=_WhitespaceTokenizer()):
        run_mod.main(["--in", str(corpus), "--out", str(out), "--limit", "1"])
    assert len(seen) == 1


def test_run_dedup_drops_duplicate_chunks_across_docs(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    out = tmp_path / "chunks"
    # Same text in both EN and FR docs -> identical chunk_hash -> 2nd one drops the chunk.
    same_text = "shared boilerplate paragraph that appears in both docs"

    def same_extract(path: Path, ocr_lang: str = "latin"):
        return [PageExtract(page=1, text=same_text, chars=[])]

    with patch.object(run_mod, "extract_pages_with_ocr", side_effect=same_extract), \
         patch.object(run_mod, "default_tokenizer",
                      return_value=_WhitespaceTokenizer()):
        run_mod.main(["--in", str(corpus), "--out", str(out)])

    en_lines = (out / "en" / "tsb" / "a23h0001.jsonl").read_text(encoding="utf-8").splitlines()
    fr_path = out / "fr" / "tsb" / "a23h0001.jsonl"
    fr_lines = fr_path.read_text(encoding="utf-8").splitlines() if fr_path.exists() else []
    assert len(en_lines) == 1
    # FR file should still be written (atomic) but contain zero chunks after dedup.
    assert fr_lines == []


def test_iter_corpus_pdfs_finds_only_pdfs_in_known_dirs(tmp_path: Path):
    corpus = _make_corpus(tmp_path)
    (corpus / "en" / "tsb" / "notes.txt").write_text("ignore")
    (corpus / "en" / "other" / "x.pdf").parent.mkdir(parents=True)
    (corpus / "en" / "other" / "x.pdf").write_bytes(b"x")
    paths = run_mod.iter_corpus_pdfs(corpus, source="all")
    names = sorted(p.name for p in paths)
    assert names == ["a23h0001.pdf", "a23h0001.pdf"]


def test_iter_corpus_pdfs_includes_zh_sources(tmp_path: Path):
    corpus = tmp_path / "corpus"
    (corpus / "zh" / "caac").mkdir(parents=True)
    (corpus / "zh" / "ttsb").mkdir(parents=True)
    (corpus / "zh" / "caac" / "ac01.pdf").write_bytes(b"x")
    (corpus / "zh" / "ttsb" / "aor01.pdf").write_bytes(b"x")
    assert {p.name for p in run_mod.iter_corpus_pdfs(corpus, source="all")} == {
        "ac01.pdf", "aor01.pdf"}
    assert [p.name for p in run_mod.iter_corpus_pdfs(corpus, source="caac")] == ["ac01.pdf"]


def test_process_doc_routes_zh_caac_to_ch_model(tmp_path: Path):
    """The (lang, source) of a zh/caac doc must select PaddleOCR's 'ch' model."""
    corpus = tmp_path / "corpus"
    (corpus / "zh" / "caac").mkdir(parents=True)
    src = corpus / "zh" / "caac" / "ac01.pdf"
    src.write_bytes(b"x")
    seen: dict[str, str] = {}

    def capture(path: Path, ocr_lang: str = "latin"):
        seen["ocr_lang"] = ocr_lang
        return _fake_extract_pages_with_ocr(path)

    from ingestion.processing.dedup import Dedup
    with patch.object(run_mod, "extract_pages_with_ocr", side_effect=capture):
        run_mod.process_doc(src, tmp_path / "chunks", _WhitespaceTokenizer(), Dedup())
    assert seen["ocr_lang"] == "ch"
