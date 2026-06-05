"""Offline tests for the eval runner.

The metrics math is covered by ``test_metrics.py``. Here we test the runner
orchestration: dataset loading, aggregation, per-lang breakdown, and that
``evaluate`` correctly drives a query runner over the dataset. An end-to-end
test wires the runner against the real retrieve+rerank pipeline using stub
models against an in-memory Qdrant — no network, no weight downloads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient

from embed.jsonl import ChunkRecord
from embed.qdrant import DENSE_DIM, ensure_collection, upsert_batch
from eval.run import EvalItem, ItemResult, evaluate, load_dataset, main


COLL = "test_eval"


def test_load_dataset_skips_blank_and_comment_lines(tmp_path: Path):
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '\n'
        '# a comment\n'
        '{"id": "q1", "query": "hello", "expected": ["d1"], "lang": "en"}\n'
        '\n'
        '{"id": "q2", "query": "bonjour", "expected": ["d2"], "lang": "fr"}\n',
        encoding="utf-8",
    )
    items = load_dataset(p)
    assert [i.id for i in items] == ["q1", "q2"]
    assert items[0].lang == "en"
    assert items[1].expected == ["d2"]


def test_load_dataset_rejects_empty(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("# nothing\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(p)


def test_evaluate_aggregates_and_splits_by_lang():
    dataset = [
        EvalItem(id="q1", query="a", expected=["doc1"], lang="en"),
        EvalItem(id="q2", query="b", expected=["doc2"], lang="en"),
        EvalItem(id="q3", query="c", expected=["doc3"], lang="fr"),
    ]
    # q1: hit at rank 1; q2: miss; q3: hit at rank 2.
    canned = {
        "a": ["doc1", "x", "y"],
        "b": ["x", "y", "z"],
        "c": ["x", "doc3", "y"],
    }
    report = evaluate(lambda q, _lang: canned[q], dataset)

    assert report["overall"]["n"] == 3
    # Recall@5 hits on q1 and q3 only → 2/3.
    assert report["overall"]["recall_at_5"] == pytest.approx(2 / 3)
    # MRR = (1 + 0 + 1/2) / 3.
    assert report["overall"]["mrr"] == pytest.approx((1 + 0 + 0.5) / 3)

    assert set(report["by_lang"]) == {"en", "fr"}
    assert report["by_lang"]["en"]["n"] == 2
    assert report["by_lang"]["en"]["recall_at_5"] == pytest.approx(0.5)
    assert report["by_lang"]["fr"]["n"] == 1
    assert report["by_lang"]["fr"]["mrr"] == pytest.approx(0.5)

    assert [r["id"] for r in report["items"]] == ["q1", "q2", "q3"]


def test_evaluate_passes_lang_to_runner():
    seen: list[str | None] = []

    def runner(q: str, lang: str | None) -> list[str]:
        seen.append(lang)
        return ["doc1"]

    dataset = [
        EvalItem(id="q1", query="x", expected=["doc1"], lang="en"),
        EvalItem(id="q2", query="y", expected=["doc1"], lang="fr"),
        EvalItem(id="q3", query="z", expected=["doc1"], lang=None),
    ]
    evaluate(runner, dataset)
    assert seen == ["en", "fr", None]


# --- end-to-end: real retrieve pipeline + stubs + in-memory Qdrant ----------

def _unit_vec(direction: int) -> list[float]:
    v = [0.0] * DENSE_DIM
    v[direction] = 1.0
    return v


def _record(text: str, *, doc_id: str, lang: str = "en", idx: int = 0) -> ChunkRecord:
    h = hashlib.sha256(f"{doc_id}:{idx}:{text}".encode()).hexdigest()
    return ChunkRecord(
        doc_id=doc_id,
        source_url=None,
        section_title="",
        page=idx + 1,
        bbox=[0.0, 0.0, 0.0, 0.0],
        chunk_hash=h,
        lang=lang,
        text=text,
    )


class _StubEmbedder:
    """Query embedding is keyed by query text — lets us steer ANN per item."""
    def __init__(self, axis_for_query: dict[str, int]):
        self._axes = axis_for_query

    def embed(self, texts):
        return [_unit_vec(self._axes[t]) for t in texts]


class _StubReranker:
    """Identity reranker: returns 1.0 for every passage so ANN order survives."""
    def score(self, query, passages):
        return [1.0 for _ in passages]


def test_main_drives_real_pipeline_with_stubs(tmp_path: Path, capsys):
    from retrieve.pipeline import retrieve_and_rerank

    client = QdrantClient(":memory:")
    ensure_collection(client, COLL)

    # 3 corpus docs on distinct vector axes; query "alpha" steers to doc-A's
    # axis, "bravo" to doc-B's, "charlie" to doc-C's.
    records = [
        _record("alpha text", doc_id="tsb/doc-A", lang="en", idx=0),
        _record("bravo text", doc_id="tsb/doc-B", lang="en", idx=1),
        _record("charlie text", doc_id="tsb/doc-C", lang="en", idx=2),
    ]
    upsert_batch(client, COLL, records, [_unit_vec(0), _unit_vec(1), _unit_vec(2)])

    embedder = _StubEmbedder({"alpha?": 0, "bravo?": 1})
    reranker = _StubReranker()

    def runner(query: str, lang: str | None) -> list[str]:
        results = retrieve_and_rerank(
            query, embedder=embedder, reranker=reranker,
            client=client, collection=COLL, ann_k=3, top_k=3, lang=lang,
        )
        return [r.record.doc_id for r in results]

    ds = tmp_path / "ds.jsonl"
    ds.write_text(
        '{"id": "q1", "query": "alpha?", "expected": ["tsb/doc-A"], "lang": "en"}\n'
        '{"id": "q2", "query": "bravo?", "expected": ["tsb/doc-B"], "lang": "en"}\n',
        encoding="utf-8",
    )

    rc = main(["--dataset", str(ds), "--json"], query_runner=runner)
    assert rc == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["overall"]["n"] == 2
    assert report["overall"]["recall_at_5"] == pytest.approx(1.0)
    assert report["overall"]["mrr"] == pytest.approx(1.0)


def test_evaluate_mode_field_in_report():
    dataset = [EvalItem(id="q1", query="x", expected=["d1"], lang="en")]
    report = evaluate(lambda q, _: ["d1"], dataset, mode="hybrid")
    assert report["mode"] == "hybrid"


def test_evaluate_default_mode_is_dense():
    dataset = [EvalItem(id="q1", query="x", expected=["d1"], lang="en")]
    report = evaluate(lambda q, _: ["d1"], dataset)
    assert report["mode"] == "dense"


def test_main_mode_flag_accepted(tmp_path: Path, capsys):
    ds = tmp_path / "ds.jsonl"
    ds.write_text('{"id": "q1", "query": "x", "expected": ["d1"], "lang": "en"}\n',
                  encoding="utf-8")
    # Inject a stub runner so no real Qdrant is needed.
    rc = main(["--dataset", str(ds), "--mode", "hybrid", "--json"],
              query_runner=lambda q, l: ["d1"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "hybrid"


def test_load_dataset_ignores_unknown_fields(tmp_path: Path):
    """Dataset items may carry extra fields (e.g. 'tags') — load must not fail."""
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '{"id": "q1", "query": "x", "expected": ["d1"], "lang": "en", "tags": ["jargon-id"]}\n',
        encoding="utf-8",
    )
    items = load_dataset(p)
    assert items[0].id == "q1"
