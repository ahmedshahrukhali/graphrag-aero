"""End-to-end test of the embed CLI orchestration.

Uses a stub BGE-M3 (deterministic vectors, no model load) and an in-memory
Qdrant client. Asserts:

  - point count matches chunk count
  - payload round-trips all metadata fields
  - re-running keeps the point count flat (idempotency)
  - --limit and filters narrow the input
"""
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient

from embed.bge_m3 import DENSE_DIM
from embed.ids import point_id_for
from embed.qdrant import count_points
from embed.run import main


class StubEmbedder:
    """Returns a deterministic 1024-vector per input text."""

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        # vector deterministic from text length + first char
        return [
            [float(((len(t) + (ord(t[0]) if t else 0) + i) % 7)) / 7.0
             for i in range(DENSE_DIM)]
            for t in texts
        ]


def _write_chunk_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _hash_for(doc_id: str, idx: int) -> str:
    """Real sha256 of (doc_id, idx) — varies in the high-order bytes used for
    the UUID derivation, unlike a zero-padded ``f"{idx:064x}"``."""
    return hashlib.sha256(f"{doc_id}:{idx}".encode()).hexdigest()


def _make_record(doc_id: str, idx: int, lang: str) -> dict:
    return {
        "doc_id": doc_id,
        "source_url": f"https://example.test/{doc_id}.pdf",
        "section_title": f"section-{idx}",
        "page": idx,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "chunk_hash": _hash_for(doc_id, idx),
        "lang": lang,
        "text": f"chunk {idx} of {doc_id}",
    }


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QDRANT_COLLECTION_DENSE", "test_dense")
    yield


@pytest.fixture
def chunks_dir(tmp_path: Path) -> Path:
    root = tmp_path / "chunks"
    _write_chunk_jsonl(root / "en" / "tsb" / "a.jsonl",
                       [_make_record("tsb/a", i, "en") for i in range(3)])
    _write_chunk_jsonl(root / "en" / "tc" / "ac.jsonl",
                       [_make_record("tc/ac", i + 10, "en") for i in range(2)])
    _write_chunk_jsonl(root / "fr" / "tsb" / "b.jsonl",
                       [_make_record("tsb/b", i + 20, "fr") for i in range(4)])
    return root


def test_main_upserts_all_chunks(chunks_dir: Path):
    client = QdrantClient(":memory:")
    stub = StubEmbedder()
    rc = main(
        ["--in", str(chunks_dir), "--batch-size", "2"],
        embedder_factory=lambda bs: stub,
        client=client,
    )
    assert rc == 0
    # 3 + 2 + 4 = 9
    assert count_points(client, "test_dense") == 9
    # batches were 2,2,2,2,1 = 5 calls
    assert len(stub.calls) == 5


def test_payload_roundtrips(chunks_dir: Path):
    client = QdrantClient(":memory:")
    main(
        ["--in", str(chunks_dir)],
        embedder_factory=lambda bs: StubEmbedder(),
        client=client,
    )
    pid = point_id_for(_hash_for("tsb/a", 0))
    retrieved = client.retrieve(
        collection_name="test_dense", ids=[pid], with_payload=True, with_vectors=True,
    )
    assert len(retrieved) == 1
    pt = retrieved[0]
    assert pt.payload["doc_id"] == "tsb/a"
    assert pt.payload["lang"] == "en"
    assert pt.payload["page"] == 0
    assert pt.payload["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert pt.payload["text"].startswith("chunk 0")
    assert len(pt.vector) == DENSE_DIM


def test_idempotent_rerun(chunks_dir: Path):
    client = QdrantClient(":memory:")
    factory = lambda bs: StubEmbedder()
    main(["--in", str(chunks_dir)], embedder_factory=factory, client=client)
    main(["--in", str(chunks_dir)], embedder_factory=factory, client=client)
    assert count_points(client, "test_dense") == 9


def test_recreate_drops_existing(chunks_dir: Path):
    client = QdrantClient(":memory:")
    factory = lambda bs: StubEmbedder()
    main(["--in", str(chunks_dir)], embedder_factory=factory, client=client)
    assert count_points(client, "test_dense") == 9
    # Run again with --recreate and --limit; only the limited subset should remain.
    main(["--in", str(chunks_dir), "--recreate", "--limit", "2"],
         embedder_factory=factory, client=client)
    assert count_points(client, "test_dense") == 2


def test_filter_lang(chunks_dir: Path):
    client = QdrantClient(":memory:")
    main(
        ["--in", str(chunks_dir), "--lang", "fr"],
        embedder_factory=lambda bs: StubEmbedder(),
        client=client,
    )
    assert count_points(client, "test_dense") == 4


def test_filter_source(chunks_dir: Path):
    client = QdrantClient(":memory:")
    main(
        ["--in", str(chunks_dir), "--source", "tc"],
        embedder_factory=lambda bs: StubEmbedder(),
        client=client,
    )
    assert count_points(client, "test_dense") == 2


def test_limit_caps_records(chunks_dir: Path):
    client = QdrantClient(":memory:")
    main(
        ["--in", str(chunks_dir), "--limit", "4"],
        embedder_factory=lambda bs: StubEmbedder(),
        client=client,
    )
    assert count_points(client, "test_dense") == 4


def test_filter_zh_ttsb_source(tmp_path: Path):
    """The CLI must admit --lang zh and --source ttsb/caac (S20 ZH corpus)."""
    root = tmp_path / "chunks"
    _write_chunk_jsonl(root / "zh" / "ttsb" / "9234_x.jsonl",
                       [_make_record("ttsb/9234_x", i, "zh") for i in range(3)])
    _write_chunk_jsonl(root / "zh" / "caac" / "P020.jsonl",
                       [_make_record("caac/P020", i + 30, "zh") for i in range(2)])
    client = QdrantClient(":memory:")
    main(
        ["--in", str(root), "--lang", "zh", "--source", "ttsb"],
        embedder_factory=lambda bs: StubEmbedder(),
        client=client,
    )
    assert count_points(client, "test_dense") == 3
