"""Walk ``data/chunks/{lang}/{source}/*.jsonl`` → stream parsed chunk records.

P1's writer (``ingestion.processing.run``) emits one chunk per line with the
schema::

    {doc_id, source_url, section_title, page, bbox, chunk_hash, lang, text}

We re-read those files and yield dicts in a deterministic order so re-runs are
reproducible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


LANGS = ("en", "fr", "zh")
SOURCES = ("tsb", "tc", "ttsb", "caac")


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    source_url: str | None
    section_title: str
    page: int
    bbox: list[float]
    chunk_hash: str
    lang: str
    text: str
    # WS-0 frozen fields. Optional with derivation so the existing index (whose
    # payloads predate them) still hydrates — the re-ingest (WS-F) populates them.
    page_bboxes: list[list[float]] = field(default_factory=list)  # [[page,x0,top,x1,bottom], ...]
    corpus: str = ""                                              # "tsb" | "tc" | "caac"
    kind: str = "text"                                            # "text" | "figure"

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkRecord":
        doc_id = d["doc_id"]
        page = d.get("page", 0)
        bbox = list(d.get("bbox", [0.0, 0.0, 0.0, 0.0]))
        # Region grounding: prefer stored page_bboxes; else derive a single rect
        # from the legacy (page, bbox) so old payloads ground at region level too.
        page_bboxes = d.get("page_bboxes")
        if page_bboxes:
            page_bboxes = [list(pb) for pb in page_bboxes]
        elif any(v != 0.0 for v in bbox):
            page_bboxes = [[float(page), *bbox]]
        else:
            page_bboxes = []
        return cls(
            doc_id=doc_id,
            source_url=d.get("source_url"),
            section_title=d.get("section_title", ""),
            page=page,
            bbox=bbox,
            chunk_hash=d["chunk_hash"],
            lang=d["lang"],
            text=d["text"],
            page_bboxes=page_bboxes,
            # corpus: stored tag, else the doc_id prefix ("tsb/abc" → "tsb").
            corpus=d.get("corpus") or doc_id.split("/", 1)[0],
            kind=d.get("kind", "text"),
        )

    def payload(self) -> dict:
        """The Qdrant point payload — exactly the on-disk schema."""
        return {
            "doc_id": self.doc_id,
            "source_url": self.source_url,
            "section_title": self.section_title,
            "page": self.page,
            "bbox": self.bbox,
            "page_bboxes": self.page_bboxes,
            "corpus": self.corpus,
            "kind": self.kind,
            "chunk_hash": self.chunk_hash,
            "lang": self.lang,
            "text": self.text,
        }


def _resolve_filter(value: str | None, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if value in (None, "all"):
        return allowed
    if value not in allowed:
        raise ValueError(f"unknown value {value!r}; allowed: {allowed + ('all',)}")
    return (value,)


def iter_chunk_files(
    chunks_root: Path,
    *,
    source: str | None = None,
    lang: str | None = None,
) -> list[Path]:
    """All JSONL files under ``chunks_root/{lang}/{source}/*.jsonl``, sorted."""
    langs = _resolve_filter(lang, LANGS)
    sources = _resolve_filter(source, SOURCES)
    out: list[Path] = []
    for lg in langs:
        for src in sources:
            d = chunks_root / lg / src
            if not d.is_dir():
                continue
            out.extend(sorted(d.glob("*.jsonl")))
    return out


def iter_records(
    chunks_root: Path,
    *,
    source: str | None = None,
    lang: str | None = None,
    limit: int | None = None,
) -> Iterator[ChunkRecord]:
    """Stream parsed chunk records from the corpus, capped at ``limit``."""
    yielded = 0
    for path in iter_chunk_files(chunks_root, source=source, lang=lang):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = ChunkRecord.from_dict(json.loads(line))
                yield rec
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
