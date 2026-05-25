"""Entity extraction hook.

Phase 4 ships the Protocol + a no-op stub. A future task wires up an
LLM-backed extractor that pulls Aircraft / Finding / Recommendation /
Regulation / AC entities from chunk text and writes the appropriate nodes +
relationships into Neo4j.

The Protocol lives here so the agent / backend can type-check against it
without importing any heavy LLM machinery.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from embed.jsonl import ChunkRecord


class ExtractedEntities(dict):
    """Bag of extracted entities for a single chunk.

    Keys (all optional, all list-of-string):
        - aircraft: list[str]           # aircraft type/model ids
        - findings: list[str]           # short finding labels
        - recommendations: list[str]    # recommendation ids (e.g. "TSB A19-01")
        - regulations: list[str]        # regulation refs (e.g. "CAR 605.31")
        - advisory_circulars: list[str] # AC ids (e.g. "AC 700-027")
    """


class EntityExtractor(Protocol):
    """Pluggable extractor; implementations may call an LLM, regex, or both."""

    def extract(self, chunk: ChunkRecord) -> ExtractedEntities: ...


class NoopExtractor:
    """Returns no entities. Default until a real extractor lands."""

    def extract(self, chunk: ChunkRecord) -> ExtractedEntities:  # noqa: ARG002
        return ExtractedEntities()


def extract_all(extractor: EntityExtractor, chunks: Sequence[ChunkRecord]) -> list[ExtractedEntities]:
    return [extractor.extract(c) for c in chunks]
