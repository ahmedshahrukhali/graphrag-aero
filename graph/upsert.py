"""Upsert nodes into Neo4j.

This phase only writes Occurrence nodes derived from chunk JSONL metadata.
Aircraft / Finding / Recommendation / Regulation / AC nodes need richer
extraction (LLM-based) — that ships separately.

The MERGE is idempotent on ``Occurrence.id``; re-runs are no-ops apart from
property refresh.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from embed.jsonl import ChunkRecord, iter_records

from .client import DriverLike


logger = logging.getLogger(__name__)


# Batch size for UNWIND — large enough to amortise round-trips, small enough
# that one bad row doesn't kill a long ingest.
BATCH_SIZE = 500


UPSERT_OCCURRENCE_CYPHER = """
UNWIND $rows AS row
MERGE (o:Occurrence {id: row.id})
SET   o.source_url = row.source_url,
      o.lang = row.lang
""".strip()


def _occurrence_row(rec: ChunkRecord) -> dict | None:
    """Return a row dict if ``rec`` is a TSB occurrence chunk, else None."""
    if not rec.doc_id.startswith("tsb/"):
        # TC ACs are Advisory Circulars (separate label, populated later);
        # skip them here.
        return None
    occ_id = rec.doc_id.split("/", 1)[1]
    return {"id": occ_id, "source_url": rec.source_url, "lang": rec.lang}


def _dedup_rows(rows: Iterable[dict]) -> list[dict]:
    """Multiple chunks share an occurrence — collapse to one row per id."""
    seen: dict[str, dict] = {}
    for r in rows:
        if r["id"] not in seen:
            seen[r["id"]] = r
    return list(seen.values())


def upsert_occurrences_from_chunks(
    driver: DriverLike,
    chunks_root: Path,
    *,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Walk ``chunks_root``, build Occurrence rows, UNWIND-MERGE into Neo4j.

    Returns the number of unique occurrences upserted.
    """
    rows = (_occurrence_row(r) for r in iter_records(chunks_root))
    deduped = _dedup_rows(r for r in rows if r is not None)
    if not deduped:
        return 0

    with driver.session() as session:
        for start in range(0, len(deduped), batch_size):
            batch = deduped[start : start + batch_size]
            session.run(UPSERT_OCCURRENCE_CYPHER, rows=batch)
            logger.debug("upserted %d occurrences (offset %d)", len(batch), start)

    logger.info("upserted %d Occurrence nodes", len(deduped))
    return len(deduped)
