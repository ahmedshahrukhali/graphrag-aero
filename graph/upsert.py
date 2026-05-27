"""Upsert nodes into Neo4j.

Three public entry points:
  upsert_occurrences_from_chunks — MERGE Occurrence nodes from TSB chunk metadata.
  upsert_acs_from_chunks         — MERGE AC nodes from TC chunk metadata.
  upsert_entities_from_chunks    — run an EntityExtractor over all chunks, then
                                   MERGE Finding/Recommendation/Regulation/AC
                                   nodes and relationships.

All MERGEs are idempotent; re-runs refresh properties without duplicating.

Node identity
-------------
  Occurrence      id = TSB occurrence id, e.g. "a13q0098"  (lang-agnostic)
  AC              id = AC number, e.g. "702-001"            (lang-agnostic)
  Regulation      id = CAR number, e.g. "602.115"           (lang-agnostic)
  Finding         id = "{occ_id}:f:{sha256[:12]}"           (report-local)
  Recommendation  id = TSB rec id (e.g. "A19-01") when present,
                      else "{occ_id}:r:{sha256[:12]}"

Provenance
----------
Every Finding and Recommendation node carries source_doc_id + page so the
agent can cite them inline — the graph facts stay grounded.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterable

from embed.jsonl import ChunkRecord, iter_records

from .client import DriverLike
from .extract import EntityExtractor, ExtractedEntities


logger = logging.getLogger(__name__)

BATCH_SIZE = 500

# Extract AC number from TC doc_id like "tc/AC_702-001_ISSUE-1"
_TC_AC_ID = re.compile(r"AC[_\s]?(\d{3}[-_]\d{3})", re.I)


# ─── Occurrence ──────────────────────────────────────────────────────────────

_UPSERT_OCCURRENCE = """
UNWIND $rows AS row
MERGE (o:Occurrence {id: row.id})
SET   o.source_url = row.source_url,
      o.lang = row.lang
""".strip()


def _occurrence_row(rec: ChunkRecord) -> dict | None:
    if not rec.doc_id.startswith("tsb/"):
        return None
    occ_id = rec.doc_id.split("/", 1)[1]
    return {"id": occ_id, "source_url": rec.source_url, "lang": rec.lang}


def _dedup_rows(rows: Iterable[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in rows:
        if r["id"] not in seen:
            seen[r["id"]] = r
    return list(seen.values())


def upsert_occurrences_from_chunks(
    driver: DriverLike, chunks_root: Path, *, batch_size: int = BATCH_SIZE,
) -> int:
    rows = (_occurrence_row(r) for r in iter_records(chunks_root))
    deduped = _dedup_rows(r for r in rows if r is not None)
    if not deduped:
        return 0
    with driver.session() as session:
        for start in range(0, len(deduped), batch_size):
            batch = deduped[start: start + batch_size]
            session.run(_UPSERT_OCCURRENCE, rows=batch)
            logger.debug("upserted %d occurrences (offset %d)", len(batch), start)
    logger.info("upserted %d Occurrence nodes", len(deduped))
    return len(deduped)


# ─── AC (from TC corpus) ─────────────────────────────────────────────────────

_UPSERT_AC = """
UNWIND $rows AS row
MERGE (a:AC {id: row.id})
SET   a.source_url = row.source_url,
      a.title = row.title
""".strip()


def _ac_id_from_doc_id(doc_id: str) -> str | None:
    """Extract canonical AC number from a tc/* doc_id."""
    stem = doc_id.split("/", 1)[-1]
    m = _TC_AC_ID.search(stem)
    if not m:
        return None
    return m.group(1).replace("_", "-").upper()


def _ac_row(rec: ChunkRecord) -> dict | None:
    if not rec.doc_id.startswith("tc/"):
        return None
    ac_id = _ac_id_from_doc_id(rec.doc_id)
    if not ac_id:
        return None
    return {"id": ac_id, "source_url": rec.source_url or "",
            "title": rec.section_title or ""}


def upsert_acs_from_chunks(
    driver: DriverLike, chunks_root: Path, *, batch_size: int = BATCH_SIZE,
) -> int:
    rows_gen = (_ac_row(r) for r in iter_records(chunks_root))
    deduped = _dedup_rows(r for r in rows_gen if r is not None)
    if not deduped:
        return 0
    with driver.session() as session:
        for start in range(0, len(deduped), batch_size):
            session.run(_UPSERT_AC, rows=deduped[start: start + batch_size])
    logger.info("upserted %d AC nodes from TC corpus", len(deduped))
    return len(deduped)


# ─── Entities (Finding / Recommendation / Regulation / AC + edges) ───────────

_UPSERT_FINDING = """
UNWIND $rows AS row
MERGE (o:Occurrence {id: row.occ_id})
MERGE (f:Finding {id: row.id})
SET   f.text           = row.text,
      f.category       = row.category,
      f.lang           = row.lang,
      f.source_doc_id  = row.source_doc_id,
      f.page           = row.page
MERGE (o)-[:HAS_FINDING]->(f)
""".strip()

_UPSERT_RECOMMENDATION = """
UNWIND $rows AS row
MERGE (o:Occurrence {id: row.occ_id})
MERGE (r:Recommendation {id: row.id})
SET   r.text           = row.text,
      r.lang           = row.lang,
      r.source_doc_id  = row.source_doc_id,
      r.page           = row.page
MERGE (o)-[:HAS_RECOMMENDATION]->(r)
""".strip()

_UPSERT_REGULATION = """
UNWIND $rows AS row
MERGE (reg:Regulation {id: row.id})
""".strip()

_LINK_FINDING_REGULATION = """
UNWIND $rows AS row
MATCH (f:Finding {id: row.finding_id})
MATCH (reg:Regulation {id: row.reg_id})
MERGE (f)-[:CITES]->(reg)
""".strip()

_LINK_OCC_REGULATION = """
UNWIND $rows AS row
MATCH (o:Occurrence {id: row.occ_id})
MATCH (reg:Regulation {id: row.reg_id})
MERGE (o)-[:CITES]->(reg)
""".strip()

_UPSERT_AC_NODE = """
UNWIND $rows AS row
MERGE (a:AC {id: row.id})
""".strip()

_LINK_OCC_AC = """
UNWIND $rows AS row
MATCH (o:Occurrence {id: row.occ_id})
MATCH (a:AC {id: row.ac_id})
MERGE (o)-[:REFERENCES_AC]->(a)
""".strip()

_LINK_REG_AC = """
UNWIND $rows AS row
MATCH (reg:Regulation {id: row.reg_id})
MATCH (a:AC {id: row.ac_id})
MERGE (reg)-[:GUIDED_BY]->(a)
""".strip()


def _finding_id(occ_id: str, text: str) -> str:
    h = hashlib.sha256(f"{occ_id}:{text}".encode()).hexdigest()[:12]
    return f"{occ_id}:f:{h}"


def _rec_id(occ_id: str, tsb_id: str | None, text: str) -> str:
    if tsb_id:
        return tsb_id
    h = hashlib.sha256(f"{occ_id}:{text}".encode()).hexdigest()[:12]
    return f"{occ_id}:r:{h}"


def upsert_entities_from_chunks(
    driver: DriverLike,
    chunks_root: Path,
    extractor: EntityExtractor,
    *,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """Run ``extractor`` over all chunks; MERGE extracted entities into Neo4j.

    Returns a count dict: {findings, recommendations, regulations, acs, chunks}.
    """
    finding_rows: list[dict] = []
    rec_rows: list[dict] = []
    reg_ids: set[str] = set()
    ac_ids: set[str] = set()
    finding_reg_links: list[dict] = []
    occ_reg_links: list[dict] = []
    occ_ac_links: list[dict] = []
    chunks_processed = 0

    for rec in iter_records(chunks_root):
        chunks_processed += 1
        ents: ExtractedEntities = extractor.extract(rec)

        # Determine the occurrence / AC context
        if rec.doc_id.startswith("tsb/"):
            occ_id = rec.doc_id.split("/", 1)[1]
            is_tsb = True
        elif rec.doc_id.startswith("tc/"):
            is_tsb = False
            ac_id = _ac_id_from_doc_id(rec.doc_id)
        else:
            continue

        # Regulations — accumulate canonical ids
        for car in ents.get("regulations") or []:
            reg_ids.add(car)
            if is_tsb:
                occ_reg_links.append({"occ_id": occ_id, "reg_id": car})

        # Advisory Circulars
        for ac in ents.get("advisory_circulars") or []:
            ac_ids.add(ac)
            if is_tsb:
                occ_ac_links.append({"occ_id": occ_id, "ac_id": ac})

        # Findings (TSB only — TC ACs don't have safety findings)
        if is_tsb:
            for f in ents.get("findings") or []:
                fid = _finding_id(occ_id, f["text"])
                finding_rows.append({
                    "id": fid, "occ_id": occ_id,
                    "text": f["text"], "category": f.get("category", "cause"),
                    "lang": f.get("lang", rec.lang),
                    "source_doc_id": rec.doc_id, "page": rec.page,
                })
                # Link finding to any regs cited in the same chunk
                for car in ents.get("regulations") or []:
                    finding_reg_links.append({"finding_id": fid, "reg_id": car})

            # Recommendations
            for r in ents.get("recommendations") or []:
                if not r.get("text") and not r.get("id"):
                    continue
                rid = _rec_id(occ_id, r.get("id"), r.get("text", ""))
                rec_rows.append({
                    "id": rid, "occ_id": occ_id,
                    "text": r.get("text", ""), "lang": r.get("lang", rec.lang),
                    "source_doc_id": rec.doc_id, "page": rec.page,
                })

    # Flush to Neo4j in batches
    with driver.session() as session:
        _batch_run(session, _UPSERT_REGULATION,
                   [{"id": r} for r in reg_ids], batch_size)
        _batch_run(session, _UPSERT_AC_NODE,
                   [{"id": a} for a in ac_ids], batch_size)
        _batch_run(session, _UPSERT_FINDING, finding_rows, batch_size)
        _batch_run(session, _UPSERT_RECOMMENDATION, rec_rows, batch_size)
        _batch_run(session, _LINK_FINDING_REGULATION, finding_reg_links, batch_size)
        _batch_run(session, _LINK_OCC_REGULATION,
                   _dedup_link_rows(occ_reg_links), batch_size)
        _batch_run(session, _LINK_OCC_AC,
                   _dedup_link_rows(occ_ac_links), batch_size)

    counts = {
        "chunks": chunks_processed,
        "regulations": len(reg_ids),
        "acs": len(ac_ids),
        "findings": len(finding_rows),
        "recommendations": len(rec_rows),
    }
    logger.info("upsert_entities_from_chunks: %s", counts)
    return counts


def _batch_run(session, cypher: str, rows: list[dict], batch_size: int) -> None:
    if not rows:
        return
    for start in range(0, len(rows), batch_size):
        session.run(cypher, rows=rows[start: start + batch_size])


def _dedup_link_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
