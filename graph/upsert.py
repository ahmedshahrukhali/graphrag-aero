"""Upsert nodes into Neo4j.

Three public entry points:
  upsert_occurrences_from_chunks — MERGE Occurrence nodes from TSB chunk metadata.
  upsert_acs_from_chunks         — MERGE AC nodes from TC chunk metadata.
  upsert_entities_from_chunks    — run an EntityExtractor over all chunks, then
                                   MERGE Finding/Recommendation/Regulation/AC
                                   nodes and relationships.

All MERGEs are idempotent; re-runs refresh properties without duplicating.

Node identity (§4 — Document-rooted)
-------------------------------------
  Occurrence      id = TSB occurrence id, e.g. "a13q0098"  (lang-agnostic)
                  doc_id = "tsb/a13q0098"   (full corpus path; carries :Document label)
  AC              id = AC number, e.g. "702-001"            (lang-agnostic)
                  doc_id = "tc/AC_702-001_ISSUE-1"          (carries :Document label)
  Regulation      id = CAR number, e.g. "602.115"           (lang-agnostic)
  Finding         id = "{occ_id}:f:{sha256[:12]}"           (report-local)
  Recommendation  id = TSB rec id (e.g. "A19-01") when present,
                      else "{occ_id}:r:{sha256[:12]}"

New §4 relationships
---------------------
  Recommendation -[:IMPLEMENTS]-> Regulation   (rec cites reg in same chunk)
  Regulation     -[:GUIDED_BY]->  AC           (TC AC doc elaborates a CAR)

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
SET   o:Document,
      o.doc_id     = row.doc_id,
      o.source_url = row.source_url,
      o.lang       = row.lang
""".strip()


def _occurrence_row(rec: ChunkRecord) -> dict | None:
    if not rec.doc_id.startswith("tsb/"):
        return None
    occ_id = rec.doc_id.split("/", 1)[1]
    return {"id": occ_id, "doc_id": rec.doc_id,
            "source_url": rec.source_url, "lang": rec.lang}


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
SET   a:Document,
      a.doc_id     = row.doc_id,
      a.source_url = row.source_url,
      a.title      = row.title
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
    return {"id": ac_id, "doc_id": rec.doc_id,
            "source_url": rec.source_url or "", "title": rec.section_title or ""}


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

# §4: Recommendation implements a Regulation cited in the same chunk
_LINK_REC_REGULATION = """
UNWIND $rows AS row
MATCH (r:Recommendation {id: row.rec_id})
MATCH (reg:Regulation {id: row.reg_id})
MERGE (r)-[:IMPLEMENTS]->(reg)
""".strip()

_UPSERT_AIRCRAFT = """
UNWIND $rows AS row
MERGE (a:Aircraft {id: row.id})
SET a.type_family = row.type_family
""".strip()

_LINK_OCC_AIRCRAFT = """
UNWIND $rows AS row
MATCH (o:Occurrence {id: row.occ_id})
MATCH (a:Aircraft {id: row.aircraft_id})
MERGE (o)-[:INVOLVES]->(a)
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
    aircraft_ids: set[str] = set()
    finding_reg_links: list[dict] = []
    rec_reg_links: list[dict] = []      # §4 Recommendation-[:IMPLEMENTS]->Regulation
    reg_ac_links: list[dict] = []       # §4 Regulation-[:GUIDED_BY]->AC (from TC corpus)
    occ_reg_links: list[dict] = []
    occ_ac_links: list[dict] = []
    occ_aircraft_links: list[dict] = []
    chunks_processed = 0

    for rec in iter_records(chunks_root):
        chunks_processed += 1
        ents: ExtractedEntities = extractor.extract(rec)

        # Determine corpus context — dispatch seam for future doc types
        occ_id: str | None = None
        doc_ac_id: str | None = None
        if rec.doc_id.startswith("tsb/"):
            occ_id = rec.doc_id.split("/", 1)[1]
            is_tsb = True
        elif rec.doc_id.startswith("tc/"):
            is_tsb = False
            doc_ac_id = _ac_id_from_doc_id(rec.doc_id)
        else:
            continue  # unknown corpus — skip until its extractor is registered

        # Regulations
        for car in ents.get("regulations") or []:
            reg_ids.add(car)
            if is_tsb and occ_id:
                occ_reg_links.append({"occ_id": occ_id, "reg_id": car})
            elif not is_tsb and doc_ac_id:
                # §4: TC AC document elaborates this regulation → GUIDED_BY edge
                reg_ac_links.append({"reg_id": car, "ac_id": doc_ac_id})

        # Advisory Circulars cited in text
        for ac in ents.get("advisory_circulars") or []:
            ac_ids.add(ac)
            if is_tsb and occ_id:
                occ_ac_links.append({"occ_id": occ_id, "ac_id": ac})

        # Aircraft
        for ac_type in ents.get("aircraft") or []:
            aircraft_ids.add(ac_type)
            if is_tsb and occ_id:
                occ_aircraft_links.append({"occ_id": occ_id, "aircraft_id": ac_type})

        # Findings (TSB only)
        if is_tsb and occ_id:
            chunk_regs = ents.get("regulations") or []

            for f in ents.get("findings") or []:
                fid = _finding_id(occ_id, f["text"])
                finding_rows.append({
                    "id": fid, "occ_id": occ_id,
                    "text": f["text"], "category": f.get("category", "cause"),
                    "lang": f.get("lang", rec.lang),
                    "source_doc_id": rec.doc_id, "page": rec.page,
                })
                for car in chunk_regs:
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
                # §4: link each rec to regulations cited in the same chunk
                for car in chunk_regs:
                    rec_reg_links.append({"rec_id": rid, "reg_id": car})

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
        _batch_run(session, _UPSERT_AIRCRAFT,
                   [{"id": a, "type_family": a.split("-")[0]} for a in aircraft_ids],
                   batch_size)
        _batch_run(session, _LINK_OCC_AIRCRAFT,
                   _dedup_link_rows(occ_aircraft_links), batch_size)
        # §4 new edges
        _batch_run(session, _LINK_REC_REGULATION,
                   _dedup_link_rows(rec_reg_links), batch_size)
        _batch_run(session, _LINK_REG_AC,
                   _dedup_link_rows(reg_ac_links), batch_size)

    counts = {
        "chunks": chunks_processed,
        "regulations": len(reg_ids),
        "acs": len(ac_ids),
        "findings": len(finding_rows),
        "recommendations": len(rec_rows),
        "aircraft": len(aircraft_ids),
        "rec_reg_links": len(rec_reg_links),
        "reg_ac_links": len(reg_ac_links),
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


# ─── Figure nodes (WS-C) ─────────────────────────────────────────────────────

_UPSERT_FIGURE = """
UNWIND $rows AS row
MERGE (fig:Figure {id: row.id})
SET   fig.doc_id   = row.doc_id,
      fig.page     = row.page,
      fig.bbox     = row.bbox,
      fig.caption  = row.caption,
      fig.ocr_text = row.ocr_text
""".strip()

_LINK_OCC_FIGURE = """
UNWIND $rows AS row
MATCH (o:Occurrence {id: row.occ_id})
MATCH (fig:Figure {id: row.fig_id})
MERGE (o)-[:HAS_FIGURE]->(fig)
""".strip()


def upsert_figures(
    driver: DriverLike,
    figure_records,
    *,
    batch_size: int = BATCH_SIZE,
) -> int:
    """MERGE :Figure nodes and HAS_FIGURE edges from *figure_records*.

    *figure_records* must be an iterable of
    ``ingestion.processing.figures.FigureRecord``.  The function keys figures
    idempotently by ``FigureRecord.figure_id`` so re-runs are safe.

    Returns the number of Figure nodes upserted.
    """
    fig_rows: list[dict] = []
    link_rows: list[dict] = []

    for fig in figure_records:
        row = {
            "id": fig.figure_id,
            "doc_id": fig.doc_id,
            "page": fig.page,
            "bbox": fig.bbox,
            "caption": fig.caption,
            "ocr_text": fig.ocr_text,
        }
        fig_rows.append(row)

        # HAS_FIGURE edges only for TSB occurrences (the only corpus with
        # Occurrence nodes at this stage).  TC/ZH support can extend this later.
        if fig.doc_id.startswith("tsb/"):
            occ_id = fig.doc_id.split("/", 1)[1]
            link_rows.append({"occ_id": occ_id, "fig_id": fig.figure_id})

    with driver.session() as session:
        _batch_run(session, _UPSERT_FIGURE, fig_rows, batch_size)
        _batch_run(session, _LINK_OCC_FIGURE, _dedup_link_rows(link_rows), batch_size)

    logger.info("upserted %d Figure nodes (%d HAS_FIGURE links)", len(fig_rows), len(link_rows))
    return len(fig_rows)
