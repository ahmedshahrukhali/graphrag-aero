"""Graph traversal queries.

graph_context_for_occurrences traverses:
  Occurrence -[:HAS_FINDING]-> Finding -[:CITES]-> Regulation
  Occurrence -[:HAS_RECOMMENDATION]-> Recommendation
  Occurrence -[:CITES]-> Regulation
  Occurrence -[:REFERENCES_AC]-> AC

Every returned row carries source_doc_id + page so the agent can cite it
inline (the graph facts stay grounded, same discipline as vector chunks).
"""
from __future__ import annotations

import logging
from typing import Iterable

from .client import DriverLike


logger = logging.getLogger(__name__)


# Single query: one round-trip per call. OPTIONAL MATCH so occurrences
# without findings/regs still return their basic record.
_TRAVERSE_CYPHER = """
UNWIND $ids AS occ_id
MATCH (o:Occurrence {id: occ_id})

// findings + their regulation citations
OPTIONAL MATCH (o)-[:HAS_FINDING]->(f:Finding)
OPTIONAL MATCH (f)-[:CITES]->(fr:Regulation)

// recommendations
OPTIONAL MATCH (o)-[:HAS_RECOMMENDATION]->(rec:Recommendation)

// direct regulation + AC links (cited in text, not necessarily in a finding)
OPTIONAL MATCH (o)-[:CITES]->(dr:Regulation)
OPTIONAL MATCH (o)-[:REFERENCES_AC]->(ac:AC)

RETURN
  o.id              AS occ_id,
  o.source_url      AS occ_url,
  collect(DISTINCT {
    text:           f.text,
    category:       f.category,
    lang:           f.lang,
    source_doc_id:  f.source_doc_id,
    page:           f.page,
    cites_reg:      fr.id
  })                AS findings,
  collect(DISTINCT {
    id:             rec.id,
    text:           rec.text,
    lang:           rec.lang,
    source_doc_id:  rec.source_doc_id,
    page:           rec.page
  })                AS recommendations,
  collect(DISTINCT dr.id)  AS direct_regs,
  collect(DISTINCT ac.id)  AS acs
""".strip()


def _clean_collect(items: list[dict | None], required_key: str) -> list[dict]:
    """Neo4j COLLECT on optional matches can return [{key: null, ...}]; strip them."""
    return [d for d in (items or [])
            if isinstance(d, dict) and d.get(required_key) is not None]


def graph_context_for_occurrences(
    driver: DriverLike,
    occurrence_ids: Iterable[str],
) -> list[dict]:
    """Traverse the knowledge graph for each occurrence and return cited facts.

    Each returned dict has shape:
      {
        occ_id, occ_url,
        findings:        [{text, category, lang, source_doc_id, page, cites_reg}],
        recommendations: [{id, text, lang, source_doc_id, page}],
        direct_regs:     [str],   # CAR numbers cited directly on occurrence
        acs:             [str],   # AC ids referenced
      }

    Missing occurrences are silently dropped.
    """
    ids = list(dict.fromkeys(occurrence_ids))
    if not ids:
        return []

    out: list[dict] = []
    with driver.session() as session:
        result = session.run(_TRAVERSE_CYPHER, ids=ids)
        for row in result:
            out.append({
                "occ_id":          row["occ_id"],
                "occ_url":         row["occ_url"],
                "findings":        _clean_collect(row["findings"], "text"),
                "recommendations": _clean_collect(row["recommendations"], "text"),
                "direct_regs":     [r for r in (row["direct_regs"] or []) if r],
                "acs":             [a for a in (row["acs"] or []) if a],
            })

    logger.debug("graph_context: %d ids in, %d rows out", len(ids), len(out))
    return out
