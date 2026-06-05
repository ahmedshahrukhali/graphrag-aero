"""Graph traversal queries.

graph_context_for_occurrences traverses the §4 schema depth:

  Occurrence -[:HAS_FINDING]->     Finding       -[:CITES]->     Regulation
  Occurrence -[:HAS_RECOMMENDATION]-> Recommendation -[:IMPLEMENTS]-> Regulation
                                                     -[:GUIDED_BY]->  AC
  Occurrence -[:CITES]->           Regulation
  Occurrence -[:REFERENCES_AC]->   AC

Every returned row carries source_doc_id + page so the agent can cite it
inline (the graph facts stay grounded, same discipline as vector chunks).
"""
from __future__ import annotations

import logging
from typing import Iterable

from .client import DriverLike


logger = logging.getLogger(__name__)


# Single query: one round-trip per call. OPTIONAL MATCH so occurrences
# without findings/recs/regs still return their basic record.
# §4: adds Recommendation-[:IMPLEMENTS]->Regulation-[:GUIDED_BY]->AC second hop.
_TRAVERSE_CYPHER = """
UNWIND $ids AS occ_id
MATCH (o:Occurrence {id: occ_id})

// findings + their regulation citations
OPTIONAL MATCH (o)-[:HAS_FINDING]->(f:Finding)
OPTIONAL MATCH (f)-[:CITES]->(fr:Regulation)

// recommendations + what they implement + ACs that elaborate those regs
OPTIONAL MATCH (o)-[:HAS_RECOMMENDATION]->(rec:Recommendation)
OPTIONAL MATCH (rec)-[:IMPLEMENTS]->(rr:Regulation)
OPTIONAL MATCH (rr)-[:GUIDED_BY]->(ra:AC)

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
  collect(DISTINCT dr.id)   AS direct_regs,
  collect(DISTINCT ac.id)   AS acs,
  collect(DISTINCT rr.id)   AS rec_regs,
  collect(DISTINCT ra.id)   AS reg_guided_acs
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
        rec_regs:        [str],   # §4 CAR numbers implemented by recommendations
        reg_guided_acs:  [str],   # §4 AC ids that elaborate those regs (2nd hop)
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
                "rec_regs":        [r for r in (row.get("rec_regs") or []) if r],
                "reg_guided_acs":  [a for a in (row.get("reg_guided_acs") or []) if a],
            })

    logger.debug("graph_context: %d ids in, %d rows out", len(ids), len(out))
    return out


# Outward second hop: regulations the seed occurrences cite that OTHER
# occurrences also cite. We traverse the *direct* Occurrence-[:CITES]->Regulation
# edge — the populated one (regex-extracted, ~875 edges) — not the
# Finding-[:CITES]->Regulation edge, which the LLM extractor barely filled
# (~166 edges) so it surfaces nothing for most occurrences. The reverse leg
# (r)<-[:CITES]-(o) gives sibling reports; count(DISTINCT o) is the recurrence
# degree. Selectivity lives in the WHERE: keep only regs shared beyond the seeds
# (deg > #seeds) and drop promiscuous hubs (deg <= $max_reg_degree, e.g. generic
# "general provisions" CARs cited by dozens of reports) so breadth is signal.
_RECURRING_CYPHER = """
UNWIND $ids AS seed_id
MATCH (s:Occurrence {id: seed_id})-[:CITES]->(r:Regulation)
WITH collect(DISTINCT seed_id) AS seeds, collect(DISTINCT r.id) AS reg_ids
UNWIND reg_ids AS reg_id
MATCH (r:Regulation {id: reg_id})<-[:CITES]-(o:Occurrence)
WITH seeds, reg_id, count(DISTINCT o) AS deg, collect(DISTINCT o.id) AS occ_ids
WHERE deg > size(seeds) AND deg <= $max_reg_degree
RETURN reg_id AS reg, deg AS occ_count,
       [x IN occ_ids WHERE NOT x IN seeds] AS sibling_ids
ORDER BY occ_count DESC
LIMIT $max_regs
""".strip()


def recurring_context_for_occurrences(
    driver: DriverLike,
    occurrence_ids: Iterable[str],
    *,
    max_regs: int = 5,
    max_siblings_per_reg: int = 3,
    max_reg_degree: int = 15,
) -> list[dict]:
    """Outward second hop — recurring regulatory threads across *other* reports.

    For the seed occurrences, find the regulations they cite (direct
    Occurrence→CITES→Regulation), then surface OTHER occurrences citing the same
    regulation. A regulation is kept only if it recurs beyond the seeds
    (``deg > #seeds``) and isn't a promiscuous hub (``deg <= max_reg_degree``) —
    the graph-IDF that keeps breadth meaningful rather than noise.

    Each returned row:
        {reg, occ_count, siblings: [{occ_id, source_doc_id}]}
    ordered by ``occ_count`` desc, capped at ``max_regs`` regs and
    ``max_siblings_per_reg`` siblings. Siblings cite at the report level
    ([tsb/<occ_id>]); Occurrence nodes carry no page, so we don't fabricate one.

    Returns ``[]`` when the seeds share no cited regulation with other
    occurrences — a direct signal the graph is too sparse for this query.
    """
    ids = list(dict.fromkeys(occurrence_ids))
    if not ids:
        return []

    out: list[dict] = []
    with driver.session() as session:
        result = session.run(
            _RECURRING_CYPHER,
            ids=ids,
            max_regs=max_regs,
            max_reg_degree=max_reg_degree,
        )
        for row in result:
            sib_ids = [x for x in (row.get("sibling_ids") or []) if x][:max_siblings_per_reg]
            if not sib_ids:
                continue
            # Occurrence ids map back to doc_ids as "tsb/<id>" (Occurrence nodes
            # are minted only for the tsb corpus), so siblings are citable.
            sibs = [{"occ_id": x, "source_doc_id": f"tsb/{x}"} for x in sib_ids]
            out.append({
                "reg": row.get("reg"),
                "occ_count": row.get("occ_count", 0),
                "siblings": sibs,
            })

    logger.debug("recurring_context: %d seeds in, %d recurring regs out", len(ids), len(out))
    return out
