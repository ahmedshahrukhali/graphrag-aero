"""Graph traversal queries.

This phase is minimal: hydrate Occurrence nodes by id. When richer extraction
lands, ``graph_context_for_occurrences`` grows to follow ``INVOLVES`` /
``HAS_FINDING`` / etc. and return related entities.
"""
from __future__ import annotations

import logging
from typing import Iterable

from .client import DriverLike


logger = logging.getLogger(__name__)


FETCH_OCCURRENCES_CYPHER = """
UNWIND $ids AS occ_id
MATCH (o:Occurrence {id: occ_id})
RETURN o.id AS id, o.source_url AS source_url, o.lang AS lang
""".strip()


def graph_context_for_occurrences(
    driver: DriverLike,
    occurrence_ids: Iterable[str],
) -> list[dict]:
    """Return ``[{id, source_url, lang}]`` for the given occurrence ids.

    Missing ids are silently dropped (the agent shouldn't crash if a chunk
    references an occurrence we haven't ingested yet).
    """
    ids = list(dict.fromkeys(occurrence_ids))  # preserve order, drop dups
    if not ids:
        return []
    out: list[dict] = []
    with driver.session() as session:
        result = session.run(FETCH_OCCURRENCES_CYPHER, ids=ids)
        for row in result:
            out.append({
                "id": row["id"],
                "source_url": row["source_url"],
                "lang": row["lang"],
            })
    logger.debug("graph_context: %d ids in, %d rows out", len(ids), len(out))
    return out
