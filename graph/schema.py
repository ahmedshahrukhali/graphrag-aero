"""Neo4j schema: constraints + indexes for the aerospace knowledge graph.

Schema diagram (§4 Document-rooted):

    (:Document:Occurrence)-[:INVOLVES]->(:Aircraft)
    (:Document:Occurrence)-[:HAS_FINDING]->(:Finding)
    (:Document:Occurrence)-[:HAS_RECOMMENDATION]->(:Recommendation)
    (:Finding)-[:CITES]->(:Regulation)
    (:Recommendation)-[:IMPLEMENTS]->(:Regulation)
    (:Regulation)-[:GUIDED_BY]->(:Document:AC)

Document is the generic root label; Occurrence and AC are typed subtypes.
The dispatch seam in extract.py allows other doc types (manuals, drawings) to
also carry :Document when their corpora are added.

``init_schema`` uses ``IF NOT EXISTS`` DDL then WHERE-guarded migrations so
re-running is a no-op at any point.
"""
from __future__ import annotations

import logging

from .client import DriverLike


logger = logging.getLogger(__name__)


CONSTRAINTS: tuple[str, ...] = (
    # Generic Document root — doc_id is the full corpus path, e.g. "tsb/a13q0098"
    "CREATE CONSTRAINT document_doc_id     IF NOT EXISTS FOR (d:Document)        REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT occurrence_id       IF NOT EXISTS FOR (o:Occurrence)       REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT aircraft_id         IF NOT EXISTS FOR (a:Aircraft)         REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT finding_id          IF NOT EXISTS FOR (f:Finding)          REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT recommendation_id   IF NOT EXISTS FOR (r:Recommendation)   REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT regulation_id       IF NOT EXISTS FOR (r:Regulation)       REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT ac_id               IF NOT EXISTS FOR (a:AC)               REQUIRE a.id IS UNIQUE",
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX occurrence_lang    IF NOT EXISTS FOR (o:Occurrence) ON (o.lang)",
    "CREATE INDEX finding_source     IF NOT EXISTS FOR (f:Finding)    ON (f.source_doc_id)",
)

# WHERE-guarded data migrations — idempotent; safe to re-run.
MIGRATIONS: tuple[str, ...] = (
    # Backfill :Document label + doc_id on Occurrence nodes created before §4
    "MATCH (o:Occurrence) WHERE o.doc_id IS NULL SET o:Document, o.doc_id = 'tsb/' + o.id",
    # Backfill :Document label on AC nodes created before §4 (doc_id set on next upsert)
    "MATCH (a:AC) WHERE NOT a:Document AND a.doc_id IS NULL SET a:Document",
)


def init_schema(driver: DriverLike) -> int:
    """Apply DDL constraints + indexes + backfill migrations. Returns statement count."""
    n = 0
    with driver.session() as session:
        for stmt in CONSTRAINTS + INDEXES + MIGRATIONS:
            logger.debug("cypher: %s", stmt)
            session.run(stmt)
            n += 1
    logger.info("schema: applied %d statements", n)
    return n
