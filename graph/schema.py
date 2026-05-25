"""Neo4j schema: constraints + indexes for the aerospace knowledge graph.

Schema diagram (relationships typed but empty until richer extraction lands):

    (:Occurrence)-[:INVOLVES]->(:Aircraft)
    (:Occurrence)-[:HAS_FINDING]->(:Finding)
    (:Finding)-[:LED_TO]->(:Recommendation)
    (:Recommendation)-[:CITES]->(:Regulation)
    (:Regulation)-[:GUIDED_BY]->(:AC)

``init_schema`` uses ``IF NOT EXISTS`` so re-running is a no-op.
"""
from __future__ import annotations

import logging

from .client import DriverLike


logger = logging.getLogger(__name__)


CONSTRAINTS: tuple[str, ...] = (
    "CREATE CONSTRAINT occurrence_id     IF NOT EXISTS FOR (o:Occurrence)     REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT aircraft_id       IF NOT EXISTS FOR (a:Aircraft)       REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT finding_id        IF NOT EXISTS FOR (f:Finding)        REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT recommendation_id IF NOT EXISTS FOR (r:Recommendation) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT regulation_id     IF NOT EXISTS FOR (r:Regulation)     REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT ac_id             IF NOT EXISTS FOR (a:AC)             REQUIRE a.id IS UNIQUE",
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX occurrence_lang IF NOT EXISTS FOR (o:Occurrence) ON (o.lang)",
)


def init_schema(driver: DriverLike) -> int:
    """Apply all constraints + indexes. Returns the count of statements run."""
    n = 0
    with driver.session() as session:
        for stmt in CONSTRAINTS + INDEXES:
            logger.debug("cypher: %s", stmt)
            session.run(stmt)
            n += 1
    logger.info("schema: applied %d statements", n)
    return n
