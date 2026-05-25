"""Neo4j driver factory.

The driver is a `neo4j.Driver` object; both real and fake implementations expose
``.session()`` returning a context manager whose ``.run(cypher, **params)``
yields records. Tests pass a fake; production reads URI/USER/PASSWORD from env.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "please_change_me"),
        )


class _SessionLike(Protocol):
    def run(self, cypher: str, **params): ...
    def __enter__(self): ...
    def __exit__(self, *exc): ...


class DriverLike(Protocol):
    """The slice of ``neo4j.Driver`` we actually use."""

    def session(self, **kwargs) -> _SessionLike: ...
    def close(self) -> None: ...


def make_driver(cfg: Neo4jConfig | None = None) -> DriverLike:
    """Lazily import neo4j so tests don't need the driver installed."""
    from neo4j import GraphDatabase  # type: ignore

    cfg = cfg or Neo4jConfig.from_env()
    logger.info("connecting to neo4j: %s", cfg.uri)
    return GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
