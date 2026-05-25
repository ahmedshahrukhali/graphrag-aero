"""Checkpoint factories.

LangGraph persists agent state via a checkpointer. Production uses
``PostgresSaver`` (so paused HITL sessions survive restarts); tests use
``MemorySaver`` (in-process, ephemeral).
"""
from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def make_memory_saver() -> Any:
    """In-process saver. Used by tests + smoke runs."""
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
    return MemorySaver()


def make_postgres_saver(dsn: str | None = None) -> Any:
    """Postgres-backed saver.

    Returns a ``PostgresSaver`` from ``langgraph.checkpoint.postgres``. The
    caller must call ``.setup()`` once per fresh database to create the
    checkpoint tables.

    Lazy import: ``langgraph-checkpoint-postgres`` pulls psycopg, which we
    only need for the agent runtime — not for the test suite.
    """
    dsn = dsn or os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN must be set for PostgresSaver")
    from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore

    logger.info("connecting to postgres: %s", dsn.split("@", 1)[-1])
    # Modern langgraph-checkpoint-postgres exposes a context manager:
    # ``with PostgresSaver.from_conn_string(dsn) as cp: cp.setup(); yield cp``
    # We return the underlying saver; the agent CLI owns lifecycle.
    return PostgresSaver.from_conn_string(dsn)
