"""Negative-feedback store: ``unaccepted_qa`` table in the Postgres DB.

When the user explicitly rejects an answer, we record the query, its dense
embedding, the answer, and the chunk hashes that produced it.

On the next query whose embedding is cosine-similar (>= threshold) to a
rejected row, the retrieve node:
  (a) sets ``state["excluded_chunk_hashes"]`` so those chunks are skipped,
  (b) sets ``state["rejected_prior"]`` so the synthesise prompt notes the
      prior failure and avoids the same framing.

All vector similarity is computed in Python (table stays small — the typical
user accumulates tens, not millions, of rejected answers).

Protocol
--------
``FeedbackStore`` is constructed with a ``get_conn`` callable that returns a
psycopg3 ``Connection`` (or any object with compatible ``.cursor()`` /
``.commit()`` API).  Tests inject a ``FakeConn`` (see agent/tests/).
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)


# ─── vector math ─────────────────────────────────────────────────────────────

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ─── store ───────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS unaccepted_qa (
    id          SERIAL PRIMARY KEY,
    query       TEXT NOT NULL,
    query_emb   TEXT NOT NULL,
    answer      TEXT NOT NULL,
    chunk_hashes TEXT NOT NULL,
    terms       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
)
"""

_DEFAULT_THRESHOLD = 0.80


class FeedbackStore:
    """Persistence for rejected QA pairs.

    Parameters
    ----------
    get_conn:
        Zero-argument callable that returns a DB connection.  Called once per
        operation so connection pooling / lifecycle live outside this class.
    threshold:
        Cosine-similarity cutoff for ``find_similar``.
    """

    def __init__(
        self,
        get_conn: Callable[[], Any],
        *,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._get_conn = get_conn
        self.threshold = threshold

    def create_table(self) -> None:
        """Create the ``unaccepted_qa`` table if it does not exist. Idempotent."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
        conn.commit()

    def write_rejection(
        self,
        query: str,
        query_embedding: Sequence[float],
        answer: str,
        chunk_hashes: Sequence[str],
        terms: Sequence[str] | None = None,
    ) -> int:
        """Insert a rejected QA row.  Returns the new row ``id``."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO unaccepted_qa "
                "(query, query_emb, answer, chunk_hashes, terms) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (
                    query,
                    json.dumps(list(query_embedding)),
                    answer,
                    json.dumps(list(chunk_hashes)),
                    json.dumps(list(terms)) if terms else None,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        row_id: int = row[0]
        logger.info("wrote rejection row id=%d query=%r", row_id, query[:60])
        return row_id

    def find_similar(
        self,
        query_embedding: Sequence[float],
        *,
        threshold: float | None = None,
    ) -> list[dict]:
        """Return unresolved rows whose stored embedding is similar to ``query_embedding``.

        Each result dict has keys:
          id, query, answer, chunk_hashes (list[str]), terms (list[str]), similarity.
        Results are sorted by descending similarity.
        """
        thr = threshold if threshold is not None else self.threshold
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, query, query_emb, answer, chunk_hashes, terms "
                "FROM unaccepted_qa WHERE resolved_at IS NULL"
            )
            rows = cur.fetchall()

        qe = list(query_embedding)
        results: list[dict] = []
        for row_id, row_query, emb_json, row_answer, hashes_json, terms_json in rows:
            emb = json.loads(emb_json) if isinstance(emb_json, str) else emb_json
            sim = _cosine(qe, emb)
            if sim >= thr:
                results.append(
                    dict(
                        id=row_id,
                        query=row_query,
                        answer=row_answer,
                        chunk_hashes=json.loads(hashes_json)
                        if isinstance(hashes_json, str)
                        else (hashes_json or []),
                        terms=json.loads(terms_json)
                        if (terms_json and isinstance(terms_json, str))
                        else (terms_json or []),
                        similarity=sim,
                    )
                )
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results

    def resolve(self, row_id: int) -> None:
        """Mark a rejected row resolved (cleared by a successful retry)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE unaccepted_qa SET resolved_at = %s WHERE id = %s",
                (datetime.now(timezone.utc), row_id),
            )
        conn.commit()
        logger.info("resolved rejection row id=%d", row_id)


# ─── DSN-based factory ───────────────────────────────────────────────────────

def make_feedback_store(dsn: str | None = None) -> FeedbackStore:
    """Build a ``FeedbackStore`` from a Postgres DSN.

    Lazy import of psycopg so tests that inject a fake conn can avoid the dep.
    DSN defaults to ``POSTGRES_DSN`` env var (same as the checkpointer).
    """
    import os

    resolved_dsn = dsn or os.environ.get("POSTGRES_DSN")
    if not resolved_dsn:
        raise RuntimeError("POSTGRES_DSN must be set for FeedbackStore")

    import psycopg  # type: ignore

    def get_conn():
        return psycopg.connect(resolved_dsn)

    return FeedbackStore(get_conn)
