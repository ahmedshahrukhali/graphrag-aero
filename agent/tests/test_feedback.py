"""Tests for agent.feedback.FeedbackStore — offline (FakeConn, no Postgres)."""
from __future__ import annotations

import json
import math

import pytest

from agent.feedback import FeedbackStore, _cosine


# ─── helpers ─────────────────────────────────────────────────────────────────

class _Cursor:
    """Minimal cursor mock that records SQL calls and can replay responses."""

    def __init__(self, rows=None):
        self.calls: list[tuple[str, tuple]] = []
        self._rows = rows or []
        self._last_id: int | None = None

    def execute(self, sql: str, params: tuple = ()):
        self.calls.append((sql.strip(), params))
        if "RETURNING" in sql.upper():
            self._last_id = 1

    def fetchone(self):
        if self._last_id is not None:
            r = (self._last_id,)
            self._last_id = None
            return r
        return None

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class FakeConn:
    """Minimal connection mock."""

    def __init__(self, rows=None):
        self._cursor = _Cursor(rows=rows)
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


def _unit_vec(n: int, dim: int = 8) -> list[float]:
    v = [0.0] * dim
    v[n % dim] = 1.0
    return v


def _store(rows=None) -> tuple[FeedbackStore, FakeConn]:
    conn = FakeConn(rows=rows)
    return FeedbackStore(lambda: conn), conn


# ─── _cosine ─────────────────────────────────────────────────────────────────

def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_partial_overlap():
    a = [1.0, 1.0, 0.0]
    b = [1.0, 0.0, 1.0]
    expected = 1.0 / (math.sqrt(2) * math.sqrt(2))
    assert _cosine(a, b) == pytest.approx(expected)


# ─── create_table ────────────────────────────────────────────────────────────

def test_create_table_executes_ddl():
    store, conn = _store()
    store.create_table()
    sql, _ = conn.cursor().calls[0]
    assert "CREATE TABLE" in sql.upper()
    assert "unaccepted_qa" in sql
    assert conn.committed


# ─── write_rejection ─────────────────────────────────────────────────────────

def test_write_rejection_inserts_row_and_returns_id():
    store, conn = _store()
    row_id = store.write_rejection(
        "fuel starvation",
        _unit_vec(0),
        "The answer was wrong.",
        ["hash1", "hash2"],
        terms=["flapper", "valve"],
    )
    assert row_id == 1
    assert conn.committed
    sql, params = conn.cursor().calls[0]
    assert "INSERT INTO unaccepted_qa" in sql
    # params: query, emb_json, answer, hashes_json, terms_json
    assert params[0] == "fuel starvation"
    assert json.loads(params[1]) == _unit_vec(0)
    assert json.loads(params[3]) == ["hash1", "hash2"]


def test_write_rejection_terms_optional():
    store, conn = _store()
    store.write_rejection("q", _unit_vec(0), "a", ["h1"])
    _, params = conn.cursor().calls[0]
    assert params[4] is None


# ─── find_similar ─────────────────────────────────────────────────────────────

def _db_row(row_id, query, emb, answer, hashes, terms=None):
    return (
        row_id,
        query,
        json.dumps(emb),
        answer,
        json.dumps(hashes),
        json.dumps(terms) if terms else None,
    )


def test_find_similar_returns_above_threshold():
    emb = _unit_vec(0)
    rows = [_db_row(1, "similar q", emb, "bad answer", ["h1"])]
    store, _ = _store(rows=rows)
    results = store.find_similar(emb, threshold=0.8)
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["chunk_hashes"] == ["h1"]
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_find_similar_skips_below_threshold():
    emb_a = _unit_vec(0)
    emb_b = _unit_vec(1)
    rows = [_db_row(1, "q", emb_b, "a", ["h1"])]
    store, _ = _store(rows=rows)
    results = store.find_similar(emb_a, threshold=0.8)
    assert results == []


def test_find_similar_sorted_descending():
    emb_query = [1.0, 0.0, 0.0]
    emb_close = [0.9, 0.1, 0.0]
    emb_far = [0.7, 0.3, 0.0]
    rows = [
        _db_row(1, "q1", emb_far, "a1", ["h1"]),
        _db_row(2, "q2", emb_close, "a2", ["h2"]),
    ]
    store, _ = _store(rows=rows)
    results = store.find_similar(emb_query, threshold=0.5)
    assert len(results) == 2
    assert results[0]["id"] == 2  # closer one first


def test_find_similar_uses_default_threshold():
    emb = _unit_vec(0)
    rows = [_db_row(1, "q", emb, "a", ["h"])]
    store, _ = _store(rows=rows)
    # Default threshold = 0.80; identical vectors → sim=1.0 → should match.
    results = store.find_similar(emb)
    assert len(results) == 1


# ─── resolve ─────────────────────────────────────────────────────────────────

def test_resolve_updates_resolved_at():
    store, conn = _store()
    store.resolve(42)
    assert conn.committed
    sql, params = conn.cursor().calls[0]
    assert "UPDATE" in sql.upper() and "unaccepted_qa" in sql.lower()
    assert "resolved_at" in sql.lower()
    assert params[1] == 42
