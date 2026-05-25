"""Tests for graph_context_for_occurrences with a FakeDriver."""
from graph.query import FETCH_OCCURRENCES_CYPHER, graph_context_for_occurrences


class FakeSession:
    """Session that, on .run(cypher, ids=...), returns canned rows for each id
    in ``self._table`` and silently drops misses (matches MATCH semantics)."""

    def __init__(self, table: dict[str, dict]):
        self._table = table
        self.last_cypher: str | None = None
        self.last_params: dict | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def run(self, cypher, **params):
        self.last_cypher = cypher
        self.last_params = dict(params)
        rows = []
        for occ_id in params["ids"]:
            if occ_id in self._table:
                rows.append(self._table[occ_id])
        return iter(rows)


class FakeDriver:
    def __init__(self, table: dict[str, dict]):
        self.session_obj = FakeSession(table)

    def session(self, **kwargs):
        return self.session_obj

    def close(self):
        pass


def test_returns_rows_for_known_ids():
    table = {
        "a01": {"id": "a01", "source_url": "u-a01", "lang": "en"},
        "b02": {"id": "b02", "source_url": "u-b02", "lang": "fr"},
    }
    d = FakeDriver(table)
    out = graph_context_for_occurrences(d, ["a01", "b02"])
    assert out == [
        {"id": "a01", "source_url": "u-a01", "lang": "en"},
        {"id": "b02", "source_url": "u-b02", "lang": "fr"},
    ]
    assert d.session_obj.last_cypher == FETCH_OCCURRENCES_CYPHER


def test_drops_unknown_ids():
    d = FakeDriver({"a01": {"id": "a01", "source_url": "u", "lang": "en"}})
    out = graph_context_for_occurrences(d, ["a01", "ghost"])
    assert len(out) == 1
    assert out[0]["id"] == "a01"


def test_empty_input_skips_session():
    d = FakeDriver({})
    out = graph_context_for_occurrences(d, [])
    assert out == []
    # No session.run() call should have happened.
    assert d.session_obj.last_cypher is None


def test_dedupes_input():
    d = FakeDriver({"a01": {"id": "a01", "source_url": "u", "lang": "en"}})
    graph_context_for_occurrences(d, ["a01", "a01", "a01"])
    # Implementation dedupes; the session should see one "a01" in the params.
    assert d.session_obj.last_params is not None
    assert d.session_obj.last_params["ids"].count("a01") == 1
