"""Tests for graph_context_for_occurrences with the new traversal shape."""
from graph.query import (
    graph_context_for_occurrences,
    recurring_context_for_occurrences,
)


class FakeSession:
    def __init__(self, table: dict):
        self._table = table
        self.last_ids: list | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def run(self, cypher, **params):
        self.last_ids = params.get("ids")
        rows = []
        for occ_id in (self.last_ids or []):
            if occ_id in self._table:
                rows.append(self._table[occ_id])
        return iter(rows)


class FakeDriver:
    def __init__(self, table: dict):
        self.session_obj = FakeSession(table)

    def session(self, **kwargs):
        return self.session_obj

    def close(self):
        pass


def _rich_row(occ_id: str, findings=None, recs=None, regs=None, acs=None) -> dict:
    return {
        "occ_id": occ_id,
        "occ_url": f"https://bst-tsb.gc.ca/{occ_id}",
        "findings": findings or [],
        "recommendations": recs or [],
        "direct_regs": regs or [],
        "acs": acs or [],
    }


def test_returns_rich_rows_for_known_ids():
    finding = {"text": "Fuel tanks empty.", "category": "cause", "lang": "en",
                "source_doc_id": "tsb/a01", "page": 5, "cites_reg": "602.115"}
    table = {"a01": _rich_row("a01", findings=[finding])}
    d = FakeDriver(table)
    out = graph_context_for_occurrences(d, ["a01"])
    assert len(out) == 1
    assert out[0]["occ_id"] == "a01"
    assert len(out[0]["findings"]) == 1
    assert out[0]["findings"][0]["cites_reg"] == "602.115"


def test_drops_unknown_ids():
    d = FakeDriver({"a01": _rich_row("a01")})
    out = graph_context_for_occurrences(d, ["a01", "ghost"])
    assert len(out) == 1


def test_empty_input_skips_session():
    d = FakeDriver({})
    out = graph_context_for_occurrences(d, [])
    assert out == []
    assert d.session_obj.last_ids is None


def test_dedupes_input():
    d = FakeDriver({"a01": _rich_row("a01")})
    graph_context_for_occurrences(d, ["a01", "a01", "a01"])
    assert d.session_obj.last_ids is not None
    assert d.session_obj.last_ids.count("a01") == 1


def test_null_findings_in_collect_stripped():
    # Neo4j COLLECT on OPTIONAL MATCH can return [{key: null, ...}] — must be cleaned.
    row = _rich_row("a01")
    row["findings"] = [{"text": None, "category": None, "lang": None,
                         "source_doc_id": None, "page": None, "cites_reg": None}]
    d = FakeDriver({"a01": row})
    out = graph_context_for_occurrences(d, ["a01"])
    # _clean_collect strips rows where "text" is None
    assert out[0]["findings"] == []


def test_occurrence_with_direct_regs_and_acs():
    row = _rich_row("a01", regs=["602.88", "602.115"], acs=["700-027"])
    d = FakeDriver({"a01": row})
    out = graph_context_for_occurrences(d, ["a01"])
    assert "602.88" in out[0]["direct_regs"]
    assert "700-027" in out[0]["acs"]


# ─── recurring_context_for_occurrences (outward second hop) ───────────────────

class _RowsSession:
    """Session that yields a fixed list of rows for any run() — the recurring
    query returns reg-shaped rows, not per-occurrence rows, so the keyed
    FakeSession above doesn't model it."""
    def __init__(self, rows):
        self._rows = rows
        self.last_ids = None
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return None
    def run(self, cypher, **params):
        self.last_ids = params.get("ids")
        return iter(self._rows)


class _RowsDriver:
    def __init__(self, rows):
        self.session_obj = _RowsSession(rows)
    def session(self, **kwargs):
        return self.session_obj
    def close(self):
        pass


def test_recurring_builds_citable_siblings_and_caps():
    rows = [{"reg": "703.07", "occ_count": 4, "sibling_ids": ["a02", "a03", "a04", "a05"]}]
    out = recurring_context_for_occurrences(_RowsDriver(rows), ["a01"], max_siblings_per_reg=3)
    assert len(out) == 1
    assert out[0]["reg"] == "703.07" and out[0]["occ_count"] == 4
    sibs = out[0]["siblings"]
    assert [s["occ_id"] for s in sibs] == ["a02", "a03", "a04"]  # capped at 3
    # occ_id maps back to a citable doc_id
    assert sibs[0]["source_doc_id"] == "tsb/a02"


def test_recurring_skips_regs_without_nonseed_siblings():
    rows = [{"reg": "602.07", "occ_count": 2, "sibling_ids": []}]
    out = recurring_context_for_occurrences(_RowsDriver(rows), ["a01"])
    assert out == []


def test_recurring_empty_ids_skips_session():
    d = _RowsDriver([])
    out = recurring_context_for_occurrences(d, [])
    assert out == []
    assert d.session_obj.last_ids is None


def test_recurring_dedups_input_ids():
    d = _RowsDriver([])
    recurring_context_for_occurrences(d, ["a01", "a01", "a01"])
    assert d.session_obj.last_ids is not None
    assert d.session_obj.last_ids.count("a01") == 1
