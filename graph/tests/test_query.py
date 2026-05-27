"""Tests for graph_context_for_occurrences with the new traversal shape."""
from graph.query import graph_context_for_occurrences


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
