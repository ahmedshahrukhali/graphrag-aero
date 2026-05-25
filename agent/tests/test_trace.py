"""Tests for trace_from_history — a synthetic graph stub stands in for LangGraph."""
from agent.trace import trace_from_history


class FakeSnap:
    def __init__(self, values, nxt=()):
        self.values = values
        self.next = nxt


class FakeGraph:
    """Returns ``history`` newest-first, matching real LangGraph semantics."""

    def __init__(self, history_newest_first):
        self._hist = history_newest_first

    def get_state_history(self, config):  # noqa: ARG002
        return iter(self._hist)


def test_orders_chronologically_oldest_first():
    # Real LangGraph yields newest-first; we feed it that way, expect oldest-first out.
    newest_first = [
        FakeSnap({"hop": 3, "final": "x"}, nxt=()),
        FakeSnap({"hop": 2, "draft": "x"}, nxt=("finalize",)),
        FakeSnap({"hop": 1, "candidates": [{"chunk_hash": "a"}]}, nxt=("synthesize",)),
        FakeSnap({"hop": 0, "candidates": []}, nxt=("retrieve",)),
    ]
    out = trace_from_history(FakeGraph(newest_first), {"configurable": {}})
    # 4 entries, hops ascend.
    assert [e["values_summary"]["hop"] for e in out] == [0, 1, 2, 3]
    assert out[0]["step"] == 0 and out[-1]["step"] == 3


def test_summary_fields():
    h = [FakeSnap({"hop": 2, "draft": "X", "final": "Y", "candidates": [1, 2]}, nxt=())]
    out = trace_from_history(FakeGraph(h), {"configurable": {}})
    s = out[0]["values_summary"]
    assert s["hop"] == 2
    assert s["n_candidates"] == 2
    assert s["draft_present"] is True
    assert s["final_present"] is True


def test_empty_history():
    assert trace_from_history(FakeGraph([]), {"configurable": {}}) == []


def test_next_tuple_preserved():
    h = [FakeSnap({"hop": 0}, nxt=("retrieve",))]
    out = trace_from_history(FakeGraph(h), {"configurable": {}})
    assert out[0]["next"] == ("retrieve",)
