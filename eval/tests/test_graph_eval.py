"""Tests for eval/graph_eval.py — traversal hit metric + dataset loader."""
import json
from pathlib import Path

import pytest

from eval.graph_eval import (
    GraphEvalItem,
    GraphItemResult,
    _keyword_hit,
    _score_item,
    graph_evaluate,
    load_graph_dataset,
)


# ─── _keyword_hit ────────────────────────────────────────────────────────────

def test_keyword_hit_match():
    findings = [{"text": "Fuel tanks were exhausted on arrival."}]
    assert _keyword_hit(findings, ["fuel", "exhaust"]) is True


def test_keyword_hit_miss():
    findings = [{"text": "The weather was calm and clear."}]
    assert _keyword_hit(findings, ["fuel", "exhaust"]) is False


def test_keyword_hit_no_keywords_always_passes():
    assert _keyword_hit([], []) is True
    assert _keyword_hit([{"text": "x"}], []) is True


def test_keyword_hit_case_insensitive():
    findings = [{"text": "FUEL EXHAUSTION was the cause."}]
    assert _keyword_hit(findings, ["fuel"]) is True


# ─── _score_item ─────────────────────────────────────────────────────────────

def _item(kw: list[str]) -> GraphEvalItem:
    return GraphEvalItem(
        id="g01", query="q", occ_id="a01",
        expect_regs=[], expect_finding_keywords=kw, lang="en", note="",
    )


def test_score_item_hit():
    ctx = {"findings": [{"text": "Fuel exhausted."}], "recommendations": [],
           "direct_regs": [], "acs": []}
    result = _score_item(_item(["fuel"]), ctx)
    assert result.score == 1.0
    assert result.keyword_hit is True
    assert result.n_findings == 1


def test_score_item_miss_no_findings():
    ctx = {"findings": [], "recommendations": [], "direct_regs": [], "acs": []}
    result = _score_item(_item(["fuel"]), ctx)
    assert result.score == 0.0


def test_score_item_miss_wrong_keywords():
    ctx = {"findings": [{"text": "Weather was clear."}], "recommendations": [],
           "direct_regs": [], "acs": []}
    result = _score_item(_item(["fuel", "exhaust"]), ctx)
    assert result.score == 0.0
    assert result.n_findings == 1
    assert result.keyword_hit is False


def test_score_item_counts_regs_from_findings():
    ctx = {
        "findings": [{"text": "Pilot violated regs.", "cites_reg": "602.115"}],
        "recommendations": [], "direct_regs": ["602.88"], "acs": [],
    }
    result = _score_item(_item([]), ctx)
    assert result.n_regs == 2  # 602.115 from finding + 602.88 direct


# ─── graph_evaluate ──────────────────────────────────────────────────────────

def _ds(n: int = 2) -> list[GraphEvalItem]:
    return [GraphEvalItem(
        id=f"g{i:02d}", query=f"q{i}", occ_id=f"a{i:02d}",
        expect_regs=[], expect_finding_keywords=["keyword"],
        lang="en", note="",
    ) for i in range(n)]


def test_graph_evaluate_all_hits():
    def runner(occ_id: str) -> dict:
        return {"findings": [{"text": "keyword match here."}],
                "recommendations": [], "direct_regs": [], "acs": []}
    report = graph_evaluate(runner, _ds(3))
    assert report["traversal_hit"] == pytest.approx(1.0)
    assert report["n"] == 3


def test_graph_evaluate_all_misses():
    def runner(occ_id: str) -> dict:
        return {"findings": [], "recommendations": [], "direct_regs": [], "acs": []}
    report = graph_evaluate(runner, _ds(3))
    assert report["traversal_hit"] == pytest.approx(0.0)


def test_graph_evaluate_partial():
    calls = iter([
        {"findings": [{"text": "keyword here."}], "recommendations": [], "direct_regs": [], "acs": []},
        {"findings": [], "recommendations": [], "direct_regs": [], "acs": []},
    ])
    def runner(occ_id: str) -> dict:
        return next(calls)
    report = graph_evaluate(runner, _ds(2))
    assert report["traversal_hit"] == pytest.approx(0.5)
    assert len(report["items"]) == 2


# ─── load_graph_dataset ──────────────────────────────────────────────────────

def test_load_graph_dataset(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({
        "id": "g01", "query": "fuel", "occ_id": "a13q0098",
        "expect_regs": [], "expect_finding_keywords": ["fuel"],
        "lang": "en", "note": "test",
    }) + "\n", encoding="utf-8")
    items = load_graph_dataset(p)
    assert len(items) == 1
    assert items[0].occ_id == "a13q0098"
    assert items[0].expect_finding_keywords == ["fuel"]


def test_load_graph_dataset_skips_comments(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        "# this is a comment\n" +
        json.dumps({"id": "g01", "query": "q", "occ_id": "a01",
                    "expect_regs": [], "expect_finding_keywords": [],
                    "lang": "en", "note": ""}) + "\n",
        encoding="utf-8",
    )
    items = load_graph_dataset(p)
    assert len(items) == 1


def test_load_graph_dataset_raises_on_empty(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_graph_dataset(p)
