import pytest
from pathlib import Path
from hf_space.graph_local import GraphArtifacts
from hf_space.api_client import ApiError


def _rich_row(occ_id: str, findings=None, recs=None, regs=None, acs=None) -> dict:
    return {
        "occ_id": occ_id,
        "occ_url": f"https://bst-tsb.gc.ca/{occ_id}",
        "findings": findings or [],
        "recommendations": recs or [],
        "direct_regs": regs or [],
        "acs": acs or [],
        "rec_regs": [],
        "reg_guided_acs": []
    }


def test_graph_context_for_known_ids():
    finding = {"text": "Fuel tanks empty.", "category": "cause", "lang": "en",
                "source_doc_id": "tsb/a01", "page": 5, "cites_reg": "602.115"}
    graph_context = {"a01": _rich_row("a01", findings=[finding])}
    artifacts = GraphArtifacts(graph_context, {})
    
    out = artifacts.graph_context(["a01"])
    assert len(out) == 1
    assert out[0]["occ_id"] == "a01"
    assert len(out[0]["findings"]) == 1
    assert out[0]["findings"][0]["cites_reg"] == "602.115"


def test_graph_context_drops_unknown_ids():
    artifacts = GraphArtifacts({"a01": _rich_row("a01")}, {})
    out = artifacts.graph_context(["a01", "ghost"])
    assert len(out) == 1


def test_doc_lookup_success_and_fail():
    artifacts = GraphArtifacts({"a01": _rich_row("a01")}, {})
    out = artifacts.doc_lookup("tsb/a01")
    assert out["occ_id"] == "a01"
    
    with pytest.raises(ApiError) as exc:
        artifacts.doc_lookup("ghost")
    assert exc.value.status == 404


def test_recurring_builds_citable_siblings_and_caps():
    cites_edges = {
        "occ_cites": {"a01": ["703.07"], "a02": ["703.07"], "a03": ["703.07"], "a04": ["703.07"], "a05": ["703.07"]},
        "reg_occs": {"703.07": ["a01", "a02", "a03", "a04", "a05"]},
        "occ_url": {}
    }
    artifacts = GraphArtifacts({}, cites_edges)
    out = artifacts.recurring_context(["a01"], max_siblings_per_reg=3)
    assert len(out) == 1
    assert out[0]["reg"] == "703.07" and out[0]["occ_count"] == 5
    sibs = out[0]["siblings"]
    assert [s["occ_id"] for s in sibs] == ["a02", "a03", "a04"]  # capped at 3
    assert sibs[0]["source_doc_id"] == "tsb/a02"


def test_recurring_skips_regs_without_nonseed_siblings():
    cites_edges = {
        "occ_cites": {"a01": ["602.07"]},
        "reg_occs": {"602.07": ["a01"]},
        "occ_url": {}
    }
    artifacts = GraphArtifacts({}, cites_edges)
    out = artifacts.recurring_context(["a01"])
    assert out == []


def test_recurring_degree_cap():
    # 1 seed + 16 siblings = 17 occurrences for 703.07, which exceeds default max_reg_degree 15
    occs = ["a01"] + [f"x{i}" for i in range(16)]
    cites_edges = {
        "occ_cites": {occ: ["703.07"] for occ in occs},
        "reg_occs": {"703.07": occs},
        "occ_url": {}
    }
    artifacts = GraphArtifacts({}, cites_edges)
    out = artifacts.recurring_context(["a01"], max_reg_degree=15)
    assert out == []
