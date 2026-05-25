"""/query → /resume HITL flow over stub agent deps."""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph")


def test_query_pauses_at_hitl_with_draft(make_client, stub_deps_with_checkpointer):
    client = make_client()
    r = client.post("/query", json={"query": "fuel", "thread_id": "t1", "max_hops": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["thread_id"] == "t1"
    assert body["draft"] == "stubbed draft answer."
    assert body["n_candidates"] > 0
    nodes = [t["node"] for t in body["trace"]]
    assert "retrieve" in nodes and "synthesize" in nodes


def test_resume_without_edit_finalizes_model_draft(make_client, stub_deps_with_checkpointer):
    client = make_client()
    client.post("/query", json={"query": "fuel", "thread_id": "t2", "max_hops": 1})
    r = client.post("/resume/t2", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["final"] == "stubbed draft answer."
    assert body["history"], "history should be populated post-finalize"


def test_resume_with_edited_draft_overrides_final(make_client, stub_deps_with_checkpointer):
    client = make_client()
    client.post("/query", json={"query": "fuel", "thread_id": "t3", "max_hops": 1})
    r = client.post("/resume/t3", json={"draft": "human-edited final answer"})
    assert r.status_code == 200, r.text
    assert r.json()["final"] == "human-edited final answer"


def test_synthesize_unloads_retrieval_models(make_client, stub_deps_with_checkpointer):
    """VRAM discipline assertion: by the time finalize runs, embedder +
    reranker have been unloaded so the LLM gets the slot."""
    client = make_client()
    client.post("/query", json={"query": "fuel", "thread_id": "t4", "max_hops": 1})
    assert stub_deps_with_checkpointer._stubs["embedder"].unloaded is True
    assert stub_deps_with_checkpointer._stubs["reranker"].unloaded is True
