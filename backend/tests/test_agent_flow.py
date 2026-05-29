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


def _sse_events(text: str) -> list[str]:
    return [ln[len("event:"):].strip() for ln in text.splitlines() if ln.startswith("event:")]


def test_query_stream_emits_sources_before_tokens(make_client):
    """The /query/stream SSE order must deliver `sources` exactly once, after
    the last retrieve status and before the first synthesize token. The `done`
    event must no longer carry a sources field (delivered via its own event)."""
    client = make_client()
    r = client.post("/query/stream", json={"query": "fuel", "thread_id": "ts1", "max_hops": 1})
    assert r.status_code == 200, r.text

    events = _sse_events(r.text)
    assert events.count("sources") == 1, f"expected one sources event, got {events}"

    sources_idx = events.index("sources")
    # sources must come before the first token
    first_token = next((i for i, e in enumerate(events) if e == "token"), len(events))
    assert sources_idx < first_token, f"sources after tokens: {events}"
    # sources must come after at least one retrieve status
    last_retrieve_before = max(
        (i for i, e in enumerate(events[:sources_idx]) if e == "status"), default=-1,
    )
    assert last_retrieve_before != -1, f"no status before sources: {events}"
    # graph_expand status comes after sources
    assert "status" in events[sources_idx + 1:], f"no status after sources: {events}"

    # done payload no longer carries sources
    import json
    done_block = [
        b for b in r.text.split("\n\n")
        if b.startswith("event: done") or "\nevent: done" in b
    ]
    assert done_block, f"no done event in stream: {r.text!r}"
    data_line = next(ln for ln in done_block[0].splitlines() if ln.startswith("data:"))
    done_data = json.loads(data_line[len("data:"):].strip())
    assert "sources" not in done_data, f"done still carries sources: {done_data}"
