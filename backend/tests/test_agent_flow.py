"""/query → /resume flow over stub agent deps (§3: HITL removed; /query completes immediately)."""
from __future__ import annotations

import pytest

pytest.importorskip("langgraph")


def test_query_completes_with_draft(make_client, stub_deps_with_checkpointer):
    """§3: /query runs to END in one call — draft is returned immediately."""
    client = make_client()
    r = client.post("/query", json={"query": "fuel", "thread_id": "t1", "max_hops": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["thread_id"] == "t1"
    assert body["draft"] == "stubbed draft answer."
    assert body["n_candidates"] > 0
    nodes = [t["node"] for t in body["trace"]]
    assert "retrieve" in nodes and "synthesize" in nodes


def test_resume_returns_final_answer(make_client, stub_deps_with_checkpointer):
    """§3: /resume on an already-completed thread returns the final answer."""
    client = make_client()
    client.post("/query", json={"query": "fuel", "thread_id": "t2", "max_hops": 1})
    r = client.post("/resume/t2", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["final"] == "stubbed draft answer."


def test_synthesize_unloads_retrieval_models(make_client, stub_deps_with_checkpointer):
    """VRAM discipline assertion: by the time finalize runs, embedder +
    reranker have been unloaded so the LLM gets the slot."""
    client = make_client()
    client.post("/query", json={"query": "fuel", "thread_id": "t4", "max_hops": 1})
    assert stub_deps_with_checkpointer._stubs["embedder"].unloaded is True
    assert stub_deps_with_checkpointer._stubs["reranker"].unloaded is True


def test_reject_stores_rejection(make_client, stub_deps_with_checkpointer):
    """POST /reject/{thread_id} persists a rejection after a completed query."""
    client = make_client()
    client.post("/query", json={"query": "fuel", "thread_id": "rej1", "max_hops": 1})
    r = client.post("/reject/rej1", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["rejection_id"], int)
    fb = stub_deps_with_checkpointer._stubs["feedback_store"]
    assert len(fb.rejections) == 1
    rec = fb.rejections[0]
    assert rec["query"] == "fuel"
    assert rec["answer"] == "stubbed draft answer."
    # Only top-3 candidates excluded by default, not the full candidate list.
    assert isinstance(rec["chunk_hashes"], list)
    assert len(rec["chunk_hashes"]) <= 3


def test_reject_unknown_thread_returns_404(make_client, stub_deps_with_checkpointer):
    """POST /reject for a thread that was never queried returns 404."""
    client = make_client()
    r = client.post("/reject/no-such-thread", json={})
    assert r.status_code == 404, r.text


def test_resolve_clears_rejection(make_client, stub_deps_with_checkpointer):
    """POST /resolve/{id} calls feedback_store.resolve with the right id."""
    client = make_client()
    client.post("/query", json={"query": "fuel", "thread_id": "res1", "max_hops": 1})
    rej = client.post("/reject/res1", json={}).json()
    rid = rej["rejection_id"]

    r = client.post(f"/resolve/{rid}")
    assert r.status_code == 200, r.text
    assert r.json()["rejection_id"] == rid

    fb = stub_deps_with_checkpointer._stubs["feedback_store"]
    assert rid in fb.resolved_ids


def _sse_events(text: str) -> list[str]:
    return [ln[len("event:"):].strip() for ln in text.splitlines() if ln.startswith("event:")]


def _sources_payload(text: str) -> list[dict]:
    """Pull the `sources` event's source list out of an SSE response body."""
    import json
    for block in text.split("\n\n"):
        if "event: sources" in block:
            data_line = next(ln for ln in block.splitlines() if ln.startswith("data:"))
            return json.loads(data_line[len("data:"):].strip()).get("sources", [])
    return []


def test_query_stream_applies_source_filter(make_client):
    """P5: the Corpus filter must reach retrieval. The stub corpus is all
    `tsb/*`, so source='tc' yields no chunks while source='tsb' yields some —
    proving lang/source plumb schema → route → retrieve node → pipeline."""
    client = make_client()

    r_tc = client.post("/query/stream", json={
        "query": "fuel", "thread_id": "tcf", "max_hops": 1, "source": "tc",
    })
    assert r_tc.status_code == 200, r_tc.text
    assert _sources_payload(r_tc.text) == [], "tc filter should exclude the tsb-only corpus"

    r_tsb = client.post("/query/stream", json={
        "query": "fuel", "thread_id": "tsbf", "max_hops": 1, "source": "tsb",
    })
    assert r_tsb.status_code == 200, r_tsb.text
    assert _sources_payload(r_tsb.text), "tsb filter should keep the tsb corpus"


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
