"""AgentState must be JSON-safe (PostgresSaver requirement)."""
import json

from embed.jsonl import ChunkRecord
from retrieve.reranker import ScoredChunk

from agent.state import (
    AgentState,
    ScoredChunkDict,
    initial_state,
    scored_chunk_to_dict,
)


def _rec(text: str = "x") -> ChunkRecord:
    return ChunkRecord(
        doc_id="tsb/a01", source_url="u", section_title="s", page=3,
        bbox=[1.0, 2.0, 3.0, 4.0], chunk_hash="0" * 64, lang="en", text=text,
    )


def test_scored_chunk_to_dict_round_trips_fields():
    sc = ScoredChunk(_rec("hello"), ann_score=0.81, rerank_score=0.42)
    d = scored_chunk_to_dict(sc)
    assert d["doc_id"] == "tsb/a01"
    assert d["text"] == "hello"
    assert d["ann_score"] == 0.81
    assert d["rerank_score"] == 0.42


def test_scored_chunk_dict_is_json_safe():
    sc = ScoredChunk(_rec(), ann_score=0.5)
    d = scored_chunk_to_dict(sc)
    blob = json.dumps(d)
    assert json.loads(blob) == d


def test_initial_state_defaults():
    s = initial_state("q")
    assert s["query"] == "q"
    assert s["hop"] == 0
    assert s["max_hops"] == 2
    assert s["candidates"] == []
    assert s["draft"] is None and s["final"] is None
    assert s["trace"] == []


def test_initial_state_is_json_safe():
    s = initial_state("q", max_hops=3)
    blob = json.dumps(s)
    parsed = json.loads(blob)
    assert parsed == s


def test_state_with_candidates_is_json_safe():
    s = initial_state("q")
    s["candidates"] = [scored_chunk_to_dict(ScoredChunk(_rec("a"), ann_score=0.1))]
    s["trace"] = [{"node": "retrieve", "elapsed_ms": 12}]
    blob = json.dumps(s)
    assert json.loads(blob) == s


def test_agent_state_total_false_allows_partial_updates():
    """TypedDict total=False — a partial dict is still a valid AgentState."""
    partial: AgentState = {"query": "q"}  # type: ignore[typeddict-item]
    assert partial["query"] == "q"
