import pytest
from unittest.mock import patch, MagicMock

from hf_space.zerogpu_engine import available, is_quota_error, answer_stream
from retrieve.reranker import ScoredChunk


def test_is_quota_error():
    assert is_quota_error(Exception("gpu quota limit reached"))
    assert not is_quota_error(Exception("connection refused"))
    
    class QuotaExceededError(Exception):
        pass
    assert is_quota_error(QuotaExceededError("random text"))


def test_available_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("hf_space.zerogpu_engine.SPACE_INDEX_DIR", tmp_path)
    assert not available()


def test_available_true(monkeypatch, tmp_path):
    monkeypatch.setattr("hf_space.zerogpu_engine.SPACE_INDEX_DIR", tmp_path)
    (tmp_path / "qdrant_local").mkdir()
    (tmp_path / "graph_context.json").write_text("{}")
    assert available()


@patch("hf_space.zerogpu_engine._load_models")
@patch("retrieve.pipeline.anchored_retrieve")
def test_answer_stream_parity(mock_retrieve, mock_load, monkeypatch):
    class DummyRecord:
        doc_id = "tsb/x"
        source_url = None
        section_title = "Title"
        page = 1
        bbox = None
        lang = "en"
        text = "sample text"
        
    chunk = MagicMock(spec=ScoredChunk)
    chunk.score = 0.95
    chunk.record = DummyRecord()
    
    mock_retrieve.return_value = [chunk]
    
    # Mock graph artifacts
    class MockGraph:
        def graph_context(self, ids): return [{"occ_id": "x"}]
        def recurring_context(self, ids): return [{"reg": "123", "occ_count": 2}]
    
    monkeypatch.setattr("hf_space.zerogpu_engine._graph_artifacts", MockGraph())
    
    # Mock threading and streamer by patching transformers directly (if it exists)
    class MockStreamer:
        def __init__(self, *args, **kwargs):
            pass
        def __iter__(self):
            yield "Hello"
            yield " World"
            
    monkeypatch.setattr("hf_space.zerogpu_engine._llm_pipeline", MagicMock())
    
    import sys
    if "transformers" not in sys.modules:
        sys.modules["transformers"] = MagicMock()
        
    monkeypatch.setattr("transformers.TextIteratorStreamer", MockStreamer)
    monkeypatch.setattr("threading.Thread", MagicMock)
    
    events = list(answer_stream("query"))
    
    # Check shape of emitted events vs backend SSE
    status_events = [e for e in events if e["event"] == "status"]
    assert len(status_events) == 5
    assert status_events[0]["data"]["node"] == "retrieve"
    assert status_events[1]["data"]["node"] == "retrieve"
    assert status_events[2]["data"]["node"] == "graph_expand"
    assert status_events[3]["data"]["node"] == "graph_expand"
    assert status_events[4]["data"]["node"] == "synthesize"
    
    sources_events = [e for e in events if e["event"] == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["data"]["sources"][0]["doc_id"] == "tsb/x"
    
    token_events = [e for e in events if e["event"] == "token"]
    assert any(t["data"]["text"] == "Hello" for t in token_events)
    # the format_sources_block tail is emitted as a token event too
    assert len(token_events) == 3
    
    done_events = [e for e in events if e["event"] == "done"]
    assert len(done_events) == 1
    assert "Hello World" in done_events[0]["data"]["draft"]
    assert "tsb/x" in done_events[0]["data"]["draft"]  # the source block is appended
