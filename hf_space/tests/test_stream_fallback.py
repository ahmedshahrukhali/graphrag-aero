"""Two-way engine fallback tests for ``engine_router.stream_with_fallback`` (S50).

Forward direction (S47): in-Space ZeroGPU engine → backend on quota errors.
Reverse direction (S50): backend → in-Space engine when the tunnel/backend is
down (ApiError 5xx before the first event). No gradio, no model loads, no
network — the zgpu module functions are monkeypatched and the client is a stub.
"""
from __future__ import annotations

import pytest

import hf_space.engine_router as router
from hf_space.api_client import ApiError
from hf_space.engine_router import stream_with_fallback


ZGPU_EVENTS = [
    {"event": "status", "data": {"node": "retrieve", "msg": "Retrieving relevant chunks…"}},
    {"event": "token", "data": {"text": "offline answer"}},
    {"event": "done", "data": {"thread_id": "offline-thread", "draft": "offline answer", "trace": []}},
]
BACKEND_EVENTS = [
    {"event": "status", "data": {"node": "retrieve", "msg": "Retrieving relevant chunks…"}},
    {"event": "token", "data": {"text": "backend answer"}},
    {"event": "done", "data": {"thread_id": "t1", "draft": "backend answer", "trace": []}},
]


class StubClient:
    """query_stream stub: yields ``events`` then raises ``exc`` (if any)."""

    def __init__(self, events=(), exc: Exception | None = None):
        self.events = list(events)
        self.exc = exc
        self.calls = 0

    def query_stream(self, q, thread_id, *, max_hops, lang, source, history):
        self.calls += 1
        yield from self.events
        if self.exc is not None:
            raise self.exc


def _zgpu(monkeypatch, *, available: bool, events=ZGPU_EVENTS, exc: Exception | None = None):
    """Patch the zgpu module surface; returns the call-counter dict."""
    calls = {"n": 0}

    def answer_stream(q, lang=None, source=None, history=None):
        calls["n"] += 1
        yield from events
        if exc is not None:
            raise exc

    monkeypatch.setattr(router.zgpu, "available", lambda: available)
    monkeypatch.setattr(router.zgpu, "answer_stream", answer_stream)
    return calls


def _run(client, *, use_zgpu: bool):
    return list(stream_with_fallback(
        client, q="fuel exhaustion", thread_id="t1", max_hops=2,
        lang=None, source=None, history=None, use_zgpu=use_zgpu,
    ))


# ── happy paths: each engine alone ───────────────────────────────────────────

def test_toggle_on_uses_zgpu_only(monkeypatch):
    zcalls = _zgpu(monkeypatch, available=True)
    client = StubClient(BACKEND_EVENTS)
    assert _run(client, use_zgpu=True) == ZGPU_EVENTS
    assert zcalls["n"] == 1 and client.calls == 0


def test_toggle_off_uses_backend_only(monkeypatch):
    zcalls = _zgpu(monkeypatch, available=True)
    client = StubClient(BACKEND_EVENTS)
    assert _run(client, use_zgpu=False) == BACKEND_EVENTS
    assert zcalls["n"] == 0 and client.calls == 1


def test_toggle_on_but_unavailable_uses_backend(monkeypatch):
    zcalls = _zgpu(monkeypatch, available=False)
    client = StubClient(BACKEND_EVENTS)
    assert _run(client, use_zgpu=True) == BACKEND_EVENTS
    assert zcalls["n"] == 0 and client.calls == 1


# ── forward fallback (S47): zgpu quota → backend ─────────────────────────────

def test_quota_error_falls_back_to_backend(monkeypatch):
    _zgpu(monkeypatch, available=True, events=[], exc=Exception("GPU quota exceeded"))
    client = StubClient(BACKEND_EVENTS)
    out = _run(client, use_zgpu=True)
    assert out[0]["data"]["node"] == "fallback"
    assert out[1:] == BACKEND_EVENTS


def test_non_quota_zgpu_error_raises(monkeypatch):
    _zgpu(monkeypatch, available=True, events=[], exc=RuntimeError("CUDA OOM"))
    client = StubClient(BACKEND_EVENTS)
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        _run(client, use_zgpu=True)
    assert client.calls == 0


# ── reverse fallback (S50): backend down → zgpu ──────────────────────────────

def test_backend_down_falls_back_to_zgpu(monkeypatch):
    zcalls = _zgpu(monkeypatch, available=True)
    client = StubClient([], ApiError(503, "POST /query/stream → backend unreachable"))
    out = _run(client, use_zgpu=False)
    assert out[0]["data"]["node"] == "fallback"
    assert "in-Space" in out[0]["data"]["msg"]
    assert out[1:] == ZGPU_EVENTS
    assert zcalls["n"] == 1


def test_backend_timeout_falls_back_to_zgpu(monkeypatch):
    _zgpu(monkeypatch, available=True)
    client = StubClient([], ApiError(504, "POST /query/stream → backend timed out"))
    out = _run(client, use_zgpu=False)
    assert out[-1]["event"] == "done"


def test_backend_down_no_zgpu_reraises(monkeypatch):
    _zgpu(monkeypatch, available=False)
    client = StubClient([], ApiError(503, "backend unreachable"))
    with pytest.raises(ApiError) as exc_info:
        _run(client, use_zgpu=False)
    assert exc_info.value.status == 503


def test_backend_4xx_never_falls_back(monkeypatch):
    zcalls = _zgpu(monkeypatch, available=True)
    client = StubClient([], ApiError(422, "validation error"))
    with pytest.raises(ApiError) as exc_info:
        _run(client, use_zgpu=False)
    assert exc_info.value.status == 422
    assert zcalls["n"] == 0


def test_midstream_drop_never_falls_back(monkeypatch):
    """Fallback after partial backend output would replay events into the
    transcript — a mid-stream death must surface, not restart."""
    zcalls = _zgpu(monkeypatch, available=True)
    client = StubClient(BACKEND_EVENTS[:1], ApiError(503, "tunnel dropped"))
    with pytest.raises(ApiError):
        _run(client, use_zgpu=False)
    assert zcalls["n"] == 0


# ── dead end: quota burned AND backend down ──────────────────────────────────

def test_quota_then_backend_down_is_friendly_dead_end(monkeypatch):
    zcalls = _zgpu(monkeypatch, available=True, events=[], exc=Exception("GPU quota exceeded"))
    client = StubClient([], ApiError(503, "backend unreachable"))
    with pytest.raises(ApiError) as exc_info:
        _run(client, use_zgpu=True)
    # No second zgpu attempt (quota is spent), and the message a visitor sees
    # via _fmt_error explains both halves.
    assert zcalls["n"] == 1
    assert "quota" in exc_info.value.detail["detail"].lower()
    assert "backend" in exc_info.value.detail["detail"].lower()
