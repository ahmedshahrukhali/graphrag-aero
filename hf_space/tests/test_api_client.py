"""api_client tests using httpx.MockTransport — no network."""
from __future__ import annotations

import json

import httpx
import pytest

from hf_space.api_client import ApiError, make_client


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})


def _err(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})


def test_retrieve_posts_json_and_parses_response():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["method"] = req.method
        seen["body"] = json.loads(req.content)
        return _ok({
            "query": "q",
            "results": [{
                "rank": 1, "doc_id": "tsb/x", "source_url": None, "section_title": "S",
                "page": 3, "bbox": [0.0, 1.0, 2.0, 3.0], "lang": "en", "text": "...",
                "ann_score": 0.5, "rerank_score": 0.9,
            }],
        })

    api = make_client("http://api.test", transport=httpx.MockTransport(handler))
    out = api.retrieve("q", lang="en", top_k=5)

    assert seen["method"] == "POST"
    assert seen["url"].endswith("/retrieve")
    assert seen["body"]["query"] == "q"
    assert seen["body"]["lang"] == "en"
    assert seen["body"]["top_k"] == 5
    assert len(out.results) == 1
    assert out.results[0].doc_id == "tsb/x"
    assert out.results[0].bbox == (0.0, 1.0, 2.0, 3.0)


def test_resume_url_encodes_thread_id():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content) if req.content else {}
        return _ok({"thread_id": req.url.path.rsplit("/", 1)[-1], "final": "ok", "trace": [], "history": []})

    api = make_client("http://api.test", transport=httpx.MockTransport(handler))
    api.resume("t/1", draft="edited")
    assert seen["url"].endswith("/resume/t%2F1")
    assert seen["body"] == {"draft": "edited"}


def test_resume_omits_body_when_draft_is_none():
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content) if req.content else {}
        return _ok({"thread_id": "t", "final": "ok", "trace": [], "history": []})

    api = make_client("http://api.test", transport=httpx.MockTransport(handler))
    api.resume("t")
    assert seen["body"] == {}


def test_non_2xx_raises_apierror_with_detail():
    def handler(req: httpx.Request) -> httpx.Response:
        return _err(400, {"detail": "ann_k must be >= top_k"})

    api = make_client("http://api.test", transport=httpx.MockTransport(handler))
    with pytest.raises(ApiError) as exc_info:
        api.retrieve("q")
    assert exc_info.value.status == 400
    assert exc_info.value.detail == {"detail": "ann_k must be >= top_k"}


def test_healthz_uses_get_and_parses_components():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        return _ok({
            "ok": False,
            "qdrant": {"ok": True, "detail": None},
            "neo4j":  {"ok": True, "detail": None},
            "ollama": {"ok": False, "detail": "ollama down"},
        })

    api = make_client("http://api.test", transport=httpx.MockTransport(handler))
    h = api.healthz()
    assert h.ok is False
    assert h.qdrant.ok is True
    assert h.ollama.ok is False
    assert h.ollama.detail == "ollama down"


def test_base_url_defaults_to_env(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://from-env:1234")
    api = make_client()
    assert api.base_url == "http://from-env:1234"
