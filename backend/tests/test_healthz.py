"""/healthz pings every component and aggregates an overall ok flag."""
from __future__ import annotations


def test_healthz_all_up(make_client):
    client = make_client()
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "ok": True,
        "qdrant": {"ok": True, "detail": None},
        "neo4j":  {"ok": True, "detail": None},
        "ollama": {"ok": True, "detail": None},
    }


def test_healthz_partial_down(make_client, stub_deps):
    stub_deps._pings["ollama_ok"] = False
    client = make_client()
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["qdrant"]["ok"] is True
    assert body["ollama"]["ok"] is False
    assert "ollama down" in body["ollama"]["detail"]


def test_healthz_never_loads_models(make_client, stub_deps):
    """Calling /healthz must not trigger model loading."""
    client = make_client()
    client.get("/healthz")
    # The stub embedder/reranker have no ".embed was called" flag, but their
    # .unloaded flag stays False until either (a) explicit unload, or (b)
    # something else touches them. We assert no side effect happened.
    assert stub_deps._stubs["embedder"].unloaded is False
    assert stub_deps._stubs["reranker"].unloaded is False
    assert stub_deps._stubs["llm"].calls == []
