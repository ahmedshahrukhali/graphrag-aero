"""/retrieve endpoint coverage."""
from __future__ import annotations


def test_retrieve_returns_ranked_chunks(make_client):
    client = make_client()
    r = client.post("/retrieve", json={"query": "fuel", "top_k": 3, "ann_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == "fuel"
    assert len(body["results"]) == 3
    # "fuel" query embeds to axis 0 → doc-A first.
    assert body["results"][0]["doc_id"] == "tsb/doc-A"
    assert body["results"][0]["rank"] == 1


def test_retrieve_lang_filter(make_client):
    client = make_client()
    r = client.post("/retrieve", json={
        "query": "carburant", "lang": "fr", "top_k": 3, "ann_k": 3,
    })
    assert r.status_code == 200
    docs = [c["doc_id"] for c in r.json()["results"]]
    # Only the FR doc should survive the payload filter.
    assert docs == ["tsb/doc-C"]


def test_retrieve_rejects_ann_k_less_than_top_k(make_client):
    client = make_client()
    r = client.post("/retrieve", json={"query": "fuel", "top_k": 5, "ann_k": 3})
    assert r.status_code == 400


def test_retrieve_rejects_empty_query(make_client):
    client = make_client()
    r = client.post("/retrieve", json={"query": ""})
    assert r.status_code == 422  # pydantic validation
