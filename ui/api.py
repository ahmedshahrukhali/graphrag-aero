"""HTTP client for the GraphRAG Aero backend."""
from __future__ import annotations

import os
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080").rstrip("/")
_TIMEOUT_HEALTH = 5
_TIMEOUT_RETRIEVE = 30
_TIMEOUT_QUERY = 180   # gemma2:9b can take a while
_TIMEOUT_RESUME = 60


def health() -> dict:
    r = requests.get(f"{BACKEND_URL}/healthz", timeout=_TIMEOUT_HEALTH)
    r.raise_for_status()
    return r.json()


def retrieve(query: str, *, lang: str | None = None,
             source: str | None = None, top_k: int = 10) -> dict:
    payload: dict = {"query": query, "top_k": top_k}
    if lang:
        payload["lang"] = lang
    if source:
        payload["source"] = source
    r = requests.post(f"{BACKEND_URL}/retrieve", json=payload,
                      timeout=_TIMEOUT_RETRIEVE)
    r.raise_for_status()
    return r.json()


def query(text: str, thread_id: str, *,
          max_hops: int = 2) -> dict:
    """POST /query — runs agent to HITL pause, returns draft + trace."""
    r = requests.post(f"{BACKEND_URL}/query",
                      json={"query": text, "thread_id": thread_id,
                            "max_hops": max_hops},
                      timeout=_TIMEOUT_QUERY)
    r.raise_for_status()
    return r.json()


def graph_query(doc_id: str) -> dict:
    """GET /graph/{doc_id} — knowledge-graph context for one occurrence."""
    r = requests.get(f"{BACKEND_URL}/graph/{doc_id}", timeout=_TIMEOUT_RETRIEVE)
    r.raise_for_status()
    return r.json()


def resume(thread_id: str, draft: str | None = None) -> dict:
    """POST /resume/{thread_id} — finalise with optional edited draft."""
    payload: dict = {}
    if draft is not None:
        payload["draft"] = draft
    r = requests.post(f"{BACKEND_URL}/resume/{thread_id}", json=payload,
                      timeout=_TIMEOUT_RESUME)
    r.raise_for_status()
    return r.json()
