"""OTel: assert /retrieve emits the expected spans into an in-memory exporter."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("opentelemetry.instrumentation.fastapi")


def test_retrieve_emits_spans(make_client):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    client = make_client(otel_exporter=exporter)

    r = client.post("/retrieve", json={"query": "fuel", "top_k": 3, "ann_k": 3})
    assert r.status_code == 200

    names = {s.name for s in exporter.get_finished_spans()}
    # Manual span (we wrap retrieve_and_rerank) + auto FastAPI span.
    assert "retrieve" in names
    assert any("POST" in n or "/retrieve" in n for n in names), names
