"""OpenTelemetry setup.

Production: OTLP gRPC exporter pointed at the ``otel-collector`` sidecar (port
4317). Tests: caller passes in an in-memory exporter so spans are inspectable
without a network.

``setup_tracing(app, exporter=...)`` is idempotent across module reloads — it
checks the global tracer provider before installing one.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

SERVICE_NAME = "graphrag-aero-backend"


def setup_tracing(app: Any, *, exporter: Any | None = None) -> Any:
    """Install a tracer provider + FastAPI instrumentation. Returns the tracer.

    Pass ``exporter`` to use a custom (e.g. ``InMemorySpanExporter`` in tests)
    span exporter; otherwise an OTLP gRPC exporter is built from
    ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default ``localhost:4317``).
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))

    if exporter is None:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        # Tests want synchronous flush.
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    return trace.get_tracer(SERVICE_NAME)


class _NoopSpan:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def set_attribute(self, *a, **kw): pass


class _NoopTracer:
    def start_as_current_span(self, *_a, **_kw): return _NoopSpan()


def get_tracer() -> Any:
    """Return the global tracer, or a noop if opentelemetry isn't installed.

    Routes always call ``get_tracer().start_as_current_span(...)``; the noop
    keeps that working in environments where OTel is an optional extra (tests
    without the OTel deps installed, ad-hoc local runs without a collector).
    """
    try:
        from opentelemetry import trace
    except ModuleNotFoundError:
        return _NoopTracer()
    return trace.get_tracer(SERVICE_NAME)
