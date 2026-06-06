# backend/ — P6

FastAPI HTTP surface for the retrieve + agent pipeline. Wires Ollama
(qwen3:4b) as the LLM and ships OpenTelemetry spans to the collector
sidecar.

## Endpoints
- `POST /retrieve` — `{query, lang?, source?, ann_k?, top_k?}` → ranked
  chunks (dense ANN + cross-encoder rerank). Thin wrapper over
  `retrieve.pipeline.retrieve_and_rerank`.
- `POST /query` — `{query, thread_id, max_hops?}` → kicks off the LangGraph
  agent, **runs until the HITL interrupt**, returns `{thread_id, draft,
  trace, n_candidates}`. The caller decides what to do with the draft.
- `POST /resume/{thread_id}` — body `{draft?}` (optional human edit) →
  finalises past the interrupt. Returns `{final, trace, history}`.
- `GET /healthz` — pings Qdrant + Neo4j + Ollama. Never loads models.

`POST /ingest` is intentionally out of scope — ingestion runs in its own
image (`docker compose --profile ingest run ingestion`).

## Run

```bash
docker compose up -d qdrant neo4j postgres ollama otel-collector
docker compose up --build backend
# → http://localhost:8080/docs
```

Local dev (skip Docker):

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8080 --workers 1
```

## VRAM discipline
**Single worker only.** Multiple uvicorn workers would race for the
3060Ti's 8GB VRAM. The current model budget (BGE-M3 0.5GB → reranker 0.5GB
→ qwen3:4b ~2.5GB, sequenced) only fits if exactly one request is using
GPU at a time. `--workers 1` is enforced in the Dockerfile CMD.

Within a request, the sequencing is:
1. `embedder.embed(query)` — loads BGE-M3 lazily on first call.
2. `reranker.score(...)` — loads cross-encoder lazily.
3. `synthesize` node calls `embedder.unload()` and `reranker.unload()`
   before invoking Ollama for generation.

## OpenTelemetry
[otel.py](otel.py) installs a `TracerProvider` + `FastAPIInstrumentor`.
The production exporter is OTLP gRPC pointed at
`$OTEL_EXPORTER_OTLP_ENDPOINT` (default `localhost:4317`). Tests inject an
`InMemorySpanExporter` so spans are inspectable without a network.

Manual spans wrap the retrieve, agent.query, and agent.resume code paths
so the LangGraph audit trail and the OTel trace can be cross-referenced.

## Tests
```bash
pytest backend/tests/
```
TestClient + stub deps + in-memory Qdrant. No real Qdrant / Neo4j /
Ollama / model weights touched.
