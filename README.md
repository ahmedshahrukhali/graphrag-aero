# Graph RAG over Aerospace Documents

End-to-end Graph RAG over **Transport Canada Advisory Circulars** and
**Transportation Safety Board (TSB) aviation investigation reports**, in
**English and French**. PDF ingestion → dense retrieval + cross-encoder
rerank → Neo4j knowledge graph → LangGraph multi-hop agent with a
Human-in-the-Loop gate → FastAPI backend with OpenTelemetry tracing →
Next.js UI with PDF citation highlighting → Gradio HuggingFace Space.

All inference is local: BGE-M3 dense embeddings + `bge-reranker-v2-m3`
cross-encoder + `gemma2:9b` via Ollama. The model budget is sequenced
for an 8GB GPU (3060Ti); see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

The project is built phase-by-phase per [MANIFEST.md](MANIFEST.md). All
nine phases land here; see the manifest for the per-phase narrative and
current test counts.

- **221 tests** offline (Python + TypeScript), no model weights or
  network calls required.
- Each phase ships a self-contained module with its own README and
  Dockerfile.

## Quickstart

```bash
# 1. configure secrets and ports
cp .env.example .env                # edit POSTGRES_PASSWORD, NEO4J_PASSWORD

# 2. start backing services
docker compose up -d qdrant neo4j postgres ollama otel-collector

# 3. pull the local LLM (one-time, ~5.5GB)
docker compose exec ollama ollama pull gemma2:9b

# 4. acquire + ingest the corpus
docker compose --profile ingest run --rm ingestion \
    python -m ingestion.acquisition.run --source tsb
docker compose --profile ingest run --rm ingestion \
    python -m ingestion.acquisition.run --source tc
docker compose --profile ingest run --rm ingestion \
    python -m ingestion.processing.run

# 5. embed → Qdrant
docker compose --profile embed run --rm embed python -m embed.run

# 6. build the graph
docker compose --profile agent run --rm agent python -m agent.run init-schema
docker compose --profile agent run --rm agent python -m agent.run upsert-graph

# 7. bring up the app
docker compose up --build backend frontend
# → backend  http://localhost:8080  (also /docs for OpenAPI)
# → frontend http://localhost:3000
```

The full deployment guide — including the Hugging Face Space — is in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Layout

| Path | Phase | What's there |
|------|-------|--------------|
| [ingestion/](ingestion/) | P1, P1b | TSB + TC scrapers; pdfplumber + PaddleOCR fallback; 512-tok chunker; SHA-256 dedup |
| [embed/](embed/) | P2 | BGE-M3 dense embedder → Qdrant collection `aerospace_dense` |
| [retrieve/](retrieve/) | P3 | Dense ANN search + `bge-reranker-v2-m3` cross-encoder; sequential VRAM via `ModelSession` |
| [graph/](graph/) | P4 | Neo4j schema + Cypher; Occurrence → Aircraft → Finding → Recommendation → Regulation → AC |
| [agent/](agent/) | P4 | LangGraph multi-hop agent; PostgresSaver; HITL `interrupt_before("finalize")` |
| [eval/](eval/) | P5 | Recall@k / MRR / nDCG@k over a curated JSONL dataset |
| [backend/](backend/) | P6 | FastAPI app; `/retrieve`, `/query`, `/resume`, `/healthz`; OpenTelemetry |
| [frontend/](frontend/) | P7 | Next.js 14 + TS; `react-pdf` bbox overlay; HITL draft editor |
| [hf_space/](hf_space/) | P8 | Gradio shell over the backend; server-side PDF rendering |
| [docs/](docs/) | P9 | [Architecture](docs/ARCHITECTURE.md), [Deployment](docs/DEPLOYMENT.md) |
| [otel/](otel/) | — | OpenTelemetry collector config |
| [data/corpus/](data/) | — | source PDFs (gitignored) |

## Tests

```bash
python -m pytest                     # 221 backend tests (Python)
cd frontend && npm install && npm test  # 18 frontend tests (Vitest)
```

All tests run offline. Model loads are stubbed; Qdrant runs in-memory;
Neo4j and Ollama are mocked via small Protocol classes. No weight
downloads, no live HTTP, no GPU required.

## Locked architecture

| Component | Choice |
|-----------|--------|
| Vector store | Qdrant — dense vectors only |
| Embeddings | BGE-M3 dense via FlagEmbedding (multilingual) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder (multilingual) |
| Chunking | 512 tokens, 64 overlap; carries `doc_id, section_title, page, bbox` |
| Graph | Neo4j 5 |
| Agent framework | LangGraph + PostgresSaver |
| LLM | `gemma2:9b` Q4_K_M via Ollama |
| Languages | English + French |
| Tracing | OpenTelemetry (OTLP gRPC) |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow, VRAM
sequencing, and HITL walk-through.

## How this was built

This repo was built with **Claude Code** working phase-by-phase per
[CLAUDE.md](CLAUDE.md), with the resume pointer in
[MANIFEST.md](MANIFEST.md) advancing one phase at a time after human
review. Per-phase commits carry a `Model:` trailer so `git blame` shows
which model authored each line.
