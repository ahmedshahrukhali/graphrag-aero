# Deployment

How to run the stack — locally with Docker Compose, and as a
HuggingFace Space — plus the full env-var catalogue and known gotchas.

## Local stack (Docker Compose)

### Prerequisites

- **Docker Desktop** (Windows/macOS) or Docker Engine + Compose v2 (Linux) — must be
  **running** before any `docker compose` command. On Windows, launch Docker Desktop from
  the Start Menu and wait for the system-tray icon to show "Docker Desktop is running."
- 16 GB RAM (Neo4j + Postgres + Ollama + Qdrant + backend)
- ~10 GB free disk (model weights + Qdrant index + Neo4j data)
- **NVIDIA GPU with 8 GB VRAM** recommended — Ollama runs gemma2:9b on
  GPU when available, on CPU as a slow fallback. Uncomment the
  `deploy.resources` block under `ollama` in `docker-compose.yml`
  if you have one.

### One-time setup

```bash
# (Windows) Start Docker Desktop from the Start Menu; wait for "Docker Desktop is running"
# (macOS)   open -a Docker && sleep 10
# (Linux)   sudo systemctl start docker   # or equivalent for your init system

cp .env.example .env
# edit: POSTGRES_PASSWORD, NEO4J_PASSWORD (anything but the default)

docker compose up -d qdrant neo4j postgres ollama otel-collector
docker compose exec ollama ollama pull gemma2:9b
```

### Build the index

Ingestion + embedding + graph upsert are **one-shot jobs** wrapped in
compose profiles so they don't auto-start with the long-running infra:

```bash
# acquire (one-time corpus pull from public TSB + TC endpoints)
docker compose --profile ingest run --rm ingestion \
    python -m ingestion.acquisition.run --source tsb
docker compose --profile ingest run --rm ingestion \
    python -m ingestion.acquisition.run --source tc

# chunk
docker compose --profile ingest run --rm ingestion \
    python -m ingestion.processing.run

# embed → Qdrant
docker compose --profile embed run --rm embed python -m embed.run

# build the Neo4j graph from chunks
docker compose --profile agent run --rm agent \
    python -m agent.run init-schema
docker compose --profile agent run --rm agent \
    python -m agent.run upsert-graph
```

### Run the app

```bash
docker compose up --build backend frontend
# → backend  http://localhost:8080  (OpenAPI docs at /docs)
# → frontend http://localhost:3000
```

### Quick smoke test (no UI)

```bash
curl -s -X POST http://localhost:8080/retrieve \
  -H 'content-type: application/json' \
  -d '{"query":"fuel exhaustion forced landing","top_k":3}' | jq .
```

## HuggingFace Space (Gradio)

The Space is a **shell** — it loads no models. Every model call goes
over HTTP to a reachable backend. This is the only way to fit a real
GraphRAG demo on the free HF Spaces tier.

### Push the Space

```bash
# the hf_space/README.md has the YAML frontmatter HF needs
cd hf_space/
huggingface-cli login
huggingface-cli upload <your-username>/graphrag-aero . . --repo-type=space
```

### Configure

In the Space's **Settings → Variables and secrets**, set:

- `BACKEND_URL` (secret): the publicly-reachable URL of your FastAPI
  backend. You'll need to deploy the backend somewhere — your own
  server, an HF Inference Endpoint, a Modal app, etc.

The Space's `Dockerfile` exposes port 7860 (the HF default); the YAML
frontmatter in `hf_space/README.md` uses `sdk: docker` so HF builds it
verbatim.

### Local Space testing

```bash
docker compose --profile hf-space up --build hf-space
# → http://localhost:7860, pointed at http://backend:8080
```

## Environment variables

All variables live in `.env` (top-level, gitignored). Defaults below
match `.env.example`.

| Variable | Used by | Default | Notes |
|----------|---------|---------|-------|
| `QDRANT_HOST` | embed, retrieve, backend | `qdrant` | container hostname inside compose; `localhost` outside |
| `QDRANT_PORT` | embed, retrieve, backend | `6333` | |
| `QDRANT_COLLECTION_DENSE` | embed, retrieve | `aerospace_dense` | |
| `NEO4J_URI` | graph, agent, backend | `bolt://neo4j:7687` | |
| `NEO4J_USER` | graph, agent, backend | `neo4j` | |
| `NEO4J_PASSWORD` | graph, agent, backend | — | **must change** from `please_change_me` |
| `POSTGRES_PASSWORD` | postgres, agent | — | **must change**; used in `POSTGRES_DSN` |
| `POSTGRES_DSN` | agent, backend | `postgresql://langgraph:.../langgraph` | LangGraph checkpointer |
| `OLLAMA_HOST` | agent, backend | `http://ollama:11434` | |
| `OLLAMA_MODEL` | agent, backend | `gemma2:9b` | Q4_K_M is the locked quant |
| `EMBED_MODEL` | embed, retrieve | `BAAI/bge-m3` | locked |
| `RERANK_MODEL` | retrieve | `BAAI/bge-reranker-v2-m3` | locked |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | backend | `http://otel-collector:4317` | gRPC; backend skips OTel if unset |
| `OTEL_SERVICE_NAME` | backend | `graphrag-aero` | |
| `BACKEND_PORT` | backend | `8080` | |
| `NEXT_PUBLIC_BACKEND_URL` | frontend (build-time) | `http://localhost:8080` | **inlined at `next build`** — rebuild image to change |
| `BACKEND_URL` | hf_space | `http://localhost:8080` | read at runtime |

## Gotchas

- **Docker Desktop must be running first** (Windows/macOS). Every `docker compose` command
  fails silently or with a confusing socket error if Docker Desktop is stopped. Launch it
  from the Start Menu (Windows) or Applications (macOS), wait for the tray icon to show
  "Docker Desktop is running," then proceed. This is the most common cause of unexplained
  stack failures — confirmed recurring across sessions S19, S22, S23.
- **`NEXT_PUBLIC_BACKEND_URL` is baked in at build time.** If the
  backend URL changes, you must rebuild the frontend image. The
  `frontend/Dockerfile` accepts it as a `--build-arg`; the compose
  service passes it through automatically.
- **Single-worker uvicorn is mandatory.** The
  `backend/Dockerfile` hardcodes `--workers 1`. Multiple workers would
  race for the GPU's 8 GB and intermittently fail to load Ollama. Do
  not change this casually.
- **Pydantic v2 is required.** This bit us mid-development: `fastapi
  <0.100` pins pydantic v1 and breaks with `langchain-core` which
  requires v2. `backend/requirements.txt` pins `fastapi >= 0.110`.
- **The TSB / TC corpus is large.** Full acquisition is ~1300 PDFs;
  budget ~2 hours for a polite, rate-limited pull. P1b is resumable —
  re-running the acquisition script skips files already on disk.
- **HuggingFace weight downloads on first run.** BGE-M3 and the
  reranker are pulled by `FlagEmbedding` the first time `embed.run` or
  `retrieve.run` execute. Pre-cache by building the `embed` /
  `retrieve` / `backend` images, which run a tiny warm-up `python -c`
  during `docker build` to populate `HF_HOME`.
- **OTel collector is optional.** If you don't run `otel-collector`,
  remove the `OTEL_EXPORTER_OTLP_ENDPOINT` env var on the backend (or
  leave it unset); the backend skips OTel installation and the routes
  fall through to a noop tracer.

## Resetting

To wipe everything and start clean:

```bash
docker compose down -v        # removes named volumes (qdrant_data, neo4j_data, postgres_data, ollama_models)
rm -rf data/chunks/            # cached chunked JSONL
# (data/corpus/ is the source PDFs — leave it if you want to skip re-acquiring)

# (Windows/macOS) Once all containers are stopped you can quit Docker Desktop
#   from the system-tray icon → "Quit Docker Desktop"
```
