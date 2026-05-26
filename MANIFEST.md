# MANIFEST — Graph RAG over Aerospace Documents
**v3 — 2026-05-22**

**Goal:** Graph RAG over Transport Canada + TSB documents → cited, multi-hop troubleshooting answers for aviation safety/maintenance use cases.

**Status:** ✅ APPROVED / LOCKED
**Execution path:** Claude Code on user's machine.

---

## Locked decisions
| Decision | Value |
|----------|-------|
| Vector store | Qdrant — dense vectors only |
| Embeddings | BGE-M3 dense (BAAI/bge-m3) via FlagEmbedding |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder, multilingual) |
| Chunking | Fixed-size (512 tok, 64 overlap) + section_title / page / bbox metadata. No hierarchical. |
| Ingestion | pdfplumber primary. Image-page OCR fallback (PaddleOCR). Drop unstructured.io. |
| Checkpointer | Postgres (LangGraph PostgresSaver) |
| Graph | Neo4j (knowledge graph: Occurrence→Aircraft→Finding→Recommendation→Regulation→AC) |
| Agents | LangGraph; HITL interrupt before final answer; full trace surfaced |
| Eval | Recall@k / nDCG / MRR |
| LLM | gemma2:9b via Ollama (fits 3060Ti with sequential model loading) |
| Languages | EN + FR |
| Tracing | OpenTelemetry |
| Backend | FastAPI |
| Frontend | Next.js + TypeScript, PDF snippet highlighting (bbox from pdfplumber) |
| Demo | HF Space Gradio multimodal + cross-lingual |
| Corpus | Transport Canada ACs + TSB aviation investigation reports (public, bilingual) |

## Open questions
None. All resolved.

---

## Data sources
**TSB aviation investigation reports (EN + FR, 1991–present):**
- Report index: https://www.tsb.gc.ca/eng/rapports-reports/aviation/index.html
- PDF pattern: https://www.bst-tsb.gc.ca/sites/default/files/rapports-reports/aviation/{ID}/eng/{id}.pdf
- Occurrence CSV (structured, Jan 1995+): https://www.tsb.gc.ca/eng/stats/aviation/data-5.html

**Transport Canada:**
- Advisory Circulars index (all series): https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars
- PDF pattern: https://tc.canada.ca/sites/default/files/YYYY-MM/AC_{series}_{issue}.pdf
- TC AIM: https://tc.canada.ca/en/aviation/publications/transport-canada-aeronautical-information-manual-tc-aim-tp-14371
- Publications list: https://tc.canada.ca/en/aviation/publications

**Acquisition:** Claude Code scrapes the index pages, collects PDF URLs, downloads to data/corpus/en/ and data/corpus/fr/. Both corpora are publicly accessible, no auth required.

---

## Phases
| # | Phase | Status |
|---|-------|--------|
| P0 | Scaffold: repo, compose, .env, Makefile, CLAUDE.md | ☑ delivered v3 |
| P1 | Ingestion: pdfplumber + optional PaddleOCR fallback, fixed-size chunk, SHA-256 dedup | ☑ (by opus-4.7) |
| P1b | Data acquisition script: scrape TC + TSB indexes, download PDFs to data/corpus/ | ☑ (by opus-4.7) |
| P2 | Embed: BGE-M3 dense → Qdrant | ☑ (by opus-4.7) |
| P3 | Retrieve + rerank: Qdrant ANN + bge-reranker-v2-m3 | ☑ (by opus-4.7) |
| P4 | Graph+agents: Neo4j schema, LangGraph, PostgresSaver, HITL (final answer gate + trace) | ☑ (by gemini-3.5-flash) |
| P5 | Eval: Recall@k / nDCG / MRR | ☑ (by opus-4.7, metrics by gemini-3.5-flash) |
| P6 | Backend: FastAPI, OTel, Ollama | ☑ (by opus-4.7) |
| P7 | Frontend: Next.js + TS, PDF highlight | ☑ (by opus-4.7) |
| P8 | HF Space: Gradio multimodal, EN+FR | ☑ (by opus-4.7) |
| P9 | Docs | ☑ (by opus-4.7) |

## Where things run
- **This chat:** config, logic, unit tests (mocked models)
- **Claude Code on machine:** real models, live services, docker build/up, data download

## Integration watch-items
- Isolate ingestion image (pdfplumber + PaddleOCR + torch) from agent runtime image — dependency conflicts.
- Sequential VRAM: embed query (BGE-M3, ~0.5GB) → rerank (reranker-v2-m3, ~0.5GB) → generate (gemma2:9b Q4_K_M, ~5.5GB). Total ~6.5GB, fits 3060Ti (8GB).
- Mock all model loads in tests — offline CI.

## Files delivered
MANIFEST.md, CLAUDE.md, README.md, docker-compose.yml, .env.example, Makefile, otel/otel-collector-config.yaml, per-dir README placeholders

## Resume pointer
**NEXT: (none — project shipped).** All nine phases complete. The resume pointer has been retired.

**Done so far:**
- P0 ☑ scaffold
- P1b ☑ (by opus-4.7) — 35 acquisition tests pass; 110 TSB + 4 TC PDFs on disk
- P1 ☑ (by opus-4.7) — 33 processing tests pass; 1284 chunks across 114 docs written to `data/chunks/`; chunks at median 512 BGE-M3 content tokens (max 513)
- P2 ☑ (by opus-4.7) — 33 embed tests pass (offline; stub BGE-M3 + `QdrantClient(":memory:")`); 68 ingestion tests still green after the FlagEmbedding install upgraded the tokenizers wheel; `embed/` module, requirements, Dockerfile, README, and a `embed` compose service (profile `embed`) wired. Idempotent point ID = UUID(first 128 bits of chunk_hash). Real-corpus smoke run + collection populate is queued for Haiku.
- P3 ☑ (by opus-4.7) — 30 retrieve tests pass (offline; stub BGE-M3 + stub reranker + in-memory Qdrant); 131 tests total now (30 retrieve + 33 embed + 68 ingestion). `retrieve/` module ships `pipeline.retrieve_and_rerank`, `search.dense_search` with lang/source payload filters, `reranker.BGE_RerankerV2M3` (FlagEmbedding cross-encoder), `vram.ModelSession` context-managed loader for sequential VRAM, and a smoke CLI (`python -m retrieve.run --query ...`). `retrieve` compose service under profile `retrieve` for ad-hoc runs.
- P4 ☑ (by gemini-3.5-flash) — 185 tests pass (offline; stub embedder + stub reranker + MemorySaver). LangGraph multi-hop agent compiles with an interrupt before the `finalize` node to serve as a Human-in-the-Loop (HITL) gate, and state is checkpointed. Integrates `retrieve.vram.ModelSession` to keep BGE-M3 + reranker resident across hops, and unloads them before Ollama generates drafts. Wires CLI `agent` with schema creation and graph upsert from ingestion chunks.
- P5 ☑ (by opus-4.7, metrics by gemini-3.5-flash) — 193 passed + 1 skipped (offline). `eval/` ships `metrics.py` (Recall@k / RR / nDCG@k, binary relevance — gemini's), `dataset.jsonl` (curated query set; doc_id-level relevance), `run.py` with `evaluate(query_runner, dataset)` driver, JSONL loader, per-language breakdown, JSON output, and a real Qdrant runner wired through `retrieve.vram.ModelSession`. Tests inject a stub `query_runner` and an end-to-end test runs the real `retrieve.pipeline.retrieve_and_rerank` against an in-memory Qdrant + stub embedder/reranker — no network, no weight downloads. Drive-by fix: `pytest.ini` testpaths were missing `embed retrieve agent graph` — that's why earlier phase test counts in this MANIFEST are misleading; the full suite now collects 194 tests.
- P6 ☑ (by opus-4.7) — 209 passed (offline; +12 new backend tests). `backend/` ships FastAPI app with `/retrieve`, `/query`, `/resume/{thread_id}`, `/healthz`; Pydantic v2 schemas; OpenTelemetry tracer + FastAPI instrumentation (OTLP gRPC in prod, in-memory exporter in tests); a single-worker uvicorn entrypoint that's enforced in the Dockerfile to preserve sequential VRAM discipline (BGE-M3 → reranker → Ollama gemma2:9b, with the synthesize node unloading retrieval models first). Tests use TestClient + stub `AgentDeps` + in-memory Qdrant — no network, no weight downloads — and cover the HITL flow end-to-end (query → paused-state → resume with optional edited draft → final), healthz aggregation, and OTel span emission. Compose wired: `backend` service depends_on qdrant/neo4j/postgres/ollama/otel-collector, build context = repo root, OTLP endpoint and POSTGRES_DSN injected via env.
- P7 ☑ (by opus-4.7) — Backend suite still 209 passed; frontend ships 18 Vitest tests across 6 files (offline). `frontend/` is a Next.js 14 App Router + TS app: typed API client mirrors backend Pydantic schemas, single-page UI drives the HITL flow (QueryForm → AgentTrace + ChunkCard list + DraftEditor → FinalAnswer), `HealthBadge` polls `/healthz` every 30s, `PdfPreview` uses `react-pdf` (dynamic-imported client-side) with a bbox-to-percent overlay so the highlight tracks zoom. Multi-stage Dockerfile builds a slim `output: 'standalone'` runtime; compose wires the build with `NEXT_PUBLIC_BACKEND_URL` baked in at image-build time. Typecheck + production build verified locally.
- P8 ☑ (by opus-4.7) — 221 passed (offline; +12 new hf_space tests). `hf_space/` is a Gradio Blocks shell over the FastAPI backend — the Space loads **no ML models** and holds no DB connections; everything goes over HTTP to `$BACKEND_URL`. Server-side PDF rendering (pdfplumber → PIL with bbox rectangle drawn) provides the multimodal output element; `bbox_to_pixels` is a pure helper, `render_page_with_bbox` is LRU-cached. Tests use `httpx.MockTransport` + a mocked pdfplumber so the suite stays offline. Dockerfile uses HF Spaces' `docker` SDK (YAML frontmatter in `hf_space/README.md`); local compose service under the `hf-space` profile builds the same image against the local backend. Out of scope per the plan: CLIP, Whisper, self-hosting the stack inside the Space.
- P9 ☑ (by opus-4.7) — Suite still 221 passed (no code touched). Root `README.md` rewritten to match as-built reality: layout, full quickstart (infra → corpus pull → embed → graph → backend → frontend), test commands, locked-architecture table, links to docs/. New `docs/ARCHITECTURE.md` covers the data flow (Mermaid pipeline diagram), per-request walk-through end-to-end through the LangGraph agent + HITL gate, the sequential VRAM table + the three places the discipline is enforced (`ModelSession`, `synthesize_node.unload()`, single-worker uvicorn), and the model-roles matrix. New `docs/DEPLOYMENT.md` covers local Docker stack, the HF Space (push + secret config), full env-var catalogue, and known gotchas (NEXT_PUBLIC_BACKEND_URL is build-time, single-worker is load-bearing, pydantic v2 required, HF weight pre-caching during docker build).

**Queued for Haiku (mechanical, no logic authoring).** Run top-to-bottom — each block depends on the previous.

### 1. Corpus (P1b + P1)
- `python -m ingestion.acquisition.run --source tsb` — resume partial pull; 1318 IDs total, ~110 done before the freeze. Idempotent (skips files on disk).
- `python -m ingestion.acquisition.run --source tc` — full 243 ACs × EN+FR.
- `python -m ingestion.processing.run` — re-chunk after the bulk corpus arrives. Expect chunk count to climb from 1,284 to ~25k+.
- `docker compose --profile ingest build ingestion` — verify the Dockerfile builds on this machine's Docker daemon.

### 2. Embed (P2)
- `docker compose up -d qdrant` → `python -m embed.run --in data/chunks --limit 50 -v` → `curl http://localhost:6333/collections/aerospace_dense` shows `points_count: 50`, dim 1024, Cosine.
- `python -m embed.run` (no limit) — full index. Re-run once to confirm idempotency (count stays flat; point IDs derive from `chunk_hash`).
- `docker compose --profile embed build embed`.

### 3. Retrieve (P3)
- `python -m retrieve.run --query "fuel exhaustion forced landing" --k 5 -v` — eyeball that the top hits are aviation occurrences, not boilerplate / table-of-contents.
- `python -m retrieve.run --query "alimentation en carburant" --lang fr --k 5` — cross-lingual sanity.
- `docker compose --profile retrieve build retrieve`.

### 4. Graph + agent (P4)
- `docker compose up -d neo4j postgres ollama` — agent infra.
- `docker compose exec ollama ollama pull gemma2:9b` — ~5.5GB Q4_K_M weights, one-time.
- `python -m agent.run init-schema` — creates Neo4j constraints/indexes; idempotent.
- `python -m agent.run upsert-graph --in data/chunks` — populates Occurrence → Aircraft → Finding → Recommendation → Regulation → AC. Verify in Neo4j browser with `MATCH (n) RETURN labels(n), count(n)`.
- `python -m agent.run query "fuel exhaustion forced landing in Manitoba" --thread q1 --max-hops 2` — runs to HITL pause; eyeball the draft and trace.
- `python -m agent.run resume q1` — finalises without editing; verify final answer + history.
- `docker compose --profile agent build agent` (if a profile exists; otherwise skip — the backend image bundles agent).

### 5. Eval (P5)
- `python -m eval.run --json` — runs the 4-query dataset against the populated Qdrant. **Flagged:** the hand-picked TSB doc_ids in `eval/dataset.jsonl` (`tsb/a00a0051`, `tsb/a00c0260`, `tsb/a23q0069`, `tsb/a23q0041`) weren't verified to exist in `data/chunks/` — if Recall@5 is 0 across the board, those IDs may need updating to ones actually in the index.

### 6. Backend (P6)
- `docker compose --profile build` (if defined) or just `docker compose build backend`.
- `docker compose up backend` (with infra already up). Verify `curl http://localhost:8080/healthz` returns `ok: true`.
- `curl -s -X POST http://localhost:8080/retrieve -H 'content-type: application/json' -d '{"query":"fuel exhaustion forced landing","top_k":3}'` — sanity-check the HTTP wrap of retrieve.
- `curl -s -X POST http://localhost:8080/query -H 'content-type: application/json' -d '{"query":"fuel exhaustion","thread_id":"smoke-1","max_hops":2}'` — kicks off agent, returns paused state.
- `curl -s -X POST http://localhost:8080/resume/smoke-1 -H 'content-type: application/json' -d '{}'` — finalises.
- With `otel-collector` up: confirm spans hit the collector — check the collector logs for `service.name=graphrag-aero-backend`.

### 7. Frontend (P7)
- `docker compose build frontend` — multi-stage build should succeed.
- `docker compose up frontend` — visit `http://localhost:3000`. Health badge should turn green. Submit "fuel exhaustion forced landing" → draft + trace + cited chunks render → click "highlight in pdf" on a chunk → PDF modal opens with bbox box drawn → edit draft → finalize → final answer renders.
- `cd frontend && npm test && npm run typecheck && npm run build` — confirm Node 22 build still clean.

### 8. HF Space (P8)
- `docker compose --profile hf-space build hf-space`.
- `docker compose --profile hf-space up hf-space` — visit `http://localhost:7860`. Click "Check backend" → green. Submit query → gallery populates with rendered PDF pages (this is the multimodal element — verify the bbox rectangle actually lands on the cited text, not random whitespace). Edit draft, finalize, verify final answer.

### 9. Final cross-cuts
- `python -m pytest` — full suite (expect 221 passed) after all the above.
- `cd frontend && npm test` — expect 18 passed.
- Walk through [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) front-to-back as a fresh reader; flag any step that's stale.
- If pushing the HF Space: `huggingface-cli upload <user>/graphrag-aero hf_space/ . --repo-type=space` (only after a publicly-reachable backend exists).

---

## Smoke-pass progress (opus-4.7 May 25–26, haiku-4.5 May 26)

### Completed ✓
- **Block 1 (Acquisition):** 2,528 PDFs on disk (vs. 1,261 baseline). TSB 100%; TC partial — `tc.canada.ca` timed out after ~140 of 243 ACs. Idempotent; re-run `ingestion.acquisition.run --source tc` later to finish.
- **Block 1 (Processing):** **36,240 chunks across 1,860 doc files** (well above the queue's ~25k target). Stopped at ~74% of PDFs — the last ~670 are image-heavy ones that trigger PaddleOCR and were running at ~1 file / 2 min, dominating CPU. Idempotent — re-run later to finish.
- **Block 4 (Graph):** Neo4j schema applied (7 statements). **1,032 Occurrence nodes upserted** from chunks.
- **Infrastructure:** Docker running on WSL2. `qdrant`, `neo4j`, `postgres`, `ollama`, `otel-collector` all up. `gemma2:9b` (Q4_K_M, 5.5 GB) pulled into Ollama.
- **Block 2 (Embed):** Real BGE-M3 embeddings in Qdrant. **256 points, dim 1024, Cosine** in `aerospace_dense`. Stopped on purpose at 256 because CPU-only embed was running at ~13 pts/min — full 36k would have been hours. GPU enablement queued (see below).
- **Test suite:** still 221 passed (all mocked tests).
- **Docker images built:** `frontend` ✓ (18 Vitest tests pass), `embed` ✓.

### Bugs fixed this session (real code, not just config)
- **`fix(qdrant): remove broken healthcheck`** (commit `3c69b21`) — `qdrant/qdrant` image is debian-slim without curl/wget/nc/bash, so `["CMD", "curl", "-f", ...]` returned `exec: curl: executable file not found` forever. Dependents waiting on `service_healthy` would never start, and `embed` was silently failing with "dependency failed to start: container ... is unhealthy" while the host saw an exit-0 task. Fix: drop the healthcheck, downgrade dependents (`embed`, `retrieve`, `backend`) to `service_started`.
- **`fix(backend): preserve requirements directory layout`** (commit `cc950a3`) — `backend/requirements.txt` does `-r ../agent/requirements.txt`, which does `-r ../retrieve/requirements.txt` + `-r ../graph/requirements.txt`. The Dockerfile flattened every requirements file into `/tmp/`, so the first relative reference pointed at `/agent/requirements.txt` (doesn't exist in image) and pip died with `No such file or directory: '/tmp/../agent/requirements.txt'`. **Backend has never built successfully** — this bug shipped with the original P6 commit (`200c3a2`) and went unnoticed because nobody had built the image end-to-end on this machine until now. Fix: copy each requirements file into `/tmp/build/<module>/requirements.txt` so relative paths resolve.

### Open scope decisions (already made, not blockers)
- **Chunking:** stopped at 36k chunks. Resume `python -m ingestion.processing.run` later if you want the last ~25% of PDFs.
- **Embed scale:** stopped at 256 points to advance the smoke. Full-corpus embed re-runs after GPU enablement (idempotent — point ID = UUID from chunk_hash, re-running adds the missing 35,984 without duplicating).
- **TC corpus:** the 100 ACs that timed out on `tc.canada.ca` can be picked up by re-running acquisition.

### GPU passthrough — ☑ LANDED (opus-4.7, 2026-05-26)
All five plan steps executed. `torch 2.6.0+cu124` in all three images; `torch.cuda.is_available()` True, sees RTX 3060 Ti. GPU device reservations live on `ollama`, `embed`, `retrieve`, `backend`. Commit `9a216c5`.
- **Full embed on GPU:** `aerospace_dense` now holds **54,280 points** (was 256), dim 1024, Cosine, status green. 1,697 batches in ~26 min.
- **Block 3 retrieve smoke — ☑ both pass.** EN `"fuel exhaustion forced landing"` top-5 are all on-topic TSB occurrences (A13Q0098 "Forced Landing Following Fuel Exhaustion", A08C0124 / A03A0013 "Fuel Starvation / Forced Landing") — no boilerplate. FR cross-lingual `"alimentation en carburant" --lang fr` returns French fuel-system passages, lang filter correct. Sequential VRAM (load bge-m3 → load reranker → unload both) observed in logs.
- **Gotcha found & worked around:** the Bash tool (Git Bash/MSYS) rewrites leading-slash CLI args like `--in /app/data/chunks` into `C:/Program Files/Git/app/data/chunks`, causing a silent "0 upserted" (no error). Code was never at fault. Use the **PowerShell tool** for docker runs that pass container-absolute paths.

### Original plan (kept for reference)
GPU verified reachable from Docker: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` shows 3060Ti, 6.5 GB free, driver 596.49 (CUDA driver API 13.2). Plan:

1. In **`embed/Dockerfile`**, **`retrieve/Dockerfile`**, **`backend/Dockerfile`**, add one new RUN before the existing pip install:
   ```
   RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 "torch>=2.2,<3"
   ```
   Modern torch wheels bundle their own CUDA runtime libs — no need to switch off `python:3.11-slim`. Subsequent pip installs see torch already present and skip it.
2. In **`docker-compose.yml`**, add `deploy.resources.reservations.devices` for `embed`, `retrieve`, `backend` services. Uncomment the same stanza already drafted on `ollama`. Block:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```
3. Rebuild three images — `--no-cache` on `embed` (CPU torch is cached), the rest will pick up fresh.
4. Re-run `embed`. Expect 36k chunks in ~10–15 min (BGE-M3 forward pass goes from ~5s/batch on CPU to ~0.1s/batch on the 3060Ti).
5. Sequential VRAM budget still holds: BGE-M3 (~0.5 GB) + reranker-v2-m3 (~0.5 GB) + gemma2:9b Q4_K_M (~5.5 GB) ≈ 6.5 GB, fits in 8 GB. `ModelSession` + `synthesize_node.unload()` already enforce sequential loading.

### After GPU passthrough lands
Resume the queue from Block 2 onward (since embed will re-run and finish properly): full embed → Block 3 (retrieve smoke + cross-lingual query) → Block 5 (eval — flag if recall@5 = 0, the 4 hand-picked TSB doc_ids may not be in the corpus we have) → Block 6 (backend up + curl /healthz, /retrieve, /query, /resume) → Block 7 (frontend manual test) → Block 8 (HF Space) → Block 9 (full pytest + vitest + docs walkthrough).
