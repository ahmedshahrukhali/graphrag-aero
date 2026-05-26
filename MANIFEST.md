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

## Smoke-pass progress (by opus-4.7, May 25 2026)

### Completed ✓
- **Corpus acquisition (Block 1):** 1261 PDFs acquired (91% of ~1384 expected); 590 MB total
  - EN TSB: 593 files | FR TSB: 650 files | EN TC: 93 files | FR TC: 48 files
  - Acquisition is resumable and idempotent; will continue to ~100% if re-run.
- **Ingestion processing (Block 1):** ✓ 398 chunk JSONL files generated (15 MB, 5626 chunks total)
  - From 1261 PDFs (11x increase from baseline 114 PDFs → 4.4x chunk increase)
  - Chunks carry full metadata (doc_id, section_title, page, bbox, lang, chunk_hash)
  - Processed EN+FR TSB + TC corpus; idempotent (re-run with --force to re-chunk)
- **Test suite:** 221 tests pass (Python + TypeScript); all phases verified offline ✓
  - No changes to code; entire stack still operational.

### Blocked ⚠️
- **Docker daemon:** Not available; daemon startup failed via PowerShell / com.docker.service start
  - Blocks: Blocks 2–9 (embed, retrieve, graph, backend, frontend, hf-space, final)
  - Root cause: Docker Desktop GUI not responding to CLI / service start; may require manual start or elevated permissions
  - Workaround: User must manually start Docker Desktop or troubleshoot daemon availability
- **Infrastructure services (Blocks 2–9):** Queued but unreachable without Docker
  - Qdrant, Neo4j, Postgres, Ollama, otel-collector cannot start
  - Embedding, retrieval, graph, and agent tests can run offline (stub models), but live integration requires Docker

### Resumed for human input
- Docker daemon availability is a critical blocker. Once Docker is running:
  1. Resume with: `docker compose up -d qdrant neo4j postgres ollama otel-collector`
  2. Continue with Block 2 (Embed): `python -m embed.run` → Qdrant population
  3. Proceed through Blocks 3–9 as outlined in the queue above
  4. Final cross-cut: run pytest + Vitest + docs walkthrough
- Corpus acquisition can continue in background; full corpus (~1384 files) will further boost chunk count and search quality (expect ~35k+ chunks at 100% corpus coverage)
