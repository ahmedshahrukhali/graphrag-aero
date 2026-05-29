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
**NEXT: open — user to visually click-test the S15 inline-gallery + citation-anchored bbox at :7860, then pick next feature.**
S15 (session 15, opus-4.7) landed two UI pivots on the hf-space, committed but **not yet
human-verified visually** (no browser MCP connectable this session): (1) PDF highlights are
now anchored to the answer's citations via `pdfplumber.Page.search`, no box if the cited span
isn't found — the mis-placed stored chunk bbox is bypassed; (2) the right "Sources" sidebar is
gone — source pages render as a zoomable `gr.Gallery` message **inline in the chat** after the
answer, chunk snippets folded into the "📑 Sources" accordion. Both pivots + revert paths are
documented in `hf_space/README.md` ("Design notes — S15 pivots"). Stack is left UP at :7860
(`docker compose stop` to free GPU/RAM). hf-space tests green (pdf_render +6: search_page_bbox
hit/miss/empty/error, locate_text hit/miss).
Candidate next work after visual confirm: (a) the **real** bbox fix upstream in
`ingestion/processing/chunk.py::_join_pages` (char→bbox alignment desync) so the stored bbox is
trustworthy; (b) pre-existing reranker dtype / tokenizer "Already borrowed" concurrency bug if it
recurs under load; (c) the trycloudflare.com tunnel exposing /retrieve — confirm intended or shut down.

**S5 = Live-verify Gradio 5 chat rebuild** ☑ (session 14, opus-4.7, 2026-05-28) — functional core verified live; visual click-through handed to S6.

### S5 results
- **SSE order (live `/query/stream`, query "fuel exhaustion"):** `status, status → sources (×1) → status×3 (graph_expand) → token×many → done`. `done` payload carries thread_id+draft, **no sources** ✅. Matches the unit test exactly.
- **Grounded, not stubbed:** retrieved the exact fuel-exhaustion report (tsb/a03q0109) + synthesized a cited answer (`[tsb/a03q0109 p.4]`, `[tsb/a97o0103 p.4]`…) ✅
- **OTel (the real correctness surface):** spans flow to collector, `service.name=graphrag-aero-backend`; custom `agent.query`/`agent.resume`/`retrieve` spans present; **0 errors on today's run**. (Two stale RuntimeErrors in the log — `expected scalar type Float but found Half` (reranker fp16/fp32) and `Already borrowed` (tokenizer concurrency) — are from 2026-05-27, a prior session; flagged below, not introduced by S5.)
- **UI shell:** :7860 serves HTTP 200 (gradio 5.50.0); config exposes the rebuilt tree — 2 sidebars, 1 chatbot, tabs+2 tabitems, gallery, accordion, 2 datasets, 2 radios, slider ✅
- **⚠ Pre-existing backend bug to watch (not S5 scope):** reranker dtype mismatch + tokenizer "Already borrowed" under concurrent `/query` (LangGraph path) on 05-27. Did not recur today. Revisit if it reappears under load.

**S1 = Smoke pass** ☑ (sessions 3–4, opus-4.7 + sonnet-4.6, 2026-05-26)
**S2 = Fix pass** ☑ (session 5–6, sonnet-4.6, 2026-05-27) — bbox fallback, CORS, eval pdfplumber refactor, force-reprocess 2878 chunks
**S3 = Embed + Eval** ☑ (session 7+, opus-4.7, 2026-05-27) — full corpus re-embed (63,946 pts), post-fix bbox eval

### S3 results
- **Post-fix bbox eval:** 50 TSB chunks sampled
  - hit_rate: 30% (15/50) — **+50% vs baseline** (20% pre-fix)
  - mean_similarity: 0.267 — **+30% vs baseline** (0.205 pre-fix)
  - errors: 0
- **Status:** Improvement confirmed but below 70% target. Worst cases remain page markers ("- 2 -", "- 7") from cross-page chunks.
- **Options:** (1) Accept 30% and move forward (real eval shows retrieval+synthesis work end-to-end); (2) Iterate bbox further (diminishing ROI); (3) Skip bbox-specific focus, address in frontend rendering layer.

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
- `python -m pytest` — full suite (expect 222 passed) after all the above.
- `cd frontend && npm test` — expect 18 passed.
- Walk through [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) front-to-back as a fresh reader; flag any step that's stale.
- If pushing the HF Space: `huggingface-cli upload <user>/graphrag-aero hf_space/ . --repo-type=space` (only after a publicly-reachable backend exists).

---

## HF Space Tier-2 UI redesign (sonnet-4.6, 2026-05-27)

**Commit:** `d99e357` `ui(hf-space): tier-2 three-pane redesign + UI mockups`

Three-pane Gradio layout replaces the single-column Tier-1 shell:
- **LEFT rail** — dark sidebar: brand, "+ New question" button, BrowserState-backed
  recent-query list (persists across page reloads in Gradio), canned sample queries
  (clicking populates the composer without auto-submitting), lang/source filter radio
  pills, feature shortcuts, health badge.
- **CENTER pane** — chat-style HITL flow: user bubble (right-aligned), HITL explainer
  card, draft card (amber) with editable textarea + Finalize / Discard buttons, final
  card (green) with rendered markdown. Sticky composer at bottom with multi-line
  textbox, Send ↑ button, ghost Stop button (cancels in-flight requests via
  `cancels=[event_a, event_b]`), and max-hops slider.
- **RIGHT rail** — Sources / Trace / Logs tabs: Sources = PDF-page gallery + chunk
  markdown; Trace = node/elapsed_ms/extras dataframe; Logs = pseudo-log derived from
  agent trace (no real log stream from backend).

`mockups/` committed alongside: three HTML design candidates used as reference.

**Pending (needs Docker Desktop + running stack):**
1. Redeploy HF Space: `huggingface-cli upload ahmedsali/graphaero-rag hf_space/ . --repo-type=space`
   (or push via HF web UI) after restarting the cloudflared tunnel and re-setting `BACKEND_URL` secret.

---

## Generation-quality fix: doc-anchored retrieval + num_ctx (opus-4.7, 2026-05-27)

**Symptom:** gemma2:9b drafts refused to synthesize — they answered "not covered
in the cited sources" or dumped bare `[1][2]...` citation markers instead of prose.

**Two root causes, both confirmed live against the running stack (Qdrant 54,280 pts):**

1. **Silent context truncation.** Production instantiated `OllamaLLM()` with no
   options, so Ollama used its default `num_ctx` (4096 for gemma2). The synthesize
   prompt runs ~4.7k tokens, so the tail of the citation block was dropped before
   the model ever saw it. Probe: identical prompt at default ctx → `prompt_eval=4096`
   (capped) vs `num_ctx=8192` → `prompt_eval=4680` (full prompt fits).
2. **Title-page retrieval.** Chunk-level ANN+rerank correctly identifies the right
   *documents* — but the chunks it returns are the keyword-rich **cover/title pages
   and running-header dates** (`§AVIATION INVESTIGATION REPORT`, `§26 JULY 2003`),
   which carry zero findings/analysis text. The LLM had nothing to synthesize.

**Fix (Part A — retrieval, no graph changes):**
- `agent/llm.py`: `OllamaLLM` now defaults options to `num_ctx` (env `OLLAMA_NUM_CTX`,
  default 8192) + `temperature` (env `OLLAMA_TEMPERATURE`, default 0.2). Centralized so
  both call sites (`agent/run.py`, `backend/deps.py`) get it.
- `retrieve/search.py`: `scroll_doc_chunks(client, collection, doc_ids)` — fetch every
  chunk of a document (no ANN).
- `retrieve/pipeline.py`: `anchored_retrieve(...)` — seed via `retrieve_and_rerank` →
  take top-N unique docs → scroll their full chunk sets → rerank the pool → greedy-fill
  a char budget (default 24k) → re-sort `(doc_id, page)` for reading order. Defaults
  `top_n_docs=3`, `char_budget=24000`.
- `agent/nodes.py`: `AgentDeps.anchored` (default **on**; opt-out via env
  `RETRIEVE_ANCHORED=0`) + `top_n_docs`/`char_budget`. Anchored `retrieve_node`
  replaces candidates wholesale (bypasses `_merge_candidates` top_k truncation so the
  char budget + reading order govern).
- `agent/prompts.py`: system prompt now instructs synthesis across citations (was
  "answer ONLY … do not speculate"); `format_citations` max_chars 800 → 2000 (pairs
  with the num_ctx headroom).

**Live validation (diagnostic probes, since productionized):** anchored retrieval on
"fuel exhaustion forced landing" pooled 248 chunks across the top-3 docs, selected 11
content pages (p.25/55/62/68/84 — real Findings), and gemma produced a cited synthesis
(`eval=240`, inline `[tsb/a13q0098 p.62]`/`[p.84]`) vs the bare-marker dump before.

**Tests:** full suite **237 passed** (was 222; +15 — `scroll_doc_chunks`,
`anchored_retrieve` char-budget/lang/reading-order, anchored `retrieve_node`, num_ctx
defaults). All offline/mocked.

**Not yet done (integration):** the backend image still has the pre-fix code baked — a
rebuild + live `/query` is pending (held because the running backend is exposed to the
published HF Space via cloudflared tunnel; restart drops demo sessions).

**Part B (next — pending live upsert-graph run):** code landed, tested offline (see below).
Live `upsert-graph` run is the Haiku/manual step to populate Neo4j from the 63,896-point
corpus. After that: re-run `eval/graph_eval.py` to verify TraversalHit > 0, rebuild the
backend image, and verify end-to-end `/query` includes graph facts.

---

## Knowledge-graph population: Part B (opus-4.7, 2026-05-27)

**Problem:** Neo4j held 1,032 isolated `Occurrence` nodes, 0 relationships, 0 other labels.
The graph_expand node in the agent returned bare `{id, source_url, lang}` rows with no
findings/regs — not citeable, zero contribution to synthesis. The "multi-hop graph RAG"
headline was hollow; every good answer came from vector retrieval alone.

**Corpus grounding** (`_diag_sections.py` probe before coding):
- section_title metadata is garbage (17,679 empty, 34,831 date-like headers) — detection
  must be content-based.
- Section header patterns confirmed in corpus (EN + FR):
  - `Findings as to Causes…` 1103 chunks, `Findings as to Risk` 862, `Safety Action` 1517
  - FR equivalents: 1093 / 838 / 1654
- `CAR \d{3}(\.\d+)+` 1024 hits; `AC \d{3}-\d{3}` 3887 hits; `A\d{2}-\d{2}` rec IDs 1495
- 1,172 / 1,680 doc_ids carry both EN+FR chunks → Occurrence nodes must be lang-agnostic.

**Changes:**

`graph/extract.py` — full rewrite. `RegexExtractor`: CAR/AC/TSB-rec citations from all
chunks; content-based EN+FR section header detection → numbered list item extraction.
`LLMExtractor`: only runs on section-bearing chunks, prompts gemma to return structured
JSON `{findings, recommendations}`, parses/validates response (markdown fences, malformed
JSON handled gracefully). `HybridExtractor`: regex on all chunks + LLM on section chunks,
merges with LLM findings/recs taking precedence over regex list items (richer text).

`graph/upsert.py` — extended. `upsert_acs_from_chunks`: mints `AC` nodes from TC corpus
(AC number extracted from doc_id). `upsert_entities_from_chunks`: runs extractor over all
chunks, MERGEs Finding/Recommendation/Regulation/AC nodes + `HAS_FINDING` / `HAS_RECOMMENDATION`
/ `CITES` / `REFERENCES_AC` / `GUIDED_BY` edges. Every Finding/Recommendation carries
`source_doc_id + page` — provenance is baked in.

`graph/query.py` — new traversal query. Follows `HAS_FINDING → CITES` and
`HAS_RECOMMENDATION` in one round-trip; returns `{findings, recommendations, direct_regs,
acs}` per occurrence with full provenance. `_clean_collect` strips Neo4j OPTIONAL MATCH
null-collection rows.

`agent/prompts.py` — `format_graph_context` rewritten. Rich traversal rows rendered as
inline-cited lines (`[tsb/a01 p.5] cause: Fuel tanks empty [cites CAR 602.115]`). Legacy
`{occ_id, occ_url}` rows with no findings fall back to minimal one-liner. System prompt
updated to mention graph context citations. User template labels updated.

`agent/run.py` — `upsert-graph` now calls `upsert_occurrences_from_chunks` +
`upsert_acs_from_chunks` + `upsert_entities_from_chunks`. `--extract` flag enables
`HybridExtractor` (regex + gemma); omitting uses `RegexExtractor` only (faster, no Ollama).

`eval/graph_eval.py` + `eval/graph_dataset.jsonl` — new multi-hop eval. `TraversalHit@occ`
metric: for each known occurrence, does the traversal return findings with expected keywords?
4 query/occurrence pairs covering fuel exhaustion, CVR findings-as-to-risk, VFR night
disorientation (FR), cable hazard. Proves graph adds value rather than asserts it.

**Tests:** **288 passed** (was 237; +51 new offline tests). Covers: all 6 EN+FR section
patterns, regex CAR/AC/TSB-rec extraction, LLM JSON parsing (incl. markdown fences + bad
JSON fallback + LLM exception handling), HybridExtractor merge logic, upsert batching +
entity graph writes + CITES links, traversal query shape, graph_eval metric.

**Pending (live steps — Haiku/manual):**
1. `python -m agent.run upsert-graph --in data/chunks` — regex-only pass (fast, no Ollama)
2. `python -m agent.run upsert-graph --in data/chunks --extract` — full hybrid (LLM) pass
3. `python -m eval.graph_eval` — verify TraversalHit > 0 on the 4 eval occurrences
4. Neo4j browser: `MATCH (n) RETURN labels(n), count(n)` — confirm all 6 labels populated
5. Rebuild backend image; live `/query` to confirm graph facts appear in synthesis

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
- **`fix(eval): dedupe repeated doc_ids in nDCG`** (commit `98cba99`) — `ndcg_at_k` iterated the raw chunk-level ranked list; multiple chunks from the same doc produced duplicate doc_ids, and a relevant doc appearing 4× in top-10 accumulated gain past IDCG (q02: nDCG=1.43). Fix: dedupe `actual` preserving first-occurrence order before computing DCG. After fix: nDCG@5=nDCG@10=1.0 for all 4 eval queries.
- **`fix(backend): enter PostgresSaver context manager for app lifetime`** (commit `1dedddc`) — `make_postgres_saver()` returns a `_GeneratorContextManager`; the backend's `build_default_deps()` was passing the unentered CM directly to LangGraph, causing `AttributeError: '_GeneratorContextManager' object has no attribute 'get_next_version'` on every `/query` call. Fix: enter via `contextlib.ExitStack` at startup in `build_default_deps`, call `setup()`, store `stack.close` as `closer` on `BackendDeps`, call in lifespan shutdown.
- **`fix(hf-space): unbreak Gradio startup`** (commit `49e4928`) — two bugs hit in sequence: (1) `gradio 4.x oauth.py` imports `HfFolder` removed in `huggingface_hub 1.0`; fix: pin `huggingface_hub>=0.23,<1.0`. (2) `gradio_client 4.x _json_schema_to_python_type` crashes on `bool` schemas (e.g. `additionalProperties: true`) with `TypeError: argument of type 'bool' is not iterable`; fix: monkeypatch `gradio_client.utils._json_schema_to_python_type` at module import time to short-circuit `isinstance(schema, bool)`.
- **`fix(hf-space): pin starlette<1.0`** (this session) — `pip` resolved `starlette 1.1.0` (starlette just went 1.x). Starlette 1.x removed backward-compat for `TemplateResponse(name, context)` — Gradio 4.x still uses that signature, so starlette passed the context dict as `name` to Jinja2's `get_template`, which tried to hash it as a cache key → `TypeError: unhashable type: 'dict'`. Fix: add `starlette<1.0` to `hf_space/requirements.txt`; pip backtracked to starlette 0.52.1.
- **`fix(acquisition): widen exception catch`** (commit `fdf9add`) — `acquisition/run.py` caught only `requests.HTTPError` around `download()` and `fetch_text()` calls, but a refused TCP connection raises `requests.ConnectionError` (not a subclass). Hit when resuming TC pull: 7 successful skips → one Connection refused on `http://tc.canada.ca:80/.../AC_507-001` → `SystemExit 1`, losing the rest of the 243 ACs in the index. Acquisition has never finished TC cleanly on this machine because of this — every prior run died on the first unreachable URL, leaving 71% of TC unfetched (the "tc.canada.ca timed out after ~140 of 243 ACs" line earlier in this section was this bug, misdiagnosed as a server timeout). Fix: catch `requests.RequestException` at all three call sites. After fix: TC pulled 367 new PDFs cleanly, 23 warnings, no crash.

### Open scope decisions (already made, not blockers)
- **Chunking:** stopped at 36k chunks. Resume `python -m ingestion.processing.run` later if you want the last ~25% of PDFs.
- **Embed scale:** stopped at 256 points to advance the smoke. Full-corpus embed re-runs after GPU enablement (idempotent — point ID = UUID from chunk_hash, re-running adds the missing 35,984 without duplicating).
- **TC corpus:** the 100 ACs that timed out on `tc.canada.ca` can be picked up by re-running acquisition.

### GPU passthrough — ☑ LANDED (opus-4.7, 2026-05-26)
All five plan steps executed. `torch 2.6.0+cu124` in all three images; `torch.cuda.is_available()` True, sees RTX 3060 Ti. GPU device reservations live on `ollama`, `embed`, `retrieve`, `backend`. Commit `9a216c5`.
- **Full embed on GPU:** `aerospace_dense` now holds **54,280 points** (was 256), dim 1024, Cosine, status green. 1,697 batches in ~26 min.
- **Block 3 retrieve smoke — ☑ both pass.** EN `"fuel exhaustion forced landing"` top-5 are all on-topic TSB occurrences (A13Q0098 "Forced Landing Following Fuel Exhaustion", A08C0124 / A03A0013 "Fuel Starvation / Forced Landing") — no boilerplate. FR cross-lingual `"alimentation en carburant" --lang fr` returns French fuel-system passages, lang filter correct. Sequential VRAM (load bge-m3 → load reranker → unload both) observed in logs.
- **Gotcha found & worked around:** the Bash tool (Git Bash/MSYS) rewrites leading-slash CLI args like `--in /app/data/chunks` into `C:/Program Files/Git/app/data/chunks`, causing a silent "0 upserted" (no error). Code was never at fault. Use the **PowerShell tool** for docker runs that pass container-absolute paths.
- **Block 5 (Eval) — ☑ (sonnet-4.6, 2026-05-26):** `python -m eval.run --json` against live Qdrant (54,280 pts). All 4 hand-picked TSB doc_ids exist in the index. Recall@5=1.0, Recall@10=1.0, MRR=1.0, **nDCG@5=1.0**, nDCG@10=1.0 across all queries. The `ndcg > 1.0` bug (doc_id dedup) was found and fixed — see Bugs section.
- **Block 6 (Backend) — ☑ (sonnet-4.6, 2026-05-26):** Backend image rebuilt. `GET /healthz → ok:true`. `/retrieve` returns ranked chunks. `/query` runs agent to HITL pause and returns draft. `/resume` finalises and returns 6-step trace + history. OTel spans confirmed in collector logs (`service.name=graphrag-aero-backend`). PostgresSaver CM lifecycle bug fixed — see Bugs section.
- **Block 7 (Frontend) — ☑ (sonnet-4.6, 2026-05-26):** `docker compose build frontend` exit 0. `docker compose up frontend` — `GET http://localhost:3000 → 200`, "GraphRAG Aero" in title, `NEXT_PUBLIC_BACKEND_URL` baked in JS bundle. Typecheck and `npm run build` clean. 18 Vitest tests pass. Visual gallery/PDF bbox test not verifiable headlessly (no browser MCP this session) — flagged.
- **Block 8 (HF Space) — ☑ (sonnet-4.6, 2026-05-26):** Three startup crashes fixed (see Bugs section). `docker compose --profile hf-space up hf-space` → `HEAD http://localhost:7860/ HTTP/1.1 200 OK` in startup logs. `GET http://localhost:7860 → 200`, "GraphRAG" in HTML. Container stays up. Visual/gallery test requires browser.
- **Block 9 (Full test suite) — ☑ (sonnet-4.6, 2026-05-26):** **222 pytest passed** (was 221; +1 nDCG dedup regression test), **18 Vitest passed**. All mocked, no model weights, no network. Suite is green after all smoke-pass bug fixes.
- **HF Space published — ☑ (opus-4.7, 2026-05-26):** Pushed to https://huggingface.co/spaces/ahmedsali/graphaero-rag (Docker SDK). Backend exposed via `cloudflared tunnel --url http://localhost:8080`; tunnel URL set as the Space's `BACKEND_URL` secret. End-to-end verified from the public Space — `/healthz`, `/retrieve`, `/query` (with PDF fetches for bbox rendering), and `/resume` all returned 200 through the tunnel to the local stack. Caveats: trycloudflare URL is ephemeral (rotates on tunnel restart → re-set secret) and unauthenticated (anyone with the URL hits the backend — kill the tunnel when not demoing).
- **TC corpus completion — ☑ (opus-4.7, 2026-05-26):** `python -m ingestion.acquisition.run --source tc` finished cleanly after fixing `fix(acquisition): widen exception catch` (see Bugs section). Pulled **367 new PDFs** (142 skipped as already present, 509 total references); TC EN 93→253, TC FR 48→255. Chunks and embeddings have **not** been refreshed — corpus on disk = 2,895 PDFs but Qdrant still holds 54,280 points from the pre-pull 2,528. Run `processing.run` then `embed.run` to fold the new TC docs into the index.

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

---

## Live smoke-pass completed (sonnet-4.6, 2026-05-27)

All pending live steps from the previous section executed and verified:
- **Processing:** all 2,895 PDFs already chunked (0 new) — `embed.run` skipped (idempotent).
- **Backend thread-safety fix** (`fix(backend): thread-safe lazy model loader`, commit `900e637`):
  `_LazySession.ensure_loaded()` + `unload()` were not thread-safe; concurrent FastAPI
  requests (sync endpoints run in thread pool) both tried to construct the PyO3 reranker
  simultaneously → `RuntimeError: Already borrowed`. Added `threading.Lock` per instance.
- **Graph populated:** `upsert-graph` regex pass: 1,199 Occurrence + 174 AC + 13,124 Finding +
  2,136 Recommendation + 847 Regulation nodes. `eval.graph_eval` → TraversalHit = 1.0 (all 4
  occurrences found with keyword hits).
- **Backend smoke:** `/healthz` ok; `/retrieve` rerank ≥ 0.994; `/query` → paused draft (coherent
  fuel-exhaustion synthesis, 11 anchored candidates, 351s gemma2:9b synthesis at 30k char context);
  `/resume` → final answer + full 5-step trace + history. Thread 2 → full HITL flow ✓.
- **LLM extraction (`--extract`):** relaunched in background (PID 27324) after installing
  `ollama` Python package (was missing; first pass used regex-only). Runs in background,
  will enrich Finding prose over several hours.
- **pytest:** 288 → 307 passed after bbox eval + OCR fix.

## bbox verification + OCR coordinate fix (sonnet-4.6, 2026-05-27)

**Bug found:** `ingestion/processing/ocr.py` stored PaddleOCR results in **pixel** coords
(200 DPI render), but `hf_space/pdf_render.py`'s `bbox_to_pixels` assumes **PDF point**
coords (72 pts/inch, top-left origin). Every OCR-page chunk had a visually misaligned
bbox highlight in the UI. Fix: divide pixel coords by `(200/72)` before storing.
Commit: `f632f91`.

**`eval/bbox_eval.py`** added: samples chunks from `data/chunks/`, renders source PDF pages
as PIL images, crops bbox regions, runs PaddleOCR on crops, reports character-level
similarity vs stored text. CLI:
```powershell
# Quick 20-chunk sample with crop images saved for inspection:
python -m eval.bbox_eval --n 20 --save-crops crops/ --source tsb

# Full JSON report:
python -m eval.bbox_eval --n 200 --json > bbox_eval.json
```
PaddleOCR must be installed (`pip install paddleocr paddlepaddle`) — it's in the ingestion
Docker image but not the host Python by default.

**19 new tests** (307 total). All offline/mocked.

**`chunk.py` bbox fallback** (sonnet-4.6, 2026-05-27, commit `2e6a8e2`): Cross-page 512-token
chunks land only a page-number line on the dominant page, producing bboxes with area ~100–2k pt²
that are useless for highlighting. `_bbox_for_range` now falls back to the full dominant-page
extent when area < `MIN_USABLE_BBOX_AREA` (5000 pt²). 1 new test (9 total in test_chunk.py).

## Pending live steps — run in order once Docker Desktop is up (sonnet-4.6, 2026-05-27)

All code is committed and tested offline (307 pytest passed). These are the remaining
integration steps that require the running stack.

**Status as of 2026-05-27 session (updated):**
- Steps 1–5 ✅ completed (see "Live smoke-pass" section above).
- Step 6 ✅ HF Space redeployed (sonnet-4.6, 2026-05-27) — see below.
- Step 7 ✅ Baseline bbox eval run — 20% hit rate (expected, pre-fix chunks) — see below.
- Steps 7b + 8: `--force` reprocess 🔄 running in background (~44 min, logs/processing_force.log).
  After it finishes: run embed, then re-run bbox eval to confirm improvement.

### Step 6 — Redeploy HF Space Tier-2 UI ☑ (sonnet-4.6, 2026-05-27)
- Tunnel: `https://transaction-mystery-hold-reform.trycloudflare.com` (ephemeral — rotate on restart)
- `BACKEND_URL` secret updated on `ahmedsali/graphaero-rag` Space.
- `hf_space/` uploaded via `HfApi.upload_folder`.
- Verified: `GET /run/on_healthz → ✅ backend: qdrant=True · neo4j=True · ollama=True`

### Step 7 — Verify bbox accuracy ☑ baseline (sonnet-4.6, 2026-05-27)
**Baseline (pre-fix chunks):** 50 TSB chunks, 0 errors, mean_sim=0.205, **hit_rate=20%**.
Worst cases: `"- 3 -"`, `"- 4 -"` (page numbers) — confirms cross-page tiny-bbox diagnosis.
Re-run after `--force` reprocess + embed to verify improvement to >70%.

### Step 7b + 8 — Reprocess all chunks + re-embed (running)
```powershell
# Running in background: logs/processing_force.log
docker compose --profile ingest run --rm ingestion --in /app/data/corpus --out /app/data/chunks --force
# After it finishes (~50 min total), re-embed:
docker compose --profile embed run --rm embed
# Then re-run bbox eval to verify hit rate > 70%:
python -m eval.bbox_eval --n 50 --save-crops crops/post_fix --source tsb
```

### Step 9 — Optional: additional corpus coverage
```powershell
python -m ingestion.processing.run     # idempotent; picks up remaining image-heavy PDFs
python -m embed.run                    # embed new chunks
```
