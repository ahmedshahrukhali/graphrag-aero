# Session Log

One entry per conversation. Most recent at top. Keep each entry under 10 lines.

---

## Session 20 — 2026-06-02 — opus-4.8
**Commits:** `ca96406`
**Achieved:**
- **CHINESE_OCR_PLAN Commit 1 DONE** (per-language OCR routing + zh/ttsb/caac plumbing). The real change: `ocr.py` now has `paddle_lang(lang, source)` → `latin` (en/fr) / `ch` (zh+caac, Simplified) / `chinese_cht` (zh+ttsb, Traditional), a per-code model cache (angle-cls ON for the scanned Chinese models, off for latin), and `ocr_page(page, page_no, ocr_lang)`.
- Wiring: `run.py` derives the code from the DocRef and threads it through `extract_pages_with_ocr`; `iter_corpus_pdfs` now sweeps `{en,fr,zh}/{tsb,tc,ttsb,caac}` and `--source` admits ttsb/caac. `lang.py` admits `zh`; `doc_id._KNOWN_SOURCES += ttsb,caac` (source_url None, TC-style). `embed/jsonl` LANGS/SOURCES + `backend/schemas` Literal filters admit zh/ttsb/caac. processing README updated.
- **+16 tests, full suite 378 passed** (was 362) — all offline; PaddleOCR stubbed via `sys.modules`, no weights/network. Fixed the 4 existing `test_run.py` extract-stub signatures for the new `ocr_lang` arg.
**Left (resume here → S20 Commit 2):**
- `ingestion/acquisition/ttsb.py` — direct static Traditional-Chinese PDF URLs; wire `--source ttsb` into `acquisition/run.py`; prefer scanned ASC-era reports. Then Commit 3 `caac.py` (search-seed manifest `data/corpus/zh/caac_seed.txt`).
- Then live scanned-Chinese-OCR acceptance test → curated re-ingest. Still open: WS-B :7860 click-through; curation criteria (§3); About tab.

## Session 19 — 2026-05-31 — sonnet-4.6
**Commits:** _(this session)_
**Achieved:**
- **Cross-check fix:** SESSIONS S17 commit list was 1 of 10 (entry written on first commit); S18 was `_(this commit)_` placeholder. Backfilled both. Fixed REINGEST §7 step-2 "word boxes" → `page_bboxes` (contradicted the §4.1 reset) + MANIFEST LLM row now flags the Qwen3-8B swap-under-eval.
- **WS-0 schema freeze DONE** (code-of-record): `page_bboxes` (region-level, one rect per page the chunk touches), `corpus` tag, `kind` discriminator frozen through the whole chain (chunk.Chunk + new `_page_bboxes_for_range`/`_page_union_bbox` → DocRef.corpus → run._chunk_to_record → embed.jsonl.ChunkRecord → agent.state → backend.schemas → hf_space.api_client). All additive/optional → existing 63,946-pt index still hydrates (derive page_bboxes from legacy `(page,bbox)`, corpus from doc_id prefix). +8 tests, full suite **366 passed**.
- **WS-0 VRAM measurement DONE** (resumed next day; Docker was down → senior started Docker Desktop, re-dispatched Haiku). qwen3:8b Q4_K_M = **6.2 GB / 8 GB, 100% GPU, ~1.8 GB free → FITS** (`docs/ws0_vram_measurement.md`). Swap is no longer VRAM-gated, only quality-gated. → **WS-0 fully closed.**
- Wrote **`docs/NEXT_SESSION.md`** (self-contained WS-B + WS-A + Qwen bake-off brief).
- **Figure-tier model DECIDED: Qwen3-VL-8B** (replaces Florence-2 + Moondream2). Spiked via Ollama on a real TSB figure (a13q0098 p.60 fuel-gauge photo, cropped): accurate caption + domain region OCR. VRAM 7.7 GB / 28% CPU spill on the 8 GB card — borderline but fine for the **offline** figure tier. `docs/ws_c_qwenvl_findings.md`; REINGEST_PLAN §1/§4.2/§4.6/§6 + NEXT_SESSION updated. NB: figure-tier only — it spills, so NOT the query-time generator.
- **Generator bench (gemma2 vs qwen3-vl):** on the real synthesis prompt, gemma2:9b produced a clean fully-cited answer; qwen3-vl (as generator) spilled + emitted thinking-only/empty content. → Do NOT unify on the VL model for generation; text generator + Qwen3-VL figures.
- **WS-B DONE:** region-grounding render. `render_page_with_bbox(region_bboxes=...)` draws stored rects; `search_page_bbox`/`locate_text` deleted (S15 desync source); `app._page_regions` filters `page_bboxes` to the cited page. Full suite **362 passed** (−4 vs 366: 7 search-path tests removed, 3 region tests added). Live :7860 click-through still pending.
- **InternVL3 bake-off RAN (real head-to-head):** embed container (host torch CPU-only), bf16+CPU-offload after bnb/CUDA/triton dead-ends. On the same p.60 crop, **Qwen3-VL read 130/57 GAL correctly (verified vs report p.16); InternVL3 misread (150/55)**, and was ~6× slower + far harder to run. → **InternVL3 dropped; Qwen3-VL confirmed with data.** 16 GB model cached in data/hf-cache (gitignored).
- **Env flag:** host starlette 1.2.1 broke fastapi 0.110 backend tests; pinned `starlette==0.36.3` (host-env drift only; Docker images pin via fastapi).
- **WS-A recon → CAAC scrape DEAD** (JS/JSONP TRS WAS5 index, unscrapable; Chrome MCP stalled on a permission prompt). **CORPUS PIVOTED** (user): keep PDF→answer demo + English TC/TSB; add a **Chinese PDF corpus** = Taiwan TTSB (Traditional, direct PDF URLs) + CAAC (Simplified, enumerate via search-engine seed, bypassing the JS index); **enable Chinese OCR** (per-lang PaddleOCR model — the real change). Must include **scanned** Chinese PDFs so OCR fires. A Wikipedia/HTML detour was proposed and **rejected** (the demo is PDF→answer). Plan APPROVED → `docs/CHINESE_OCR_PLAN.md`.
**Left (resume here → S20 = [docs/CHINESE_OCR_PLAN.md](docs/CHINESE_OCR_PLAN.md)):**
- Implement Chinese OCR (per-lang model in `ingestion/processing/ocr.py`) + offline tests FIRST.
- Then `acquisition/ttsb.py` (direct PDFs) + `acquisition/caac.py` (search-seed); lang/doc_id/filter plumbing (add `zh`/`ttsb`/`caac`).
- Live scanned-Chinese-OCR acceptance test, then curated re-ingest. Also still open: WS-B :7860 click-through; curation criteria (§3); About tab.

## Session 18 — 2026-05-31 — opus-4.8
**Commits:** `201cc06`, `aaf41a1`
**Achieved (planning/recon only — no phase code):**
- **ZH sourcing RESOLVED** (closes the §2 gate): two verified, enumerable, license-checked axes — CAAC 咨询通告 ACs ↔ TC ACs (caac.gov.cn, robots GREEN, the spine) + CAAC↔TSB investigation reports (ASN **index-only** → primary PDFs; ASN signals `ai-train=no`, so never bulk-fetch its PDFs). User approved both.
- **Re-sequenced §6** around a write-shape freeze: new **WS-0** (freeze ChunkRecord schema + curation criteria once), ZH demoted from "parallel track" to a fail-fast spike, B→C dependency made explicit.
- **Bbox RESET (§4.1)** — scrapped word-level highlighting (source of the S15 desync + word-box payload + per-word OCR). Grounding is now **region-level** from the chunk's own stored bbox (`page_bboxes`); WS-B collapses to "carry rect + draw rect," delete `page.search`.
- **Model swap under eval (§4.6)** — gemma2:9b → Qwen3-8B generation (VRAM-gated: ~7–9 GB vs 6.5 GB budget, measure in WS-0) + Qwen3-VL-8B on the figure tier (collapses Florence-2+Moondream2); both decided by bake-off on our own docs, not benchmarks.
- Added **§10 Haiku overnight-run monitoring runbook**. Deleted stray `image1.png`.
**Left (resume here):**
- Implement **WS-0** (freeze schema incl. `page_bboxes`/`corpus`/figure `kind`; measure Qwen3-8B VRAM) — plan-mode + HITL per CLAUDE.md. Then WS-A (ZH scraper), WS-B (region render).
- Run the bake-offs: gemma2 vs Qwen3-8B (EN+ZH answer quality); Qwen3-VL vs Moondream (figure captions).
- Big re-ingest (WS-F) is LAST, after all WS code lands & tests pass; Haiku monitors per §10.

## Session 17 — 2026-05-30 — opus-4.8
**Commits:** `e3974fd`, `f17c169`, `af8ad50`, `1411efd`, `b5dec42`, `9cf7748`, `f9271da`, `74e98dc`, `b80757c`, `049b976`
**Achieved:**
- Finished + committed unlogged WIP from a token-limited session: hf-space **Corpus / Graph / Eval tabs** alongside Chat (`gr.Tabs()` in center pane), all over the existing FastAPI backend — no new ML in the Space.
- corpus_tab = retrieve+rerank search w/ PDF bbox preview; graph_tab = per-doc KG lookup via new `ApiClient.graph_lookup` → `GET /graph/{doc_id}`; eval_tab = live Recall@k/MRR/nDCG bench w/ embedded 4-query dataset (metrics inlined, no eval/ dep).
- Verified: 15 new tab tests green on host; `make_app()` builds all 4 tabs in the hf-space image (gradio 5.50.0, offline stub client).
- **Deployed live** to HF Space `ahmedsali/graphaero-rag`: stack up (backend healthz all-green), cloudflared tunnel → `BACKEND_URL` secret, uploaded hf_space/. Config now = 64 comps / 4 tabs (Chat·Corpus·Graph·Eval); `/graph/tsb/a13q0098` returns 23 findings / 5 regs.
- **Deploy gotcha found** (wasted one cycle): Space Dockerfile `COPY hf_space /app/hf_space` + `python -m hf_space.app` → only the `hf_space/` SUBDIR is build-authoritative. Must upload `path_in_repo="hf_space"`, NOT `"."`. Saved as project memory.
- **Undeployed** on request: Space PAUSED, tunnel killed; rebuilt local hf-space image + brought up at **http://localhost:7860** (4 tabs verified).
- **PDF highlight Phase 1 shipped** (`1411efd`): two-tier multi-occurrence highlight. `search_page_terms` boxes every on-page query-term hit (title + all mentions, light wash) under the solid cited box; `_query_terms` (EN+FR stopword filter) feeds them. Render-only, no re-ingest. Live-verified on tsb/a13q0098 p.3 (8 boxes incl. all 4 title words; title solid, body washed — screenshot inspected). +13 tests, full hf_space suite green in-image.
- **PDF highlight Phase 2a shipped** (`9cf7748`): red-box figures on cited pages — `page_image_bboxes` reads pdfplumber `page.images`; `render_page_with_bbox(box_images=True)` draws red outline (no fill). No image *understanding* (no VLM in budget) — just marks figures. Render-only, no re-ingest. Live-verified tsb/a13q0098 p.60 (both aerial photos boxed red). +4 tests.
- **3D embedding-space tab SHIPPED** (`74e98dc`): Plotly 3D scatter of BGE-M3 vectors. `build_embedding_space.py` scrolls Qdrant → UMAP(cosine, PCA fallback)→3D → baked `embedding_space.json` (12k-pt projection of live 63,946; tsb/tc, en/fr), `corpus` from doc_id prefix (ready for caac). `embedding_tab.py` = color-by corpus/lang, legend toggle, hover. Neo4j graph_tab KEPT alongside. plotly dep added. Static baked JSON (no runtime DB) = deployable. 67 tests, make_app builds all 5 tabs. Live render not browser-screenshotted (kaleido flaky; data+figure verified, user testing :7860).
- **Re-ingest program plan written** → `docs/REINGEST_PLAN.md` (self-contained brief for a fresh session): curation-first principle, locked decisions (Florence-2+Moondream, one tagged collection, word-grounded bboxes, ZH corpus, drop FR emphasis), ZH sourcing as the GATING unknown, arch changes by stage (pdf/ocr/chunk word index → embed→backend→hf_space passthrough; figures.py VLM; Neo4j :Figure; cross-corpus eval; CJK chunking), WS-A..F breakdown, run sequence, risks. Stack shut down + resources returned earlier this session.
**Left (user-directed roadmap, S17):**
- **The big re-ingest program** — see `docs/REINGEST_PLAN.md`. Start: WS-A (ZH recon) + WS-B (word-bbox on EN/TC), in parallel. Land ALL code before the overnight run.
- **About tab** — What/Why/How of the project. Authoring pre-approved by user. Not started.
- **PDF highlight Phase 2b** — scanned/image-only pages: persist per-line OCR bbox (ocr.py already computes it) through Chunk→chunk JSON→Qdrant payload→backend→`RetrievedChunk`, use in pdf_render when text-layer search is empty. Requires re-ingest (= `processing.run` picks up the ~670 image-heavy + new TC PDFs, idempotent, OCR runs anyway; +1 metadata field) then `embed.run`. User ready; land the chunk/ocr code BEFORE kicking off the hours-long run.
- Graph-native breadth A/B/C (Neo4j recurrence) — **parked** (user doesn't want it now).
- Stack UP (local :7860/:8080). Space PAUSED. `image1.png` untracked.

## Session 16 — 2026-05-29 — opus-4.8
**Commits:** `e8eb8c0`, `885dcda`, `a7f006e`, `b931ec2`, `3b6bc64`
**Achieved:**
- Root cause of "nothing changes after edits": app images bake source (no volume) → rebuilt backend+hf-space; all fixes verified live in browser
- hf-space UX: cross-doc cite-every-claim prompt, sources sorted high→low, lang/source filters wired, citation regex tolerates `§section` so bbox draws; instant cached examples (`build_sample_cache.py`→`sample_cache.json`, 6 queries); Source pages → collapsible full-width preview gallery+reel below composer; HITL gate removed (normal chat); Thought/Sources collapsed
- **Graph outward hop v1** (`3b6bc64`): concentration-gated `recurring_context_for_occurrences` on the direct `Occurrence→CITES→Regulation` edge; `graph_broaden` trace; rendered into synthesis. 109 graph+agent tests; live = 4 regs / 12 siblings for "engine failure after takeoff" (generic-CAR hubs filtered at deg>15)
**Left:** recurrence is reg-level-thin (`Finding→Regulation` edge only 1.5% filled, 166/10719) → next = **A** densify LLM finding-extraction (deferred data expansion, highest payoff), **B** cheap-enrich siblings with their own top finding text+page, or **C** fix gemma's uncited clarifying-question hedge on procedure-heavy bare-phrase queries (pre-existing, not the graph). Stack stopped to free CPU/RAM.

## Session 15 — 2026-05-29 — opus-4.7
**Commits:** `<this>` hf-space: citation-anchored bbox + inline page gallery (two UI pivots)
**Achieved:**
- **Pivot 1 — bbox:** highlights now anchored to the answer's citations (`_parse_citations` + `pdfplumber.Page.search` via `search_page_bbox`), **no box if the cited span isn't found**. Bypasses the mis-placed stored chunk bbox (root cause: `_join_pages` char→bbox desync — left for upstream fix). DEFAULT_DPI 120→100.
- **Pivot 2 — layout:** dropped the right "Sources" sidebar; source pages render as a zoomable `gr.Gallery` message **inline in the chat** after the answer (no token-stream blocking), chunk snippets folded into the "📑 Sources" accordion, dead `on_select` click-to-render removed. Bbox toggle re-renders the inline gallery.
- Verified: `gr.Gallery`-in-message postprocesses on gradio 5.50; `make_app` + full Blocks build clean in-container; hf-space tests green (+6 pdf_render tests).
- Documented both pivots + revert paths in `hf_space/README.md` ("Design notes — S15 pivots").
**Left:** human visual click-through at :7860 (highlight lands on cited sentence, lightbox zoom, toggle redraw) — NOT done this session (no browser MCP). Stack left UP. Real bbox fix still lives upstream in `ingestion/.../chunk.py::_join_pages`.

## Session 14 — 2026-05-28 — opus-4.7
**Commits:** `5d87bfc` Gradio 4→5.14+ upgrade, `50e7d2a` backend sources SSE event, `05d515f` gr.Chatbot three-zone UI rebuild, `616b15c` S14 handover, +S5 verify (this entry)
**Achieved:**
- Pivot UI from Streamlit back to canonical Gradio hf-space (:7860); rebuilt chat on gr.Chatbot three-zone layout (left/right Sidebars + center chat), collapsible thinking + sources accordions, streaming, UI-side HITL, right Sidebar Pages/Chunks tabs on chat.select
- Backend `/query/stream`: dedicated `sources` SSE event before tokens; dropped sources from `done`; +StubLLM.chat_stream + ordering test. 311 tests pass
- **S5 live-verified (stack up):** SSE order correct + `done` has no sources; real grounded cited answer (tsb/a03q0109); OTel spans flow w/ 0 errors today; :7860 serves gradio 5.50.0 w/ full component tree
- Flagged pre-existing (not mine): stale 05-27 reranker dtype + tokenizer "Already borrowed" errors under concurrent /query
**Left:** S6 = human visual click-through at :7860 (no browser connectable this session). Stack left UP — `docker compose stop` to free GPU/RAM.

## Session 13 — 2026-05-28 — sonnet-4.6
**Commits:** `089f3b7` chunk overlap+sections, `df96f5f` Streamlit chat app, `fbe7d6a` corpus+graph viewers
**Achieved:**
- North Star pivot: Streamlit chat app with RAG, replacing Next.js + Gradio
- Chunker: overlap 64→128 tokens; content-based section_title detection (TSB EN/FR + numbered AC sections); 310 tests pass
- Re-chunk started in background (`ingestion` container, ~45 min total, force-reprocess all 2895 PDFs)
- Streamlit UI: chat with HITL draft gate + x-ray expander, corpus viewer (/retrieve search), graph viewer (/graph/{doc_id})
- Backend: added GET /graph/{doc_id} endpoint
- docker-compose: `ui` service on port 8501 replaces frontend/hf-space
**Left:** re-chunk (correct image, running now) → re-embed → restart backend; add PDF bbox rendering to corpus viewer; live UI test still pending

## Session 12 — 2026-05-27 — opus-4.7 + haiku-4.5
**Commits:** none (recon only; artifacts in data/recon/tc/, cleanup scripts in repo root)
**Achieved:**
- Diagnosed P1b TC corpus gap: `en/tc/` empty because index page has no direct PDF links
- Haiku fetched EN+FR AC indexes + all 281 same-host links to `data/recon/tc/`
- Identified two-pass structure: index → per-AC detail pages (`/advisory-circular-ac-no-*`) → PDFs
- Confirmed `extract_pdf_urls` works correctly on detail pages (1 PDF found on sample AC page)
**Left:** Opus to author `extract_subpage_urls` + two-pass `run_tc` fix + tests; Haiku to run sample scrape and mark P1b complete

---

## Session 11 — 2026-05-27 — opus-4.7
**Commits:** (MANIFEST.md updated with S3 results, pending git commit)
**Achieved:**
- Monitored full-corpus embed completion: 63,946 points in Qdrant (from prior 54,280 + TC corpus new chunks)
- Ran post-fix bbox evaluation (S3): 50 TSB chunks sampled, mean_sim=0.267 (+30% vs pre-fix 0.205), hit_rate=30% (+50% vs 20% baseline)
- Analysis: Improvement confirmed; worst cases remain cross-page chunks with page markers ("- 2 -", "- 7") despite bbox fallback
- Updated MANIFEST.md with S3 results and three next-step options (further iteration, move forward with validation, frontend-layer fix)
**Left:** Commit updates; decide S4 direction (bbox iteration vs. retrieval validation); await user guidance on path forward

---

## Session 10 — 2026-05-27 — haiku-4-5-20251001
**Commits:** (none yet)
**Achieved:**
- Pre-flight verification: Docker daemon running, all services healthy (qdrant, postgres, neo4j, ollama, otel-collector, backend, frontend)
- Verified Qdrant `aerospace_dense` collection: 63,946 points (beyond expected ~5626), vectors dim 1024, Cosine distance
- Found 7,302 chunks across 2,878 JSONL files in data/chunks/ (more than prior summary indicated)
- Identified accelerate/peft dependency conflict in FlagEmbedding import chain (import error)
**Left:** Fix accelerate dependency; resume Block 2 (embed.run --limit 50, full embed, idempotency check); proceed with Blocks 3–9

---

## Session 9 — 2026-05-27 — sonnet-4.6
**Commits:** (none committed — .gitignore edit staged)
**Achieved:**
- Triage: "10k lines uncommitted" was untracked scratch dirs (logs/, crops/, scripts/) — actual diff only 182 lines across 2 eval files
- Confirmed ingestion container still alive (not cancelled — just detached when CLI closed); stack fully healthy
- Added .gitignore entries for logs/, crops/, scripts/, smoke_query.py to prevent binary/log bloat
- Moved session surface from VS Code CLI to desktop app to avoid Windows terminal issues
**Left:** commit .gitignore + eval files; wait for ingestion to finish; re-embed; post-fix bbox eval (target hit_rate >70%)

---

## Session 8 — 2026-05-27 — opus-4.7
**Commits:** (none — diagnostic/audit only)
**Achieved:**
- Diagnosed synthesis bug root causes: num_ctx unset (silent truncation), temperature unset (0.8 too loose), title-bait chunking, section_title garbage (55% running headers), doc_id EN+FR collision (70% docs)
- Test 1: num_ctx alone insufficient (hedge persists at both default & 8192). Test 2: document-anchored retrieval (full doc chunks + rerank + budget) produces real grounded multi-doc answer.
- Qdrant audit (63,896 points): confirmed 1,172/1,680 doc_ids carry both langs, section_title mostly dates/boilerplate, but highlights work via source_url (not broken as initially claimed)
- Key insight: re-ingest cheap (payload-only, no GPU) since text/vectors unchanged. Revised plan: query-time fixes (num_ctx, anchor node, tight bbox) + cheap section_title rebuild
- Created session handoff prompt for next session; one open decision (doc_id+lang cosmetic vs defer)
**Left:** Haiku to author code (num_ctx, expand node, section_title, bbox, tests); decide doc_id fork; execute re-process + payload-update

---

## Session 7 — 2026-05-27 — opus-4.7
**Commits:** `65fb7e5` (Baseline: P1b acquisition + P1 ingestion)
**Achieved:**
- P1b: TC two-step crawl (index→detail→PDF) + TSB scraper; 35/35 tests pass offline
- P1: pdfplumber text+tables+bbox; PaddleOCR fallback; 512-token chunking with 64 overlap; 33/33 tests
- End-to-end verified: 2,183 chunks across 114 PDFs; cross-doc dedup, section-title, bbox aggregation
- Chunk schema locked: {doc_id, source_url, section_title, page, bbox, chunk_hash, lang, text}
**Left:** P2 plan drafted; Chat-Opus to author embed module (backend/embed/, Qdrant, BGE-M3)

---

## Session 6 — 2026-05-27 — sonnet-4-6
**Commits:** `b4095e7` (CORS fix), `63df76e` (eval pdfplumber refactor + .gitignore)
**Achieved:**
- Fixed CORS preflight (OPTIONS 405 → 200); frontend at localhost:3000 connects to backend
- Confirmed full pipeline live: /healthz ✅ /retrieve ✅ agent (embed→rerank→gemma2 draft) ✅
- Force-reprocessed all 2878 chunk files with the bbox fallback fix from Session 5
- Refactored bbox_eval.py to use pdfplumber crop (no OCR needed); committed
- Kicked off re-embed to Qdrant with fixed-bbox chunks (still running as of session end)
- Created SESSIONS.md, fixed MANIFEST resume pointer, added session discipline to CLAUDE.md
**Left:** post-fix bbox eval once embed finishes (target hit_rate >70% vs 20% baseline)

---

## Session 5 — 2026-05-27 — sonnet-4-6
**Commits:** `d49feae` (manifest: smoke-pass + bbox eval baseline), `2e6a8e2` (chunk.py bbox fallback), `25fa56b` `b4cebc5` (manifest updates)
**Achieved:**
- Root-caused cross-page bbox problem: tiny bbox area (<5000 pt²) from page-number-only chunks
- Added `MIN_USABLE_BBOX_AREA` fallback in `_bbox_for_range()` + 1 new test (9 total in test_chunk.py)
- Ran baseline bbox eval: 20% hit rate pre-fix (confirmed page-number worst cases "- 3 -", "- 4 -")
- Launched --force reprocessing of all 2892 docs; monitored loop confirmed completion
- Added pdfplumber crop eval strategy to eval/bbox_eval.py
**Left:** re-embed fixed chunks, post-fix bbox eval

---

## Session 4a — 2026-05-26 — sonnet-4.6
**Commits:** `3927684` (GPU passthrough docs), `98cba99` (nDCG dedup), `1dedddc` (PostgresSaver CM), `49e4928` (HF Space: hub<1.0 + bool-schema), `216099a` (HF Space: starlette<1.0)
**Achieved:**
- Smoke-pass blocks 2–8 driven end-to-end: full embed (54,280 pts, dim 1024 Cosine on 3060Ti), EN+FR retrieve smoke, eval Recall@5=MRR=nDCG@5=1.0 on all 4 queries, `/healthz`+`/retrieve`+`/query`+`/resume` verified with OTel spans, frontend `:3000` 200, HF Space `:7860` 200
- Fixed 5 real bugs found by running it: nDCG>1.0 (doc_id dedup), backend `/query` AttributeError (PostgresSaver CM lifecycle), HF Space 3 startup crashes (huggingface_hub 1.0 `HfFolder` import, gradio_client bool-schema `TypeError`, starlette 1.x `TemplateResponse` API break)
- Documented MSYS path-mangling gotcha (Bash tool rewrites `/app/...` → Windows path; use PowerShell for container-absolute paths) in `memory/`
- 222 pytest + 18 Vitest pass (was 221; +1 nDCG dedup regression test)
**Left:** visual gallery/PDF-bbox test (no browser MCP this session)

---

## Session 4 — 2026-05-26/27 — opus-4.7 + sonnet-4.6 + haiku-4.5
**Commits:** `e4c3c89` (haiku smoke-pass status), `3c69b21` (qdrant healthcheck fix), `cc950a3` (backend Dockerfile -r layout fix), `d882e1e` (manifest handoff snapshot), `9a216c5` (GPU passthrough), `e8767c7` (graph hybrid extractor), `900e637` (thread-safe lazy loader), `f632f91` (bbox eval + OCR fix), and others
**Achieved:**
- Full smoke-pass: all 9 live blocks (corpus, embed, retrieve, graph, eval, backend, frontend, HF Space, tests)
- Graph populated: 1,199 Occurrences + 13,124 Findings + 2,136 Recommendations via regex+LLM extractor
- GPU passthrough enabled; full embed 54,280 pts in ~26 min (was CPU-only, would have taken hours)
- Eval: Recall@5=1.0, MRR=1.0, nDCG@5=1.0 on all 4 queries
- HF Space published + cloudflared tunnel; end-to-end verified from public URL
- TC corpus completion: fixed ConnectionError catch; pulled 367 new PDFs
- 307 pytest + 18 Vitest passed (all mocked/offline)
**Left:** LLM extraction (--extract) running in background (PID 27324); TC corpus not yet re-chunked/embedded

---

## Session 3 — 2026-05-26 — opus-4.7
**Commits:** `d99e357` (HF Space tier-2 redesign), `d9f0828` (manifest)
**Achieved:**
- Tier-2 three-pane Gradio redesign: left sidebar, chat-style HITL center, sources/trace/logs right rail
- Designed and committed UI mockups in mockups/
- Generation-quality fix: anchored retrieval (full doc chunks) + num_ctx=8192 for gemma2
- Corpus: 2,895 PDFs on disk (TSB 100%, TC partial)
**Left:** redeploy HF Space, live backend rebuild with anchored retrieval

---

## Sessions 1–2 — 2026-05-22–25 — opus-4.7 + haiku-4.5
**Commits:** P0–P9 delivery commits
**Achieved:** Full project build — P0 scaffold through P9 docs. All 9 phases shipped offline (221 pytest + 18 Vitest). Architecture locked: Qdrant/BGE-M3/reranker-v2-m3/gemma2:9b/Neo4j/Postgres/FastAPI/Next.js/Gradio.
**Left:** live smoke-pass (no real corpus, no running stack yet)
