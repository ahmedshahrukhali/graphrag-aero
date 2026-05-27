# Session Log

One entry per conversation. Most recent at top. Keep each entry under 10 lines.

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
**Commits:** `63df76e` (eval pdfplumber refactor + .gitignore), `b4095e7` (CORS fix)
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
**Commits:** `2e6a8e2` (chunk.py bbox fallback), `b4cebc5` `25fa56b` `d49feae` (manifest updates)
**Achieved:**
- Root-caused cross-page bbox problem: tiny bbox area (<5000 pt²) from page-number-only chunks
- Added `MIN_USABLE_BBOX_AREA` fallback in `_bbox_for_range()` + 1 new test (9 total in test_chunk.py)
- Ran baseline bbox eval: 20% hit rate pre-fix (confirmed page-number worst cases "- 3 -", "- 4 -")
- Launched --force reprocessing of all 2892 docs; monitored loop confirmed completion
- Added pdfplumber crop eval strategy to eval/bbox_eval.py
**Left:** re-embed fixed chunks, post-fix bbox eval

---

## Session 4 — 2026-05-26/27 — opus-4.7 + sonnet-4.6
**Commits:** `e8767c7` (graph hybrid extractor), `900e637` (thread-safe lazy loader), `f632f91` (bbox eval + OCR fix), `9a216c5` (GPU passthrough), and others
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
