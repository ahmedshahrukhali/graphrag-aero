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
| P2 | Embed: BGE-M3 dense → Qdrant | ☐ |
| P3 | Retrieve + rerank: Qdrant ANN + bge-reranker-v2-m3 | ☐ |
| P4 | Graph+agents: Neo4j schema, LangGraph, PostgresSaver, HITL (final answer gate + trace) | ☐ |
| P5 | Eval: Recall@k / nDCG / MRR | ☐ |
| P6 | Backend: FastAPI, OTel, Ollama | ☐ |
| P7 | Frontend: Next.js + TS, PDF highlight | ☐ |
| P8 | HF Space: Gradio multimodal, EN+FR | ☐ |
| P9 | Docs | ☐ |

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
**NEXT: P2** — embed: BGE-M3 dense → Qdrant. Read `data/chunks/**/*.jsonl`, embed `text`, upsert into Qdrant collection with payload `{doc_id, source_url, section_title, page, bbox, chunk_hash, lang}`. Sequential VRAM (load BGE-M3 → embed batch → unload).

**Done so far:**
- P0 ☑ scaffold
- P1b ☑ (by opus-4.7) — 35 acquisition tests pass; 110 TSB + 4 TC PDFs on disk
- P1 ☑ (by opus-4.7) — 33 processing tests pass; 1284 chunks across 114 docs written to `data/chunks/`; chunks at median 512 BGE-M3 content tokens (max 513)

**Queued for Haiku (mechanical, no logic authoring):**
- `python -m ingestion.acquisition.run --source tsb` (resume partial pull; 1318 IDs total, ~110 done)
- `python -m ingestion.acquisition.run --source tc` (full 243 ACs × EN+FR)
- Re-run `python -m ingestion.processing.run` after Haiku bulk pull to expand chunks.
- `docker compose --profile ingest build ingestion` (verify Dockerfile build when Docker daemon is up).
