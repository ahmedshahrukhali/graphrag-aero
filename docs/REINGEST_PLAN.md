# Re-ingest Program — Plan

**Author:** opus-4.8 (S17, 2026-05-31) · **Revised:** opus-4.8 (S18, 2026-05-31) · **Status:** APPROVED scope + ZH sources, NOT STARTED
**Purpose:** a self-contained brief so a fresh session can execute the "big re-ingest"
without re-deriving context. Read this + `MANIFEST.md` resume pointer, then start at §6.

**S18 revision:** ZH sourcing RESOLVED (§2 — two verified axes, approved). Work re-sequenced
around a write-shape freeze (§6 — new WS-0). **Bbox approach RESET (§4.1): word-level highlighting
scrapped; grounding is now region-level from the chunk's own stored bbox — kills the S15 desync,
the word-box payload, and per-word OCR.** Model swap under evaluation (§4.6): gemma2:9b → Qwen3-8B
generation + Qwen3-VL-8B on the figure tier, both decided by bake-off. Overnight-run monitoring
runbook for Haiku added (§10).

---

## 0. North Star & the curation principle

We are rebuilding the corpus to make the demo *show* three things, all grounded in real data:
1. **Pixel-grounded highlighting** — bboxes anchored from ingestion (incl. scanned pages),
   not re-searched at render time (the S15 hack desyncs).
2. **Figure understanding** — every figure red-boxed *and* captioned by a VLM, with the
   caption living in the knowledge graph and retrievable.
3. **Dual cross-lingual corpus** — an EN/TC + a Chinese corpus in one shared BGE-M3 space,
   visibly overlapping where they agree (consistency) and extending where each is unique.

**Curation principle (user-stated):** *the visual aids are for the demo; the data must be
well curated.* Quality over volume. Curation is a first-class workstream (§3), not cleanup.
We would rather ingest a smaller, clean, balanced corpus that demos beautifully than dump
everything. Every doc admitted should be justifiable.

---

## 1. Locked decisions (from the S17 discussion — do not relitigate)

| Decision | Value | Why |
|---|---|---|
| Image→text models | **Florence-2 + Moondream2** | Florence-2 (230M/770M) = OCR + `OCR_WITH_REGION` quad boxes + region caption in one model; Moondream2 (~1.9B, 4-bit ≈2.4GB) = richer prose blurb. |
| OCR word boxes | PaddleOCR `return_word_box=True` (per-char/word boxes — confirmed) and/or Florence-2 `OCR_WITH_REGION` | grounds scanned-page highlights |
| Born-digital word boxes | pdfplumber `page.extract_words()` (native per-word x0/top/x1/bottom) | grounds text-page highlights |
| Vector store layout | **one collection, `corpus` payload tag** (`tsb`/`tc`/`caac`) | overlap demo is trivial in shared space; filter to separate |
| Embeddings | BGE-M3 (unchanged) — XLM-R, 100+ langs, shared cross-lingual space | overlap is free from the model |
| Languages | Add **Chinese**; drop FR *emphasis* (FR still embeds fine, was only a cross-lingual proof) | new axis = EN/TC ↔ ZH |
| Where VLM/OCR runs | the **isolated ingestion image**, **offline batch** | no query-time VRAM contention with gemma2:9b |

---

## 2. Chinese corpus sourcing — RESOLVED (S18, approved 2026-05-31)

S17 feared no clean public CAAC index. S18 recon found **two clean, enumerable ZH sources**,
each a topical twin of an existing corpus half. **Approved: ingest both, ACs as the spine.**
The §2 gate is closed; the EN/TC word-bbox + figure work (§4.1–4.3) is still independent of ZH
and can land first regardless.

### Axis 1 (spine) — Advisory Circulars: CAAC 咨询通告 ↔ Transport Canada AC
- Primary government PDFs: `https://www.caac.gov.cn/XXGK/XXGK/GFXWJ/{YYYYMM}/P{…}.pdf`.
- Same *genre* + same `AC-xxx` numbering as TC ACs (airworthiness / maintenance / flight
  standards) → the "same docs, different language" overlap is honest, not forced.
- **Enumerable:** GFXWJ monthly folders + the self-describing catalog AC `AC-01-AA-2017-01R26`
  ("发布的适航规章及规范性文件目录"). Live 2026 docs confirmed (`AC-21-AA-2026-44/45`).
- Many older ACs are scanned typeset PDFs → exercises the OCR word-box path (§4.1) on a
  non-Latin script (a feature, not a problem).
- **License/robots: GREEN.** caac.gov.cn robots.txt disallows only `/CAAC/local/` and `/image/`;
  `/XXGK/` and `/GFXWJ/` permitted, no crawl-delay. PRC public gov documents. Rate-limit anyway.

### Axis 2 (second stratum) — Investigation reports: CAAC ↔ TSB
- The twin of the TSB occurrence half. Chinese reports: 民用航空器事件调查报告 /
  航空器严重征候调查报告 (e.g. Bell 407 CFIT; Sichuan 3U8633 windscreen).
- **License/robots: AMBER → ASN is INDEX-ONLY.** aviation-safety.net signals `ai-train=no` and
  disallows AI-training bots (ClaudeBot). So: use ASN's China profile
  (`/database/country/country.php?id=b`) to *discover* occurrences, then pull the actual PDFs
  from their **primary host** (caac.gov.cn TZTG announcements, regional CAAC bureaus, original
  issuer). Reports that exist *only* on ASN → **skip or ask**; do not bulk-fetch PDFs from ASN.

### Scraper
Model on `ingestion/acquisition/` (rate-limited, robots-respecting, idempotent) →
`data/corpus/zh/{ac,reports}/`. Note: asn.flightsafety.org's cert chain fails strict
verification — the httpx scraper must handle that explicitly (don't disable verification
globally; scope it to that host).

---

## 3. Data-curation workstream (cross-cutting, gated quality)

Define and enforce admission criteria; emit a **curation manifest** (counts per
corpus/lang, born-digital vs OCR, figures captioned, rejects + reasons).

- SHA-256 dedup already exists (`ingestion/processing/dedup.py`) — keep, tune.
- Reject: empty/failed scans, cover-only or boilerplate-only docs, non-report pages,
  language-misdetected docs, sub-threshold content length.
- Balance: keep the corpus topically comparable across EN/TC and ZH so the overlap demo is
  honest (display is tag-stratified anyway, but the underlying set should be balanced).
- Provenance: every chunk already carries `{doc_id, source_url, page, lang, ...}`; add
  `corpus` + (for figures) figure provenance.

---

## 4. Architecture changes by stage (the spine)

### 4.1 Region-level grounding — RESET (S18; word-level highlighting scrapped)

**Decision (opus-4.8, S18, user gave free rein):** stop chasing word-level pixel highlighting.
Ground citations at the **region/block level** using the chunk's *own* stored bbox. Everything
painful — the S15 `page.search` desync, the word-box payload, per-word OCR quads, render-time
re-search, char-alignment — exists only to highlight an exact sentence. That feature is not worth
its fragility for a demo. Region-level "here is the page, and the block we pulled this from" is
what most production cited-RAG UIs actually do, it's deterministic, and it never desyncs.

**The principle:** grounding is the rectangle the chunk *occupies on the page*, computed once at
ingest and rendered directly. No re-search, no word index.

- `ingestion/processing/chunk.py` — the chunk already gets its region via `_bbox_for_range`
  (char-range → page → bbox). Keep exactly that. A chunk can span pages, so store a small
  **per-page region list**: `page_bboxes: tuple[(page, x0, top, x1, bottom), ...]` (one rect per
  page the chunk touches). That is the *entire* grounding payload — bounded, tiny, deterministic.
- `ingestion/processing/ocr.py` — **no per-word boxes needed.** OCR (PaddleOCR / Florence /
  Qwen-VL — see §4.2 bake-off) only has to produce *text* + enough layout to place the chunk's
  region. Scanned and born-digital pages ground identically: a rect per page.
- **Schema passthrough:** `embed/jsonl.py ChunkRecord → Qdrant payload → backend/schemas.py
  RetrievedChunk → hf_space/api_client.py RetrievedChunk` already carries `bbox`. Generalize that
  one field to `page_bboxes` at every hop. No `word_boxes` anywhere — the (c) payload debate is
  **moot and removed**: there are no word boxes to store.
- `hf_space/pdf_render.py` — render the page image and draw the stored region rectangle(s).
  Delete the `page.search` path entirely (no fallback needed — every chunk has its region from
  ingest). This collapses WS-B to roughly: carry `page_bboxes` through + draw a rect.
- Figures (§4.2) keep their own coarse region box from detection — same tier, consistent UX.

### 4.2 Image understanding — Florence-2 + Moondream2
- New `ingestion/processing/figures.py`: detect figures (`page.images`), and per figure run
  **Florence-2** (`OCR_WITH_REGION` + region caption) and **Moondream2** (prose blurb).
  Emit `Figure` records `{doc_id, page, bbox, caption, ocr_text}`.
- Deps: add Florence-2 + Moondream to the **ingestion** image (torch already present;
  `trust_remote_code`/revision pinning needed — verify). Sequential load fits 8GB; offline.
- Outputs: **(a)** Neo4j `Figure` nodes (§4.3); **(b)** the blurb **embedded as a chunk**
  (tag `kind=figure`) so figures become *retrievable*, not just graph decoration.
- **Open bake-off (§4.6):** one `qwen3-vl:8b` may replace *both* Florence-2 + Moondream2 on the
  **figure tier** (region detection + caption + figure-internal OCR, incl. Chinese). Coarse figure
  boxes are fine for a VL model; decide by measurement, not specs. Word-tier OCR is gone (§4.1).

### 4.3 Graph (Neo4j)
- `graph/schema.py` — add `:Figure` constraint + `(:Occurrence)-[:HAS_FIGURE]->(:Figure)`
  with `{caption, ocr_text, page, bbox, doc_id}`. (Existing labels: Occurrence, Aircraft,
  Finding, Recommendation, Regulation, AC.)
- `graph/extract.py` + `graph/upsert.py` — figure upsert from the Figure records; key figures
  idempotently by `(doc_id, page, bbox)`.

### 4.4 Dual corpus + overlap demo
- Make `corpus` a **first-class chunk/payload field** (today it's derived from the `doc_id`
  prefix only in the 3D-viz build script).
- `hf_space/build_embedding_space.py` already colors by `corpus`; **rebuild
  `embedding_space.json`** after re-ingest so `caac` appears and the overlap shows.
- **New cross-corpus metric** in `eval/`: cross-corpus nearest-neighbor similarity +
  cross-lingual retrieval recall (EN query → ZH hits and vice versa). This *measures* the
  "knowledge extension," not asserts it (matches the grounding/measure discipline).

### 4.5 Chunking strategy (it will change again)
- Current: fixed 512 tok / 128 overlap + content-based section titles (S13).
- Schema change is now just `page_bboxes` (§4.1), not a word index.
- **CJK token density:** BGE-M3's tokenizer handles Chinese, but 512 tokens of zh ≈ much more
  text than 512 of en — consider **language-aware windowing** or sentence/layout-aware
  boundaries. Keep fixed-512 as baseline; **evaluate** before committing a change. (Search-quality
  tuning deferred to a later session — user, S18.)

### 4.6 Model swap — under evaluation (S18; relitigates a CLAUDE.md locked decision)
Two independent swaps, decide each by measurement on *our* docs (not benchmarks):
- **Generation `gemma2:9b` → `Qwen3-8B`.** Better-justified now the corpus is bilingual: Qwen3 is
  markedly stronger on Chinese/CJK, and the agent will synthesize answers over ZH ACs.
  **VRAM caveat:** Qwen3-8B Q4_K_M benches ~7–9 GB vs gemma2's ~5.5 GB — the §VRAM plan budgets a
  6.5 GB peak on the 8 GB 3060Ti. **Measure the real quant footprint in WS-0**; may need Q4_K_S or
  to confirm it's sole-resident at generation time (reranker/embed already unloaded).
- **`Qwen3-VL-8B` on the figure tier** (§4.2) — collapses Florence-2 + Moondream2 into one model.
  Released 2025-10-15, on Ollama (`qwen3-vl:8b`); 32-lang OCR, robust to blur/tilt, normalized 2D
  grounding. Good fit for *coarse* figure boxes; **not** for word-level (we scrapped that anyway).
  Throughput: an 8B VL per *figure* is fine (sparse); never per page.
- **Bake-offs:** (a) gemma2 vs Qwen3-8B answer quality on the same EN+ZH docs; (b) Qwen3-VL caption
  vs Moondream on sample figures. Fold (a) into WS-0, (b) into WS-C.
- **Possible future capability (not scoped):** a VL generator could *look at* the cited page/figure
  at answer time — multimodal grounding right at the HITL gate.

---

## 5. Test strategy (CLAUDE.md: offline, mocked, no weight downloads)
- Mock Florence-2 / Moondream2 / PaddleOCR / pdfplumber in all unit tests.
- Pure-unit-test the new logic: `page_bboxes` capture (§4.1), figure-record shaping, payload
  passthrough, region-rect render, cross-corpus metric.
- **Live verification on a small curated sample** (a handful of EN + scanned + ZH docs) BEFORE
  the full overnight run — render a page with its region rect, check a figure caption, eyeball the
  3D overlap on the sample.

---

## 6. Work breakdown — re-sequenced (S18; each WS lands its own commit; ALL code lands & tests pass BEFORE §7 run)

**Ordering principle:** WS-F (the re-ingest) runs **once, overnight**, so every decision that
changes what gets written to disk / Qdrant / Neo4j must be **frozen before it**. Sequence by
"freeze the write-shape first," not by feature. This repo is single-model and session-budgeted
(CLAUDE.md) — there is no real parallelism, so the only genuinely riskable item (ZH source) is a
**fail-fast spike**, not a co-equal parallel track.

- **WS-0 — Freeze the write-shape** *(FIRST; design lock; cheap)*. One migration covering every
  field that touches the `embed/jsonl.py ChunkRecord → Qdrant payload → backend/schemas.py
  RetrievedChunk → hf_space/api_client.py RetrievedChunk` chain — done **once** so the chain isn't
  re-edited 3–4 times: `page_bboxes` (region-level grounding, §4.1 — **not** word boxes),
  `corpus` tag (`tsb`/`tc`/`caac`), figure-chunk fields + a `kind` discriminator (§4.2), and
  whether the chunking window changes (§4.5). Also **lock the curation admission criteria** (§3)
  here — they decide what's *in* the
  corpus, so they are a write-shape decision, not trailing cleanup. Output: one frozen
  `ChunkRecord` dataclass + the decisions written down. Nothing in B/C/E touches code until set.
- **WS-A — ZH source spike** *(early; fail-fast)*. Stand up the `data/corpus/zh/` scraper: Axis 1
  (caac.gov.cn ACs, GREEN) + Axis 2 via ASN-as-index → primary PDFs only (AMBER, §2). Goal: prove
  the feeds return real documents and surface any blocker to the §2 ask-gate *early*. Once the
  feeds are proven, the bulk pull is Haiku-grunt.
- **WS-B — region-grounding render** *(against the frozen schema; EN/TC first)*. Per the §4.1
  reset: carry `page_bboxes` through the passthrough chain and draw the chunk's region rect on the
  page in `hf_space/pdf_render.py`; **delete the `page.search` path**. Much smaller than the old
  word-box plan. Proves the grounding UX without waiting on ZH.
- **WS-C — figures** *(depends on WS-B)*. Florence-2 + Moondream2 module; `:Figure` nodes;
  figure-blurb-as-chunk. Reuses B's stored-box render path for red-boxing — **that dependency is
  why C follows B**, not the reverse.
- **WS-E — dual-corpus + cross-corpus eval** *(after ZH lands)*. `corpus` tagging already frozen
  in WS-0; add the cross-corpus NN-similarity + cross-lingual recall metric; wire the 3D-viz refresh.
- **WS-D — curation manifest emission** *(criteria already locked in WS-0)*. The manifest itself
  is emitted during the run (§7).
- **WS-F — the single re-ingest run** (§7; last). Haiku monitors overnight — see **§10**.

---

## 7. The re-ingest run (LAST; only after all code is in & tested)
Sequence, idempotent, curated, ~overnight:
1. `ingestion.acquisition.run` for ZH (+ finish any TC gaps) → `data/corpus/`.
2. `ingestion.processing.run` — chunks **with** word boxes; figures captioned (Florence+Moondream).
   Idempotent via `chunk_hash`; figures keyed by `(doc_id, page, bbox)`.
3. `embed.run` — embed chunks (incl. figure-blurb chunks) into the **tagged** collection.
4. `agent.run upsert-graph` — Finding/Rec/Reg/AC **+ Figure** nodes.
5. Rebuild `hf_space/embedding_space.json`; run cross-corpus eval; redeploy Space.
Cost: OCR + Florence + Moondream per image + full re-embed = hours. Background run.

---

## 8. Definition of done / demo deliverables
- Highlights grounded from stored word boxes, **including scanned pages**.
- Figures **red-boxed + captioned** (VLM blurb on hover / in graph); `Figure` nodes queryable.
- Chinese corpus ingested + tagged; **3D tab shows EN/TC ↔ ZH overlap**; cross-corpus metric reported.
- **Curation manifest** committed (what's in, what was rejected and why).
- Full offline test suite green; small-sample live verification done before the big run.

---

## 9. Risks / open decisions
1. ~~**ZH corpus reachability**~~ — **RESOLVED** (§2): two verified axes; AC axis GREEN, report
   axis via ASN-as-index → primary PDFs.
2. ~~**Qdrant payload size from word boxes**~~ — **MOOT** (§4.1 reset): no word boxes; grounding
   is one region rect per page the chunk touches.
3. **ASN `ai-train=no`** (§2) — report axis must be index-only, primary PDFs; never bulk-fetch
   PDFs from asn.flightsafety.org.
4. **Qwen3-8B VRAM** (§4.6) — ~7–9 GB Q4_K_M may exceed the 6.5 GB peak budget on the 8 GB card;
   measure in WS-0 before committing the generation swap. *(open)*
5. **Florence-2 vs Qwen3-VL on the figure tier** (§4.2/§4.6) — decide by bake-off;
   `trust_remote_code` / revision pinning for whichever wins. *(open)*
6. **Chunking change vs EN eval** — measure before changing windowing (§4.5); tuning deferred
   (user, S18). *(open, low priority)*
7. **Re-ingest idempotency** for figures (key by doc+page+bbox) so re-runs don't duplicate. *(open)*

---

## 10. Overnight run — Haiku monitoring runbook (WS-F)

The §7 run is launched by a senior model **only after all WS code is in & tested**. Haiku does
**not** author or fix logic — it monitors, retries mechanically, and reports. During the run:

- **Watch stage transitions** in the run log: acquire → process (chunks + word boxes) → figures
  (Florence + Moondream) → embed → graph upsert → 3D-viz rebuild → cross-corpus eval. Confirm each
  stage starts and the process stays alive.
- **Heartbeat** every ~20–30 min: record progress (docs processed, chunks embedded, figures
  captioned, current corpus/lang) to the run log / a SESSIONS.md scratch line.
- **Resource guard:** confirm **sequential** VRAM — only one of OCR / Florence / Moondream / embed
  loaded at a time (CLAUDE.md VRAM plan). Flag if peak risks the 8 GB budget.
- **Idempotency = safe restart:** the run is resumable (chunk_hash dedup; figures keyed by
  doc+page+bbox). If a stage dies, **re-run that stage** — do not edit code.
- **Mechanical retries are fine:** network blips, failed downloads, disk, container restarts.
- **Stop & QUEUE (do NOT fix):** any exception in `.py` logic, a schema mismatch, OCR producing
  no word boxes, or a garbage figure-caption pattern → capture the traceback, stop the affected
  stage, and queue with the `⛔ NEEDS SONNET 4.6+` banner (CLAUDE.md degraded mode).
- **On completion:** verify the **curation manifest** emitted, `hf_space/embedding_space.json`
  rebuilt with `caac` present, and the cross-corpus metric printed. Log a SESSIONS.md entry.
  Do **not** redeploy the Space or mark MANIFEST ☑ — leave those for senior review.
