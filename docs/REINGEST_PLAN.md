# Re-ingest Program — Plan

**Author:** opus-4.8 (S17, 2026-05-31) · **Status:** APPROVED scope, NOT STARTED
**Purpose:** a self-contained brief so a fresh session can execute the "big re-ingest"
without re-deriving context. Read this + `MANIFEST.md` resume pointer, then start at §6.

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

## 2. The hard unknown — Chinese corpus sourcing (GATING; do first)

Recon (S17) found **no clean public CAAC PDF index** like TSB's. This is the single biggest
risk and **gates the ZH half of the program**. First deliverable of a fresh session:

- **Recon** reachable, license-clear Chinese aviation-safety documents. Candidates to probe:
  CAAC Chinese site (`caac.gov.cn` 中文), CAAC monthly/annual safety bulletins, Aviation
  Safety Network (ASN) Chinese entries, ICAO accident report repository, HK CAD / Macau,
  academic / Kaggle datasets, university aviation-safety archives.
- **Output:** a target set + a scraper modeled on `ingestion/acquisition/` (rate-limited,
  robots-respecting, idempotent), writing to `data/corpus/zh/<source>/`.
- **Decision gate:** if no adequate aviation ZH corpus is reachable, surface fallback options
  (an adjacent CJK-heavy safety/technical domain) and **ask** — do **not** silently substitute.
  The whole "similar docs, different language" story depends on topical similarity to EN/TC.
- The EN/TC word-bbox + figure work (§4.1–4.3) is **independent of ZH** and can land first.

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

### 4.1 Word/pixel-grounded bboxes  *(fixes the S15 desync at root)*
- `ingestion/processing/pdf.py` — `PageExtract` has per-char `Char`. **Add word extraction**
  via pdfplumber `page.extract_words()` → per-word `(text, x0, top, x1, bottom, page)`.
- `ingestion/processing/ocr.py` — `ocr_page` currently keeps one `Char` per **line**. Switch
  to **`return_word_box=True`** (per-char/word boxes) — or Florence-2 `OCR_WITH_REGION` — and
  emit word-level entries in the same PDF-point space (the `_PTS_PER_PIXEL` conversion stays).
- `ingestion/processing/chunk.py` — `Chunk` dataclass gains a compact **word index**:
  `word_boxes: tuple[(text, page, x0, top, x1, bottom), ...]` for the words in the chunk.
  `_join_pages`/`_bbox_for_range` already map char→page→bbox; extend to carry word spans.
  This replaces brittle char-alignment + the render-time `page.search` hack.
- **Schema passthrough** (the chain confirmed this session):
  `embed/jsonl.py` `ChunkRecord` + `.payload()` → Qdrant payload → retrieve `r.record` →
  `backend/schemas.py:RetrievedChunk` (`bbox`, `section_title`, …) → `hf_space/api_client.py:RetrievedChunk`.
  Add `word_boxes` at every hop.
- `hf_space/pdf_render.py` — when highlighting, **map the cited span / query terms to the
  stored word boxes directly** (no `page.search`). Keep render-time search as a fallback for
  legacy points without `word_boxes`. Scanned pages become highlightable from stored OCR boxes.
- **⚠ Payload size decision:** word boxes per chunk are bulky. Options: (a) store all in
  payload (simplest, fattens Qdrant), (b) store a per-page word sidecar keyed by doc+page and
  join at render, (c) store only the chunk's own words. Pick before WS-B; (c) is the lean default.

### 4.2 Image understanding — Florence-2 + Moondream2
- New `ingestion/processing/figures.py`: detect figures (`page.images`), and per figure run
  **Florence-2** (`OCR_WITH_REGION` + region caption) and **Moondream2** (prose blurb).
  Emit `Figure` records `{doc_id, page, bbox, caption, ocr_text}`.
- Deps: add Florence-2 + Moondream to the **ingestion** image (torch already present;
  `trust_remote_code`/revision pinning needed — verify). Sequential load fits 8GB; offline.
- Outputs: **(a)** Neo4j `Figure` nodes (§4.3); **(b)** the blurb **embedded as a chunk**
  (tag `kind=figure`) so figures become *retrievable*, not just graph decoration.

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
- Carrying the word index changes the chunk schema.
- **CJK token density:** BGE-M3's tokenizer handles Chinese, but 512 tokens of zh ≈ much more
  text than 512 of en — consider **language-aware windowing** or sentence/layout-aware
  boundaries. Keep fixed-512 as baseline; **evaluate** before committing a change (don't
  regress the EN eval).

---

## 5. Test strategy (CLAUDE.md: offline, mocked, no weight downloads)
- Mock Florence-2 / Moondream2 / PaddleOCR / pdfplumber in all unit tests.
- Pure-unit-test the new logic: word-box extraction/alignment, chunk word index, figure-record
  shaping, payload passthrough, render-from-stored-boxes, cross-corpus metric.
- **Live verification on a small curated sample** (a handful of EN + scanned + ZH docs) BEFORE
  the full overnight run — render a page from stored boxes, check a figure caption, eyeball the
  3D overlap on the sample.

---

## 6. Work breakdown (each WS lands its own commit; ALL code lands & tests pass BEFORE §7 run)

- **WS-A — ZH corpus recon + scraper** *(gating for ZH; start immediately, parallel)*.
- **WS-B — word-bbox capture + passthrough + render** *(independent of ZH; land on EN/TC first)*.
- **WS-C — figure VLM module + Figure nodes + figure-chunk embedding**.
- **WS-D — curation: admission filters, dedup tuning, curation manifest**.
- **WS-E — dual-corpus tagging + cross-corpus eval metric + 3D viz refresh hooks**.
- **WS-F — the re-ingest run** (§7).

Recommended start: **WS-A (recon) + WS-B (word-bbox on EN/TC)** in parallel — WS-B proves the
grounded-bbox architecture without waiting on the ZH source.

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

## 9. Risks / open decisions (resolve early)
1. **ZH corpus reachability** (§2) — biggest; gates half the program.
2. **Qdrant payload size** from word boxes (§4.1) — pick storage strategy (lean default: chunk's own words).
3. **Florence-2 `trust_remote_code` / revision pinning**; Moondream revision pinning.
4. **Chunking change vs EN eval regression** — measure before changing windowing.
5. **License/robots** for the ZH source.
6. **Re-ingest idempotency** for figures (key by doc+page+bbox) so re-runs don't duplicate.
