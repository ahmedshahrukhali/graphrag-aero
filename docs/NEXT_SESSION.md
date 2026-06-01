# Next-session plan — WS-A + WS-B (and the Qwen3-8B bake-off)

**Author:** sonnet-4.6 (S19, 2026-05-31). **Status of the re-ingest program:** WS-0 ☑ fully closed.
Read this + `MANIFEST.md` resume pointer + `docs/REINGEST_PLAN.md` §6, then start at WS-A or WS-B
(they're independent; WS-B is lower-risk and lands a visible win — recommended first).

---

## What's done (so you don't redo it)
- **WS-0 schema freeze** (`62b5fa3`): `page_bboxes` (region-level, one rect per page the chunk
  touches), `corpus` tag, `kind` discriminator frozen through the whole chain
  `chunk.Chunk → DocRef.corpus → run._chunk_to_record → embed.jsonl.ChunkRecord(+payload) →
  agent.state.ScoredChunkDict → backend.schemas.RetrievedChunk → hf_space.api_client.RetrievedChunk`.
  All additive/optional → the **existing 63,946-pt Qdrant index still hydrates** (page_bboxes
  derived from legacy `(page,bbox)`; corpus from doc_id prefix; kind=text). 366 tests green.
- **WS-0 VRAM measurement** (`docs/ws0_vram_measurement.md`): qwen3:8b Q4_K_M = **6.2 GB, 100% GPU,
  ~1.8 GB free. FITS.** VRAM is no longer the gate for the text-generation swap.
- **Figure-tier model DECIDED (S19, `docs/ws_c_qwenvl_findings.md`):** **Qwen3-VL-8B** adopted,
  replacing Florence-2 + Moondream2. Strong caption + region OCR on a sample TSB figure; VRAM 7.7 GB
  (28% CPU spill — tolerable offline). InternVL3 bake-off now optional. This lands in **WS-C**.

## Still open from WS-0
- **Curation admission criteria (§3) NOT yet frozen.** They're a write-shape decision (they decide
  what's *in* the corpus), so freeze them before WS-F, ideally alongside WS-A when the ZH feed shape
  is known. Output: explicit reject rules + a balance target across EN/TC ↔ ZH.

---

## WS-B — region-grounding render *(recommended first; lowest risk, schema already frozen)*
Goal: render the chunk's stored `page_bboxes` directly; stop re-searching at render time.
1. `hf_space/pdf_render.py`: add a path that draws the stored region rect(s) from
   `RetrievedChunk.page_bboxes` (already carried, WS-0). Each entry is `(page, x0, top, x1, bottom)`
   — draw the rect on its page using the existing `bbox_to_pixels` + `_draw_boxes`.
2. **Delete the `page.search` path** (`search_page_bbox` + the `locate_text` branch in
   `render_page_with_bbox`). No fallback needed — every re-ingested chunk has its region; legacy
   chunks have the WS-0-derived single rect. Keep `page_image_bboxes` (figure red-boxing) and the
   term-wash (`search_page_terms`) only if you still want them; the *cited* box now comes from
   `page_bboxes`, not search.
3. `hf_space/app.py`: pass `page_bboxes` into the render call instead of `locate_text`.
4. Tests: mock pdfplumber; assert the rect is drawn from `page_bboxes` and that the search path is
   gone. Live-verify on one EN doc at :7860 once the stack is up.
**Note:** old-index chunks render a single coarse rect (the WS-0 derivation). The *sharp* per-page
region only appears for docs re-chunked in WS-F. That's expected — WS-B proves the mechanism.

## WS-A — ZH source spike *(fail-fast; the genuinely riskable item)*
Goal: prove the two ZH feeds return real documents; surface blockers early.
1. New scraper modelled on `ingestion/acquisition/` (rate-limited, robots-respecting, idempotent),
   writing to `data/corpus/zh/{ac,reports}/`.
2. **Axis 1 (GREEN, the spine):** caac.gov.cn 咨询通告 ACs via the GFXWJ monthly folders + the
   catalog AC `AC-01-AA-2017-01R26`. `/XXGK/` + `/GFXWJ/` are robots-permitted.
3. **Axis 2 (AMBER):** ASN China profile as an *index only* → pull actual PDFs from their primary
   host (caac.gov.cn TZTG / regional bureaus). **Never bulk-fetch PDFs from asn.flightsafety.org**
   (it signals `ai-train=no`). asn's cert chain fails strict verify — scope the httpx verify
   exception to that host, don't disable globally.
4. Extend `doc_id._KNOWN_SOURCES`/path parsing + `DocRef.corpus` to mint `caac` for the ZH tree.
   (WS-0 left `corpus=source`; the ZH axis is where `caac` first appears.) Also extend
   `embed.jsonl.LANGS`/`SOURCES` and the backend `Literal["en","fr"]`/`["tsb","tc"]` filters to
   admit `zh`/`caac` — these were intentionally left untouched in WS-0 (no ZH data existed yet).
5. Spike deliverable: a handful of real ZH PDFs on disk + a note on volume/quality. If a feed is
   dead or licence-ambiguous, STOP and ask — don't improvise sourcing.

## The Qwen3-8B bake-off (fold into whichever session does generation work)
VRAM is settled (fits). Decide the swap on **answer quality over our own EN+ZH docs**, not
benchmarks:
- Run the same N queries (EN + ZH once ZH lands) through gemma2:9b and qwen3:8b at the production
  prompt/num_ctx. Compare grounding, citation discipline, and CJK fluency. qwen3:8b is already
  pulled in Ollama.
- If qwen wins: swap `OLLAMA_*` model default in `agent/llm.py` (+ `agent/run.py`, `backend/deps.py`
  call sites) and re-verify the HITL flow. Sequential-VRAM discipline already covers the +0.7 GB.
- Minor doc nit found during measurement: this machine's gemma2:9b is **Q4_0**, while CLAUDE.md/
  MANIFEST say Q4_K_M. Reconcile the docs when you touch the LLM row.

## WS-C figure tier (when you reach it — depends on WS-B)
Model is decided: **Qwen3-VL-8B** via HF transformers in the ingestion image (`docs/ws_c_qwenvl_findings.md`).
Implementation notes from the S19 spike: **crop each figure before captioning** (full pages exceed
the vision-token budget → empty output); **strip the `thinking` field**, keep `response`; ctx 8192
is ample for a crop. Emit `Figure {doc_id, page, bbox, caption, ocr_text}` → Neo4j `:Figure` +
a `kind=figure` retrievable chunk. `qwen3-vl:8b` is already pulled in Ollama for further spiking.

## Sequencing reminder (REINGEST_PLAN §6)
WS-B → WS-C (figures, depends on B) → WS-A → WS-E (dual-corpus eval) → **WS-F (single overnight
re-ingest, LAST)**. Land ALL code + tests before WS-F; Haiku monitors WS-F per §10.
