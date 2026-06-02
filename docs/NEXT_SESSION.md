# Next-session plan — WS-F curated re-ingest (the big run)

**Author:** sonnet-4.6 (S21, 2026-06-02). **Status:** Chinese-OCR plan DONE + GPU-accelerated;
the only remaining big step is the curated re-ingest at scale. Read this + the `MANIFEST.md`
resume pointer + `docs/REINGEST_PLAN.md` §3 (FROZEN v2 curation) / §7 (run sequence) / §10
(Haiku monitoring), then execute.

## Where we are (end of S21)
- **Chinese OCR proven end-to-end** (scanned PDF → `ch`/`chinese_cht` PaddleOCR → 中文 chunks →
  retrieval #1 → cited Chinese `/query` answer → region highlight). Qdrant at **64,440** pts
  (63,946 EN/FR/TC + ~494 zh sample).
- **OCR is on GPU** (paddlepaddle-gpu 3.0 / PP-OCRv5, self-contained wheel). Device auto-detects
  (GPU→CPU fallback, printed to stderr — no silent CPU drop). `text_rec_score_thresh=0.5`.
- **Curation FROZEN v2** (`curation.py` + `--curate`): empty / sub_threshold(<200) / cover_only /
  lang_misdetect(CJK<0.10). Auto-rejects the broken pre-2018 ASC-era TTSB CID-mojibake PDFs.
- **EN/FR OCR mapped to valid PP-OCRv5 models** (`en`/`fr`; the 2.x shared `latin` model is gone).
  Construction verified in-image; the predict→parse path is the same one proven on `ch`.

## Do, in order

### 1. (Quick, before the big run) EN-image OCR smoke
A born-digital-heavy corpus rarely OCRs, but ~670 image-heavy PDFs exist. Confirm the `en` path
produces sane text on one real EN image-only page (predict, not just construct):
`docker compose --profile ingest run --rm ingestion --in /app/data/corpus --out /app/data/chunks --source tsb --limit <a scanned one> -v`
— watch for `ocr fallback … (en)` + sane Latin text in the chunk. (Only `ch`/`chinese_cht` are
predict-verified so far.)

### 2. Scale the ZH corpus (optional, for balance)
Current sample = 10 zh docs. If the overlap demo needs more, add TTSB/CAAC seeds and pull
(`acquisition.run --source ttsb|caac`), then process `--curate`. Keep the v2 balance band in mind
(ZH : EN_TC admitted within 0.5–2.0×; manifest logs a `balance_warning`).

### 3. WS-F — the curated re-ingest (multi-hour, Haiku-monitored per REINGEST §10)
`docker compose --profile ingest run --rm ingestion --in /app/data/corpus --out /app/data/chunks --curate -v`
then `docker compose --profile embed run --rm embed`.
- Writes `data/chunks/curation_manifest.json` — eyeball admitted/rejected per corpus+lang and the
  reject-reason histogram before trusting the index.
- Idempotent (chunk_hash point IDs); safe to resume.

### 4. Re-verify after the run
- `python -m eval.run --json` (Recall@k / nDCG / MRR) — NB the eval dataset has **no ZH queries**;
  add a few zh query/relevance pairs to `eval/dataset.jsonl` to measure Chinese retrieval honestly.
- Spot-check a Chinese `/query` + an EN `/query` against the refreshed index.

## Performance note (the real WS-F ceiling is NOT the GPU)
OCR + embed are GPU now. The bottleneck is **CPU + serial orchestration**:
- pdfplumber page rasterization (scanned pages render to ~6889×9745 px) and text extraction for the
  born-digital majority are single-threaded CPU.
- `processing.run` does one doc at a time, so CPU stages and GPU OCR don't overlap across docs — the
  GPU idles between pages.
**If WS-F is too slow:** parallelize ingestion across docs (worker pool: doc N's CPU work overlaps
doc N-1's GPU OCR) and/or drop the OCR render DPI from 200 (those 200 MB bitmaps are overkill; ~150
likely fine). These are the highest-leverage speedups and are *not* yet done.

## VRAM discipline (8 GB card)
OCR peaks at ~7.9 GB during inference; embed ~2 GB; gemma2:9b ~5.5 GB. They must run in **separate
phases** — never concurrently. WS-F is ingest-then-embed (sequential), so this holds; just don't fire
`/query` (gemma) during the batch.

## Still open (smaller)
- Browser click-through screenshot of the ZH bbox highlight at :7860 (page_bboxes confirmed in API).
- About tab (What/Why/How) — author pre-approved.
- Query-time latency: gemma2:9b ~350 s/query — the Qwen3-8B generator swap is still under eval.
- Each `--rm` ingestion container re-downloads PP-OCR models; mount a model-cache volume to skip it
  on repeat runs.
