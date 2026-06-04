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

### 1. (Quick, before the big run) EN-image OCR smoke — ☑ DONE (S22, opus-4.8)
`en` path predict-verified on real EN image-only pages. **Finding: EN TSB is 100% born-digital**
(0/1199 image-only), so `--source tsb` never exercises OCR — EN image-only pages live only in **TC**
scanned inserts/covers. Verified by running the OCR fallback directly on `data/corpus/en/tc/ac_605_002.pdf`
p.2 and `2422.pdf` p.44: `en_PP-OCRv5_mobile_rec` loads on GPU, `predict()`→`rec_texts` recovers correct
English ("Transport/Transports Canada" + TOC; "ISBN 978-92-9249-232-8") with per-glyph PDF-point bboxes.
Minor: a near-empty region can slip past `text_rec_score_thresh=0.5` as a full-page-bbox glyph (harmless
for region-level grounding). No code change needed — the `en`/`fr` mapping in `ocr.py` is confirmed live.

### 2. Scale the ZH corpus (optional, for balance)
Current sample = 10 zh docs. If the overlap demo needs more, add TTSB/CAAC seeds and pull
(`acquisition.run --source ttsb|caac`), then process `--curate`. Keep the v2 balance band in mind
(ZH : EN_TC admitted within 0.5–2.0×; manifest logs a `balance_warning`).

### 3. WS-F — the curated re-ingest (multi-hour, Haiku-monitored per REINGEST §10)

**S23 prep landed** (opus-4.7, 2026-06-04): manifest is now resume-safe (fresh-skipped docs
carry-recorded; manifest flushed every 25 docs atomically) and a quarantine tool is in place
for the 15 corrupt TC PDFs. Run in order:

#### 3a. (One-time) Quarantine unopenable PDFs
```
python -m ingestion.maintenance.quarantine_corrupt_pdfs --dry-run
# Eyeball the list (~15 expected; "No /Root object" under data/corpus/en/tc/), then:
python -m ingestion.maintenance.quarantine_corrupt_pdfs --apply
```
Broken files move to `data/corpus_quarantine/{lang}/{source}/` + `manifest.csv`. Reversible.

#### 3b. (Destructive) Clear stale pre-WS-0 chunks
Existing EN/FR chunks are pre-WS-0 (missing `page_bboxes`/`corpus`/`kind`) and mtime-fresh,
so `--curate` alone no-ops them. Move them aside so `--force` reprocesses cleanly:
```
# PowerShell:
Move-Item data\chunks "data\chunks_pre_ws0_$(Get-Date -Format yyyyMMdd)"
mkdir data\chunks
# bash:
mv data/chunks data/chunks_pre_ws0_$(date +%Y%m%d)
mkdir data/chunks
```
(`data/chunks_pilot/` is the S22 scratch — leave it.)

#### 3c. Curated ingest (multi-hour; GPU OCR; incremental manifest is safe now)
```
docker compose --profile ingest run --rm ingestion \
  --in /app/data/corpus --out /app/data/chunks --curate --force -v
```
`data/chunks/curation_manifest.json` is rewritten atomically every 25 docs — eyeball it mid-run
to confirm admitted/rejected per corpus+lang and the reject-reason histogram. Idempotent on
chunk_hash; safe to resume.

#### 3d. Re-embed into a fresh Qdrant collection (DESTROYS the live 64,440-pt index)
```
docker compose --profile embed run --rm embed --recreate
```
This is the irreversible step. Only proceed once 3c looks healthy.

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
