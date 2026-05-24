# ingestion/processing/ — P1

PDF → JSONL chunks. Reads `data/corpus/{en,fr}/{tsb,tc}/*.pdf` and writes
`data/chunks/{en,fr}/{tsb,tc}/{stem}.jsonl` (one chunk per line).

## Install (host)

```bash
pip install -r ingestion/processing/requirements.txt        # text-only
pip install -r ingestion/processing/requirements-ocr.txt    # + PaddleOCR (heavy)
pip install -r ingestion/processing/requirements-dev.txt    # + pytest
```

## Run

```bash
# smoke: 3 docs from each source
python -m ingestion.processing.run --in data/corpus --out data/chunks --limit 3 -v

# only TSB
python -m ingestion.processing.run --source tsb

# force reprocess
python -m ingestion.processing.run --force
```

Idempotent: a doc whose output `.jsonl` is newer than its source PDF is skipped.

## Chunk record schema

One JSON object per line:

```json
{
  "doc_id":        "tsb/a00a0110",
  "source_url":    "https://www.bst-tsb.gc.ca/.../a00a0110.pdf",
  "section_title": "Findings",
  "page":          4,
  "bbox":          [72.0, 410.5, 540.0, 502.8],
  "chunk_hash":    "<sha256 of normalized text>",
  "lang":          "en",
  "text":          "..."
}
```

- `doc_id = "{source}/{stem}"` — derived from the corpus path.
- `source_url` is reconstructed from acquisition URL patterns; `null` for TC
  (the upstream URL needs a `YYYY-MM` segment we don't carry on disk).
- `bbox = [x0, y0, x1, y1]` in pdfplumber's coordinate space, on `page`.
- `chunk_hash` is `sha256(normalize(text))` — used for cross-doc dedup.

## Pipeline

1. **`pdf.py`** — pdfplumber per-page `extract_text()` + chars list. A page
   with no extractable text but with images is flagged `image_only`.
2. **`ocr.py`** — lazy-imported PaddleOCR fallback for image-only pages.
3. **`chunk.py`** — joins page text, tokenizes with the BGE-M3 (XLM-R)
   tokenizer, emits 512-token windows with 64-token overlap. Each chunk
   carries the dominant page and the union of contributing chars' bboxes on
   that page. Section title is the most recent header-style line seen (short
   line with above-median font size).
4. **`dedup.py`** — drops chunks whose hash was already emitted this run.
5. **`run.py`** — orchestration + CLI, atomic JSONL writes.

## Tests

```bash
pytest ingestion/processing/tests
```

All offline. The tokenizer and pdfplumber are mocked; PaddleOCR is import-
guarded so its absence doesn't break collection.

## Docker

Two stages — text-only (default) and `ocr` (adds PaddleOCR):

```bash
docker compose --profile ingest build ingestion
docker compose --profile ingest run --rm ingestion
```

The image pre-downloads the BGE-M3 tokenizer (no model weights) at build time
so runtime has no HuggingFace network call.
