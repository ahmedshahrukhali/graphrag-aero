# ingestion/ — P1 + P1b

Two stages, each with its own subdir:

- **`acquisition/` (P1b)** — scrape the TC + TSB index pages and download PDFs (EN + FR)
  into `data/corpus/{en,fr}/{tc,tsb}/`. Lightweight deps (requests + bs4); safe to run on host.
- **`processing/` (P1)** — pdfplumber for text/bbox; PaddleOCR fallback only for image-only
  pages; fixed-size chunking (512 BGE-M3 tokens, 64 overlap); cross-doc chunk_hash dedup.
  Emits `data/chunks/{en,fr}/{tsb,tc}/{stem}.jsonl`. Every chunk carries
  `{doc_id, source_url, section_title, page, bbox, chunk_hash, lang, text}`.

This subtree is isolated from the agent runtime image — PaddleOCR + torch conflict with the
agent stack.

## Acquisition (P1b)

```bash
pip install -r ingestion/acquisition/requirements.txt
python -m ingestion.acquisition.run --source all --limit 2 -v   # smoke test
python -m ingestion.acquisition.run --source all                # full run
```

Outputs land under `data/corpus/`. Existing files are skipped (idempotent re-runs).

See `ingestion/acquisition/README.md` for details.

## Processing (P1)

```bash
pip install -r ingestion/processing/requirements.txt
python -m ingestion.processing.run --in data/corpus --out data/chunks --limit 3 -v
```

Outputs land under `data/chunks/`. Idempotent re-runs (skip if `.jsonl` newer than PDF).

See `ingestion/processing/README.md` for the chunk schema, pipeline, and Docker build.
