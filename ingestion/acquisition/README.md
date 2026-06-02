# ingestion/acquisition/ — P1b

Scrape Transport Canada Advisory Circulars + TSB aviation investigation reports (EN + FR)
plus Taiwan TTSB reports (Traditional Chinese) and download PDFs into `data/corpus/`.

## Install

```bash
pip install -r ingestion/acquisition/requirements.txt        # runtime
pip install -r ingestion/acquisition/requirements-dev.txt    # + pytest
```

## Run

```bash
# smoke test: 2 docs per source
python -m ingestion.acquisition.run --source all --limit 2 -v

# only TSB, full
python -m ingestion.acquisition.run --source tsb

# only TC, full
python -m ingestion.acquisition.run --source tc

# only TTSB (Taiwan, Traditional Chinese) — sample 5 reports per listing
python -m ingestion.acquisition.run --source ttsb --limit 5
```

Re-runs are idempotent: existing non-empty files at the destination path are skipped.

## Output layout

```
data/corpus/
  en/
    tsb/<id>.pdf
    tc/<filename>.pdf      # e.g. AC_100-001_e08_20210622.pdf
  fr/
    tsb/<id>.pdf
    tc/<filename>.pdf      # e.g. AC_100-001_f08_20210622.pdf
  zh/
    ttsb/<media_id>_<name>.pdf   # e.g. 9234_安捷b-86002調查報告.pdf
```

TC ACs publish separate EN and FR PDFs (`_e` / `_f` revision suffix). We crawl
the EN index and the FR index separately and store each under its language
subdir. Each AC is two HTTP fetches: index → detail page → PDF.

TTSB (Taiwan) is a **single-step** crawl: the Traditional-Chinese listing pages
(`/1133/1154/1155/{1159,1157}/Lpsimplelist`) link the report PDFs directly under
`/media/{id}/`. The numeric media id prefixes the local filename — report
basenames repeat across cases (`00_general.pdf`), so the id keeps them unique and
preserves the source pointer. Older ASC-era reports are often scanned (image-only)
→ they exercise the Chinese OCR path in `processing/`. Pagination of the listings
is not yet followed; `--limit` plus the curated subset is the current scope.

## Politeness

- Rate-limited (default 1s between requests) — passed as `rate_limit_s` to
  `fetch_text` / `download`.
- Identifying User-Agent (`graphrag-aero/...`).
- Retries 429 / 5xx with backoff via `urllib3.util.retry.Retry`.
- Atomic writes via `.part` → rename; partial downloads are cleaned up on error.

## Test

```bash
pytest ingestion/acquisition/tests
```

All tests are offline: HTTP is mocked. CI must pass without network access.

## Layout

| File | What |
|------|------|
| `http_client.py` | session factory, rate-limited `fetch_text`, streaming `download` with SHA-256 |
| `tsb.py`         | Canada TSB occurrence ID extraction + PDF URL construction (EN + FR) |
| `tc.py`          | TC AC PDF link extraction from the index page |
| `ttsb.py`        | Taiwan TTSB `/media/` PDF link extraction + media-id filenames (ZH) |
| `run.py`         | CLI entry point — orchestrates all sources |
| `tests/`         | Offline unit tests (HTTP mocked) |
