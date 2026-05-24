# ingestion/acquisition/ — P1b

Scrape Transport Canada Advisory Circulars + TSB aviation investigation reports (EN + FR)
and download PDFs into `data/corpus/`.

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
```

TC ACs publish separate EN and FR PDFs (`_e` / `_f` revision suffix). We crawl
the EN index and the FR index separately and store each under its language
subdir. Each AC is two HTTP fetches: index → detail page → PDF.

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
| `tsb.py`         | TSB occurrence ID extraction + PDF URL construction (EN + FR) |
| `tc.py`          | TC AC PDF link extraction from the index page |
| `run.py`         | CLI entry point — orchestrates both sources |
| `tests/`         | Offline unit tests (HTTP mocked) |
