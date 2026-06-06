# embed/ — P2

JSONL chunks → BGE-M3 dense vectors → Qdrant collection.

Reads `data/chunks/{en,fr}/{tsb,tc}/*.jsonl` (output of P1), encodes each
chunk's `text` with BGE-M3 (1024-dim dense, L2-normalised), and upserts into a
single Qdrant collection (`aerospace_dense` by default) with the chunk metadata
as the payload.

## Install (host)

```bash
pip install -r embed/requirements.txt        # torch + FlagEmbedding + qdrant-client
pip install -r embed/requirements-dev.txt    # + pytest
```

## Run

```bash
# bring up Qdrant first
docker compose up -d qdrant

# smoke: 50 chunks
python -m embed.run --in data/chunks --limit 50 -v

# full corpus (1,284 chunks)
python -m embed.run

# only French TSB
python -m embed.run --source tsb --lang fr

# drop and rebuild (e.g. embedding model changed)
python -m embed.run --recreate
```

Idempotent: point ID is `UUID(first 128 bits of chunk_hash)`, so re-runs upsert
in place — the point count stays flat.

## Collection schema

| field    | value                                |
|----------|--------------------------------------|
| name     | `$QDRANT_COLLECTION_DENSE` (default `aerospace_dense`) |
| size     | 1024                                 |
| distance | Cosine                               |

Each point payload mirrors the on-disk chunk record:

```json
{
  "doc_id":        "tsb/a00a0110",
  "source_url":    "https://www.bst-tsb.gc.ca/.../a00a0110.pdf",
  "section_title": "Findings",
  "page":          4,
  "bbox":          [72.0, 410.5, 540.0, 502.8],
  "chunk_hash":    "<sha256 hex>",
  "lang":          "en",
  "text":          "..."
}
```

## Pipeline

1. **`jsonl.py`** — iterate chunk records under `data/chunks/`, with
   `--source` / `--lang` / `--limit` filters.
2. **`bge_m3.py`** — `FlagEmbedding.BGEM3FlagModel(use_fp16=True)`,
   `encode(..., return_dense=True)` → `dense_vecs` (numpy → list).
3. **`ids.py`** — sha256 hex → deterministic UUID for the point ID.
4. **`qdrant.py`** — client factory, `ensure_collection`, `upsert_batch`.
5. **`run.py`** — orchestration + CLI. Streams batches; never loads the full
   corpus' vectors into memory at once.

## Environment

Reads from `.env` (or the process env):

| variable                 | default          |
|--------------------------|------------------|
| `QDRANT_HOST`            | `localhost`      |
| `QDRANT_PORT`            | `6333`           |
| `QDRANT_COLLECTION_DENSE`| `aerospace_dense`|
| `EMBED_MODEL`            | `BAAI/bge-m3`    |

## Tests

```bash
pytest embed/tests -q
```

All offline. BGE-M3 is replaced with a stub that returns deterministic
1024-vectors; Qdrant runs in `:memory:` mode (no Docker, no network).

## Docker

```bash
docker compose --profile embed build embed
docker compose --profile embed run --rm embed --limit 50
```

The image pre-caches the BGE-M3 weights at build time (~1 GB layer) so runs
don't need HuggingFace network access.

## VRAM (3060Ti, 8GB)

P2 is single-model: BGE-M3 fp16 ≈ 0.5 GB. Sequential load/unload across BGE-M3
→ reranker → qwen3:4b becomes load-bearing in P3+, not here.
