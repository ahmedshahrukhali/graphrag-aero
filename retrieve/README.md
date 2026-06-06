# retrieve/ — P3

Dense ANN + cross-encoder rerank over the P2 Qdrant collection.

```
query → BGE-M3 dense (1024-dim) → Qdrant ANN top-50 →
        bge-reranker-v2-m3 cross-encoder → top-10 ScoredChunks
```

Used by:
- **P4** LangGraph agent (multi-hop retrieval)
- **P5** eval harness (Recall@k / nDCG / MRR)
- **P6** FastAPI `/retrieve` endpoint

## Install (host)

```bash
pip install -r retrieve/requirements.txt
pip install -r retrieve/requirements-dev.txt   # + pytest
```

## Run

```bash
# bring up Qdrant + ensure the index is populated (P2)
docker compose up -d qdrant
python -m embed.run --in data/chunks   # if not already indexed

# query
python -m retrieve.run --query "fuel exhaustion forced landing" --k 5 -v
python -m retrieve.run --query "alimentation en carburant" --lang fr --k 5
python -m retrieve.run --query "..." --ann-k 100 --top-k 20 --json
```

CLI prints `rank, final_score, ann_score, doc_id p.page  §section` plus a
120-char snippet — or JSON with `--json` for piping into downstream tools.

## API

```python
from qdrant_client import QdrantClient

from embed.bge_m3 import BGE_M3Embedder
from embed.qdrant import QdrantConfig, make_client
from retrieve.pipeline import retrieve_and_rerank
from retrieve.reranker import BGE_RerankerV2M3
from retrieve.vram import ModelSession

cfg = QdrantConfig.from_env()
client = make_client(cfg)

with ModelSession(BGE_M3Embedder) as emb, \
     ModelSession(BGE_RerankerV2M3) as rer:
    results = retrieve_and_rerank(
        "tail rotor failure",
        embedder=emb, reranker=rer,
        client=client, collection=cfg.collection,
        ann_k=50, top_k=10, lang=None, source=None,
    )

for r in results:
    print(r.record.doc_id, r.record.page, r.final_score, r.record.text[:80])
```

Each `ScoredChunk` carries the full `ChunkRecord` payload (doc_id, source_url,
section_title, page, bbox, lang, text) plus `ann_score` and `rerank_score`.

## VRAM (3060Ti, 8 GB)

Sequential discipline: BGE-M3 (~0.5 GB) → query encode → reranker (~0.5 GB) →
rerank → qwen3:4b (~2.5 GB, P4). Total ~3.5 GB. `ModelSession` is the
enforcement seam — interactive callers (P6) keep models resident across
requests; batch callers wrap each stage in `with ModelSession(...)`.

## Tests

```bash
pytest retrieve/tests -q
```

All offline. Embedder + reranker are stubbed; Qdrant runs `:memory:`.

## Docker

```bash
docker compose --profile retrieve build retrieve
docker compose --profile retrieve run --rm retrieve --query "..."
```

The image pre-caches BGE-M3 + reranker-v2-m3 (~2 GB of layers) so runs don't
need HuggingFace network access.
