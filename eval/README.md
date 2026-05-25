# eval/ — P5

Retrieval-quality eval for the dense ANN + cross-encoder rerank stack.
Reports **Recall@5 / Recall@10**, **MRR**, and **nDCG@5 / nDCG@10**, plus a
per-language breakdown.

## Dataset
[dataset.jsonl](dataset.jsonl) — one item per line:

```json
{"id": "q01", "query": "...", "expected": ["tsb/<id>"], "lang": "en"}
```

`expected` is a list of `doc_id`s (matches `embed.jsonl.ChunkRecord.doc_id`,
which is `{source}/{stem}`). A retrieved chunk counts as a hit when its
`doc_id` is in `expected`. The starter set has 4 queries (3 EN + 1 FR) — extend
as the corpus and use cases grow.

## Run
Real Qdrant (needs `qdrant` up, the collection populated, and `BGE-M3` +
`bge-reranker-v2-m3` resolvable — VRAM-sequenced via `retrieve.vram.ModelSession`):

```bash
python -m eval.run                            # text report
python -m eval.run --json                     # machine-readable
python -m eval.run --dataset path/to.jsonl    # alternate set
python -m eval.run --ann-k 100 --top-k 20
```

## Layout
- [metrics.py](metrics.py) — pure-function `recall_at_k`, `reciprocal_rank`, `ndcg_at_k` (binary relevance).
- [run.py](run.py) — dataset loader, `evaluate()` driver, real-pipeline runner, CLI.
- [dataset.jsonl](dataset.jsonl) — curated query set.
- [tests/](tests/) — offline tests; CI-safe (stub embedder + stub reranker + in-memory Qdrant).

## Tests
```bash
pytest eval/tests/
```
No network, no weight downloads.
