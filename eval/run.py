"""Evaluation runner: Recall@k / MRR / nDCG@k over the retrieve+rerank pipeline.

Reads a JSONL dataset of ``{id, query, expected, lang}`` items, runs each query
through ``retrieve.pipeline.retrieve_and_rerank``, and reports aggregate +
per-language metrics. Treats relevance at the ``doc_id`` level: a retrieved
chunk counts as a hit if its ``doc_id`` is in the item's ``expected`` set.

CLI::

    python -m eval.run                        # dense (default)
    python -m eval.run --mode hybrid          # dense+sparse RRF fused
    python -m eval.run --json                 # machine-readable
    python -m eval.run --dataset path/to.jsonl

Test entry point: ``evaluate(query_runner, dataset)`` takes any callable
``(query, lang) -> list[doc_id]`` so tests can inject a stub pipeline against
in-memory Qdrant without touching the network.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


logger = logging.getLogger(__name__)


DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"
DEFAULT_KS = (5, 10)


@dataclass(frozen=True)
class EvalItem:
    id: str
    query: str
    expected: list[str]
    lang: str | None


@dataclass
class ItemResult:
    id: str
    query: str
    lang: str | None
    expected: list[str]
    retrieved: list[str]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float


# A query runner returns the ranked list of doc_ids for a (query, lang) pair.
QueryRunner = Callable[[str, "str | None"], list[str]]


def load_dataset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            d = json.loads(raw)
            items.append(EvalItem(
                id=d["id"],
                query=d["query"],
                expected=list(d["expected"]),
                lang=d.get("lang"),
            ))
    if not items:
        raise ValueError(f"dataset {path} is empty")
    return items


def _score_item(item: EvalItem, retrieved: Sequence[str]) -> ItemResult:
    return ItemResult(
        id=item.id,
        query=item.query,
        lang=item.lang,
        expected=list(item.expected),
        retrieved=list(retrieved),
        recall_at_5=recall_at_k(retrieved, item.expected, k=5),
        recall_at_10=recall_at_k(retrieved, item.expected, k=10),
        mrr=reciprocal_rank(retrieved, item.expected),
        ndcg_at_5=ndcg_at_k(retrieved, item.expected, k=5),
        ndcg_at_10=ndcg_at_k(retrieved, item.expected, k=10),
    )


def _aggregate(items: Iterable[ItemResult]) -> dict[str, float]:
    items = list(items)
    n = len(items)
    if n == 0:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0,
                "ndcg_at_5": 0.0, "ndcg_at_10": 0.0, "n": 0}
    return {
        "recall_at_5":  sum(r.recall_at_5  for r in items) / n,
        "recall_at_10": sum(r.recall_at_10 for r in items) / n,
        "mrr":          sum(r.mrr          for r in items) / n,
        "ndcg_at_5":    sum(r.ndcg_at_5    for r in items) / n,
        "ndcg_at_10":   sum(r.ndcg_at_10   for r in items) / n,
        "n": n,
    }


def evaluate(query_runner: QueryRunner, dataset: Sequence[EvalItem], *, mode: str = "dense") -> dict:
    """Run ``query_runner`` over ``dataset`` and return a metrics report.

    Report shape::

        {
          "overall": {...},
          "by_lang": {"en": {...}, "fr": {...}},
          "items": [ItemResult-as-dict, ...],
        }
    """
    results: list[ItemResult] = []
    for item in dataset:
        retrieved = query_runner(item.query, item.lang)
        results.append(_score_item(item, retrieved))

    by_lang: dict[str, dict[str, float]] = {}
    langs = {r.lang for r in results if r.lang}
    for lang in sorted(langs):
        by_lang[lang] = _aggregate(r for r in results if r.lang == lang)

    return {
        "overall": _aggregate(results),
        "by_lang": by_lang,
        "items": [asdict(r) for r in results],
        "mode": mode,
    }


def _real_query_runner(
    *,
    top_k: int = 10,
    ann_k: int = 50,
    mode: str = "dense",
) -> QueryRunner:
    """Build a runner that hits the real BGE-M3 + reranker + Qdrant stack.

    ``mode`` selects the retrieval strategy:
      "dense"  — ANN over the dense vector only (default, original behaviour).
      "hybrid" — dense + sparse RRF fusion, then rerank (§1).

    Imports the heavy deps lazily so test runs (which inject a stub runner)
    never pay the import cost or trigger weight downloads.
    """
    from embed.bge_m3 import get_embedder
    from embed.qdrant import QdrantConfig, make_client
    from retrieve.pipeline import hybrid_retrieve_and_rerank, retrieve_and_rerank
    from retrieve.reranker import get_reranker
    from retrieve.vram import ModelSession

    cfg = QdrantConfig.from_env()
    client = make_client(cfg)
    client.get_collections()  # fail fast if Qdrant isn't reachable

    def run(query: str, lang: str | None) -> list[str]:
        with ModelSession(get_embedder, name="bge-m3") as embedder, \
             ModelSession(get_reranker, name="bge-reranker-v2-m3") as reranker:
            if mode == "hybrid":
                results = hybrid_retrieve_and_rerank(
                    query,
                    embedder=embedder,
                    reranker=reranker,
                    client=client,
                    collection=cfg.collection,
                    ann_k=ann_k,
                    top_k=top_k,
                    lang=lang,
                )
            else:
                results = retrieve_and_rerank(
                    query,
                    embedder=embedder,
                    reranker=reranker,
                    client=client,
                    collection=cfg.collection,
                    ann_k=ann_k,
                    top_k=top_k,
                    lang=lang,
                )
        return [r.record.doc_id for r in results]

    return run


def _format_text(report: dict) -> str:
    lines: list[str] = []
    overall = report["overall"]
    mode = report.get("mode", "dense")
    lines.append(f"Eval over {overall['n']} queries  [mode={mode}]")
    lines.append("=" * 40)
    lines.append(f"  Recall@5:  {overall['recall_at_5']:.4f}")
    lines.append(f"  Recall@10: {overall['recall_at_10']:.4f}")
    lines.append(f"  MRR:       {overall['mrr']:.4f}")
    lines.append(f"  nDCG@5:    {overall['ndcg_at_5']:.4f}")
    lines.append(f"  nDCG@10:   {overall['ndcg_at_10']:.4f}")
    for lang, agg in report["by_lang"].items():
        lines.append(
            f"  -- {lang} (n={agg['n']}): "
            f"R@5={agg['recall_at_5']:.3f}  R@10={agg['recall_at_10']:.3f}  "
            f"MRR={agg['mrr']:.3f}  nDCG@10={agg['ndcg_at_10']:.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, query_runner: QueryRunner | None = None) -> int:
    p = argparse.ArgumentParser(description="Recall@k / MRR / nDCG over retrieve+rerank.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help="JSONL of {id, query, expected, lang} items.")
    p.add_argument("--ann-k", type=int, default=50)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--mode", choices=["dense", "hybrid"], default="dense",
                   help="Retrieval mode: dense (default) or hybrid (dense+sparse RRF).")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="Emit the full report as JSON.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dataset = load_dataset(args.dataset)
    runner = query_runner or _real_query_runner(
        top_k=args.top_k, ann_k=args.ann_k, mode=args.mode,
    )
    report = evaluate(runner, dataset, mode=args.mode)

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
