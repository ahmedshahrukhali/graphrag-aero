"""CLI: ad-hoc retrieve-and-rerank against a live Qdrant.

Run::

    python -m retrieve.run --query "fuel exhaustion forced landing"
    python -m retrieve.run --query "alimentation en carburant" --lang fr --k 5
    python -m retrieve.run --query "..." --ann-k 100 --top-k 20 --json

Reads ``QDRANT_*`` from env (via ``embed.qdrant.QdrantConfig``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from embed.qdrant import QdrantConfig, make_client

from .pipeline import DEFAULT_ANN_K, DEFAULT_TOP_K, retrieve_and_rerank
from .reranker import ScoredChunk
from .vram import ModelSession


logger = logging.getLogger(__name__)


def _default_embedder(batch_size: int = 1):
    from embed.bge_m3 import get_embedder
    return get_embedder(batch_size=batch_size)


def _default_reranker():
    from .reranker import get_reranker
    return get_reranker()


def _format_text(results: list[ScoredChunk]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        section = f"§{r.record.section_title}" if r.record.section_title else ""
        lines.append(
            f"{i:>3}. score={r.final_score:.4f} "
            f"ann={r.ann_score:.4f}  "
            f"{r.record.doc_id} p.{r.record.page}  {section}"
        )
        # First 120 chars of text, single line.
        snippet = r.record.text.replace("\n", " ").strip()[:120]
        lines.append(f"     {snippet}...")
    return "\n".join(lines)


def _format_json(results: list[ScoredChunk]) -> str:
    return json.dumps(
        [
            {
                "rank": i,
                "doc_id": r.record.doc_id,
                "page": r.record.page,
                "section_title": r.record.section_title,
                "lang": r.record.lang,
                "source_url": r.record.source_url,
                "bbox": r.record.bbox,
                "ann_score": r.ann_score,
                "rerank_score": r.rerank_score,
                "text": r.record.text,
            }
            for i, r in enumerate(results, 1)
        ],
        ensure_ascii=False, indent=2,
    )


def main(
    argv: list[str] | None = None,
    *,
    embedder_factory=None,
    reranker_factory=None,
    client=None,
) -> int:
    p = argparse.ArgumentParser(description="Dense retrieve + cross-encoder rerank.")
    p.add_argument("--query", "-q", required=True, help="Natural-language query.")
    p.add_argument("--ann-k", type=int, default=DEFAULT_ANN_K)
    p.add_argument("--top-k", "--k", type=int, default=DEFAULT_TOP_K,
                   dest="top_k", help="Final result count after reranking.")
    p.add_argument("--lang", choices=["en", "fr"], default=None,
                   help="Filter by chunk language (payload filter, not Python-side).")
    p.add_argument("--source", choices=["tsb", "tc"], default=None)
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="Emit machine-readable JSON.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = QdrantConfig.from_env()
    if client is None:
        client = make_client(cfg)

    ef = embedder_factory or _default_embedder
    rf = reranker_factory or _default_reranker

    # Sequential VRAM: load BGE-M3 to embed the query (free it), then load the
    # reranker. Interactive callers in P6 will keep both resident; this CLI
    # round-trips through ModelSession so the discipline is exercised end-to-end.
    with ModelSession(ef, name="bge-m3") as embedder, \
         ModelSession(rf, name="bge-reranker-v2-m3") as reranker:
        results = retrieve_and_rerank(
            args.query,
            embedder=embedder,
            reranker=reranker,
            client=client,
            collection=cfg.collection,
            ann_k=args.ann_k,
            top_k=args.top_k,
            lang=args.lang,
            source=args.source,
        )

    if args.as_json:
        print(_format_json(results))
    else:
        print(_format_text(results) if results else "(no results)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
