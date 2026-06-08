"""CLI: stream chunks from ``data/chunks/`` → embed with BGE-M3 → upsert to Qdrant.

Idempotent: point ID is derived from ``chunk_hash``, so re-running the same
input upserts in place. Use ``--recreate`` to drop and rebuild the collection
(e.g. when the embedding model changes).

Run::

    python -m embed.run --in data/chunks --limit 50 -v
    python -m embed.run --source tsb --lang en
    python -m embed.run --recreate            # full re-index
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .bge_m3 import DenseEmbedder
from .jsonl import ChunkRecord, iter_records
from .qdrant import (
    QdrantConfig,
    chunks as _batched,
    count_points,
    ensure_collection,
    make_client,
    upsert_batch,
    upsert_hybrid_batch,
)

logger = logging.getLogger(__name__)


def _default_embedder(batch_size: int) -> DenseEmbedder:
    """Real BGE-M3 — import lazily so tests don't trigger the FlagEmbedding load."""
    from .bge_m3 import get_embedder
    return get_embedder(batch_size=batch_size)


def embed_and_upsert(
    records,
    embedder: DenseEmbedder,
    client,
    collection: str,
    *,
    batch_size: int,
    sparse_embedder=None,
) -> int:
    """Stream ``records`` through ``embedder`` in batches, upsert into Qdrant.

    When ``sparse_embedder`` is provided (an object with ``embed_sparse(texts)``),
    upsert both dense and sparse vectors via ``upsert_hybrid_batch``.
    """
    total = 0
    for batch in _batched(records, batch_size):
        texts = [r.text for r in batch]
        vectors = embedder.embed(texts)
        if sparse_embedder is not None:
            sw = sparse_embedder.embed_sparse(texts)
            n = upsert_hybrid_batch(client, collection, batch, vectors, sw)
        else:
            n = upsert_batch(client, collection, batch, vectors)
        total += n
        logger.info("upserted batch: +%d  (total: %d)", n, total)
    return total


def main(
    argv: list[str] | None = None,
    *,
    embedder_factory=None,
    client=None,
) -> int:
    p = argparse.ArgumentParser(description="Embed chunks (BGE-M3 dense) → Qdrant.")
    p.add_argument("--in", dest="in_root", type=Path, default=Path("data/chunks"))
    p.add_argument("--source", choices=["tsb", "tc", "ttsb", "caac", "all"], default="all")
    p.add_argument("--lang", choices=["en", "fr", "zh", "all"], default="all")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on number of chunks (smoke runs).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--recreate", action="store_true",
                   help="Drop and rebuild the collection before upserting.")
    p.add_argument("--sparse", action="store_true",
                   help="Also embed and upsert BGE-M3 sparse (lexical) weights. "
                        "Requires --recreate when switching from dense-only (schema change).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = QdrantConfig.from_env()
    if client is None:
        client = make_client(cfg)
    ensure_collection(client, cfg.collection, recreate=args.recreate, with_sparse=args.sparse)

    source = None if args.source == "all" else args.source
    lang = None if args.lang == "all" else args.lang
    records = iter_records(args.in_root, source=source, lang=lang, limit=args.limit)

    factory = embedder_factory or _default_embedder
    embedder = factory(args.batch_size)
    sparse_embedder = embedder if args.sparse else None

    total = embed_and_upsert(
        records, embedder, client, cfg.collection,
        batch_size=args.batch_size, sparse_embedder=sparse_embedder,
    )
    final = count_points(client, cfg.collection)
    logger.info("done: %d upserted this run; %d total points in '%s'",
                total, final, cfg.collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
