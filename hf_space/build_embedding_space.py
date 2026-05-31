"""Pre-compute a 3D projection of the BGE-M3 embedding space into
``hf_space/embedding_space.json`` for the Embedding Space tab.

The Space loads the baked JSON as a static asset (same idea as
``sample_cache.json``) — it never connects to Qdrant at runtime, so this stays
deployable. Re-run on the machine whenever the index changes:

    python -m hf_space.build_embedding_space --max-points 12000

Projection: UMAP (cosine) if ``umap-learn`` is importable — best at separating
clusters, which is what makes the cross-lingual overlap visible — else a pure
NumPy PCA fallback (no extra deps). ``corpus`` is derived from the doc_id prefix
(``tsb`` / ``tc`` / later ``caac``) so the viz can colour by corpus and show
overlap between the EN/TC and ZH databases in one shared space.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).with_name("embedding_space.json")
SNIPPET_CHARS = 160


def _scroll_all(client, collection: str):
    """Yield (vector, payload) for every point, in batches."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_vectors=True,
            with_payload=True,
            limit=1024,
            offset=offset,
        )
        for p in points:
            yield p.vector, (p.payload or {})
        if offset is None:
            break


def _stratified_sample(rows: list[dict], max_points: int, seed: int = 0) -> list[dict]:
    """Downsample to ``max_points``, keeping each corpus proportionally present."""
    if len(rows) <= max_points:
        return rows
    rng = random.Random(seed)
    by_corpus: dict[str, list[dict]] = {}
    for r in rows:
        by_corpus.setdefault(r["corpus"], []).append(r)
    out: list[dict] = []
    total = len(rows)
    for corpus, group in by_corpus.items():
        take = max(1, round(max_points * len(group) / total))
        out.extend(group if take >= len(group) else rng.sample(group, take))
    rng.shuffle(out)
    return out[:max_points]


def _project(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    """Project (N, D) → (N, 3). UMAP if available, else NumPy PCA."""
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=3, metric="cosine", random_state=42)
        return reducer.fit_transform(vectors), "umap"
    except Exception as e:  # noqa: BLE001 — fall back to PCA
        print(f"  umap unavailable ({e}); using PCA", flush=True)
        x = vectors - vectors.mean(axis=0, keepdims=True)
        # Top-3 right singular vectors → 3 principal components.
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        return x @ vt[:3].T, "pca"


def _corpus_of(doc_id: str) -> str:
    return doc_id.split("/", 1)[0] if "/" in doc_id else (doc_id or "unknown")


def main() -> None:
    from embed.qdrant import QdrantConfig, make_client  # local: needs qdrant-client

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION_DENSE", "aerospace_dense"))
    ap.add_argument("--max-points", type=int, default=12000, help="display budget (browser stays smooth)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = QdrantConfig.from_env()
    client = make_client(cfg)
    print(f"scrolling {args.collection} …", flush=True)

    rows: list[dict] = []
    vecs: list[list[float]] = []
    for vec, pl in _scroll_all(client, args.collection):
        if vec is None:
            continue
        doc_id = pl.get("doc_id", "")
        rows.append({
            "doc_id": doc_id,
            "corpus": _corpus_of(doc_id),
            "lang": pl.get("lang", "?"),
            "page": pl.get("page", 0),
            "snippet": (pl.get("text", "") or "").strip().replace("\n", " ")[:SNIPPET_CHARS],
        })
        vecs.append(vec)
    print(f"  fetched {len(rows)} points", flush=True)
    if not rows:
        raise SystemExit("no points in collection — is Qdrant populated?")

    # Pair rows+vectors, downsample together.
    paired = list(zip(rows, vecs))
    sampled_rows = _stratified_sample([{**r, "_v": i} for i, (r, _) in enumerate(paired)],
                                      args.max_points, seed=args.seed)
    idx = [r.pop("_v") for r in sampled_rows]
    matrix = np.asarray([paired[i][1] for i in idx], dtype=np.float32)
    print(f"  projecting {matrix.shape} …", flush=True)

    coords, method = _project(matrix)
    # Normalise to roughly [-1, 1] on each axis for a tidy plot.
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = np.abs(coords).max() or 1.0
    coords = coords / scale

    points = []
    for r, (x, y, z) in zip(sampled_rows, coords):
        points.append({
            "x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4),
            "doc_id": r["doc_id"], "corpus": r["corpus"], "lang": r["lang"],
            "page": r["page"], "snippet": r["snippet"],
        })

    OUT_PATH.write_text(
        json.dumps({"projection": method, "n": len(points), "points": points},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH} ({len(points)} points, {method})", flush=True)


if __name__ == "__main__":
    main()
