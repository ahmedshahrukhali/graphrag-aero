"""S44 — Export self-contained Space artifacts for the ZeroGPU Space.

Builds the three artifacts the in-Space ZeroGPU engine (S45) needs so it can run
the full Ask pipeline *without* the local stack:

  1. ``qdrant_local/``  — a qdrant-client local-mode storage dir holding an exact
     copy (identical point IDs, payloads, dense + named-sparse vectors) of the
     live Qdrant collection. Tarred to ``qdrant_local.tar.gz``. The Space
     downloads + untars it and opens ``QdrantClient(path=...)`` — no server, no
     rebuild at boot, no snapshot restore (local mode can't restore .snapshot).
  2. ``graph_context.json`` — ``{occ_id: row}`` precomputed by running the
     existing ``graph_context_for_occurrences`` over every Occurrence in Neo4j
     (exact parity with the live graph node, zero porting risk). Reused by the
     Graph tab too.
  3. ``cites_edges.json`` — the seed-set-independent inputs that the S45 pure-
     Python port of ``recurring_context_for_occurrences`` needs: every
     ``Occurrence-[:CITES]->Regulation`` edge, plus reg degrees (derivable) and
     occurrence URLs.

Then it runs a **built-in parity check**: embed a handful of sample queries and
assert the live server and the freshly-built local-mode index return the same
top-k doc_ids — proof the copy is faithful before anything ships.

⚠️  Requires the local stack (Qdrant + Neo4j) to be UP. This script is a *client*
of those services; per the Docker hard rule it never starts/stops Docker itself.

Run (from repo root, stack up)::

    python scripts/export_space_artifacts.py                 # build + parity, no upload
    python scripts/export_space_artifacts.py --upload        # also push to the dataset repo
    python scripts/export_space_artifacts.py --skip-parity   # skip the embed/search check

Upload target: dataset repo ``ahmedsali/graphaero-corpus`` under ``space_index/v1/``
(needs ``HF_TOKEN`` with write scope). Versioned prefix → trivial rollback.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import tarfile
import time

# Run as a plain script (`python scripts/export_space_artifacts.py`): put the
# repo root on sys.path so `embed`, `graph`, `retrieve` import cleanly.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from embed.qdrant import (
    QdrantConfig,
    make_client,
    ensure_collection,
    count_points,
    SPARSE_VECTOR_NAME,
)
from graph.client import make_driver
from graph.query import graph_context_for_occurrences
from retrieve.search import dense_search


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("export_space_artifacts")


DEFAULT_OUT = _REPO_ROOT / "data" / "space_index" / "v1"
DEFAULT_REPO = "ahmedsali/graphaero-corpus"
DEFAULT_PREFIX = "space_index/v1"

# Multilingual smoke queries for the parity check. Identical dense vectors +
# identical collection config ⇒ server (HNSW) and local-mode (exact) should
# agree on the top-k doc_ids. A 1-doc boundary swap can happen because the
# server is approximate and local mode is exact — that's why the pass criterion
# is set-overlap with a small tolerance, not byte-equality.
DEFAULT_PARITY_QUERIES = [
    "fuel exhaustion forced landing",          # en
    "engine failure after takeoff",            # en
    "runway excursion on landing",             # en
    "alimentation en carburant insuffisante",  # fr
    "发动机故障",                                 # zh — engine failure
]


# ── pure helpers (no I/O; trivially testable) ──────────────────────────────

def _to_sparse(sv) -> qm.SparseVector | None:
    """Normalise a scrolled sparse vector back into a ``qm.SparseVector``.

    qdrant-client returns it with ``.indices``/``.values`` attributes (or, in
    some versions, a plain mapping); handle both.
    """
    if sv is None:
        return None
    indices = getattr(sv, "indices", None)
    values = getattr(sv, "values", None)
    if indices is None and isinstance(sv, dict):
        indices, values = sv.get("indices"), sv.get("values")
    return qm.SparseVector(
        indices=[int(i) for i in (indices or [])],
        values=[float(v) for v in (values or [])],
    )


def _split_vector(vec) -> tuple[list[float] | None, qm.SparseVector | None]:
    """Pull (dense, sparse) out of a scrolled point's ``.vector``.

    Our collection is *unnamed dense* (key ``""``) + named ``"sparse"``. A
    dense-only collection returns a bare list instead of a dict.
    """
    if isinstance(vec, dict):
        sparse = _to_sparse(vec.get(SPARSE_VECTOR_NAME))
        dense = vec.get("")
        if dense is None:  # be tolerant of an alternate dense key name
            for name, v in vec.items():
                if name != SPARSE_VECTOR_NAME:
                    dense = v
                    break
        return (list(dense) if dense is not None else None), sparse
    return (list(vec) if vec is not None else None), None


def _aggregate_cites_edges(rows: list[dict]) -> dict:
    """Fold ``Occurrence-[:CITES]->Regulation`` rows into the S45 port's inputs.

    Each row: ``{"occ": occ_id, "reg": reg_id, "url": source_url}``. Returns::

        {
          "occ_cites": {occ_id: [reg_id, ...]},   # regs each occurrence cites
          "reg_occs":  {reg_id: [occ_id, ...]},    # occurrences citing each reg
          "occ_url":   {occ_id: source_url},
        }

    ``reg_occs[reg]`` length is the recurrence degree the port filters on
    (``deg > #seeds`` and ``deg <= max_reg_degree``); siblings = that list minus
    the seeds. Lists are de-duped and order-stable.
    """
    occ_cites: dict[str, list[str]] = {}
    reg_occs: dict[str, list[str]] = {}
    occ_url: dict[str, str] = {}
    for r in rows:
        occ, reg, url = r.get("occ"), r.get("reg"), r.get("url")
        if occ is None or reg is None:
            continue
        if url is not None:
            occ_url.setdefault(occ, url)
        regs = occ_cites.setdefault(occ, [])
        if reg not in regs:
            regs.append(reg)
        occs = reg_occs.setdefault(reg, [])
        if occ not in occs:
            occs.append(occ)
    return {"occ_cites": occ_cites, "reg_occs": reg_occs, "occ_url": occ_url}


def _compare_top_doc_ids(server_ids: list[str], local_ids: list[str]) -> dict:
    """Compare two ordered top-k doc_id lists. Returns a small report dict.

    ``overlap`` is a **multiset** intersection (Counter), so it measures how many
    of the k ranked slots agree — NOT how many unique documents are shared. The
    distinction matters because the top-k chunks routinely come from fewer than k
    distinct documents (several chunks per doc); a plain ``set`` intersection
    would cap a byte-identical result at the unique-doc count and spuriously
    "fail" parity.
    """
    from collections import Counter

    s_ms, l_ms = Counter(server_ids), Counter(local_ids)
    overlap = sum((s_ms & l_ms).values())
    s_set, l_set = set(server_ids), set(local_ids)
    return {
        "exact_order_match": server_ids == local_ids,
        "set_match": s_set == l_set,
        "overlap": overlap,
        "k": max(len(server_ids), len(local_ids)),
        "server_only": sorted(s_set - l_set),
        "local_only": sorted(l_set - s_set),
    }


# ── export stages (I/O) ────────────────────────────────────────────────────

def build_local_index(
    server: QdrantClient,
    collection: str,
    out_dir: pathlib.Path,
    *,
    dense_only: bool,
    page_size: int = 256,
    upsert_batch: int = 256,
) -> tuple[pathlib.Path, int]:
    """Copy every point (id + vectors + payload) from the server into a
    qdrant-client local-mode storage dir. Returns (qdrant_dir, n_points)."""
    info = server.get_collection(collection_name=collection)
    server_has_sparse = bool(getattr(info.config.params, "sparse_vectors", None))
    with_sparse = server_has_sparse and not dense_only
    logger.info(
        "source collection %s: %d points, sparse=%s → exporting with_sparse=%s",
        collection, count_points(server, collection), server_has_sparse, with_sparse,
    )

    qdrant_dir = out_dir / "qdrant_local"
    if qdrant_dir.exists():
        import shutil
        logger.info("clearing existing local index dir: %s", qdrant_dir)
        shutil.rmtree(qdrant_dir)
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    local = QdrantClient(path=str(qdrant_dir))
    try:
        ensure_collection(local, collection, recreate=True, with_sparse=with_sparse)

        total = 0
        batch: list[qm.PointStruct] = []
        offset = None
        t0 = time.time()
        while True:
            points, offset = server.scroll(
                collection_name=collection,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for pt in points:
                dense, sparse = _split_vector(pt.vector)
                if dense is None:
                    logger.warning("point %s has no dense vector, skipping", pt.id)
                    continue
                if with_sparse:
                    vector: object = {"": dense}
                    if sparse is not None:
                        vector[SPARSE_VECTOR_NAME] = sparse  # type: ignore[index]
                else:
                    vector = dense
                batch.append(qm.PointStruct(id=pt.id, vector=vector, payload=pt.payload))
                if len(batch) >= upsert_batch:
                    local.upsert(collection_name=collection, points=batch, wait=True)
                    total += len(batch)
                    batch = []
                    if total % 5120 == 0:
                        logger.info("  copied %d points (%.0fs)…", total, time.time() - t0)
            if offset is None:
                break
        if batch:
            local.upsert(collection_name=collection, points=batch, wait=True)
            total += len(batch)

        local_count = count_points(local, collection)
        server_count = count_points(server, collection)
        logger.info("copied %d points in %.0fs; local=%d server=%d",
                    total, time.time() - t0, local_count, server_count)
        if local_count != server_count:
            raise RuntimeError(
                f"point-count mismatch after copy: local={local_count} server={server_count}"
            )
    finally:
        local.close()
    return qdrant_dir, total


def dump_graph_artifacts(driver, out_dir: pathlib.Path, *, batch: int = 200) -> tuple[int, int]:
    """Write graph_context.json + cites_edges.json from live Neo4j.

    Returns (n_occurrences_with_context, n_cites_edges)."""
    with driver.session() as s:
        occ_ids = [r["id"] for r in s.run("MATCH (o:Occurrence) RETURN o.id AS id") if r["id"]]
    logger.info("Neo4j: %d Occurrence nodes", len(occ_ids))

    # graph_context: reuse the *exact* live traversal, batched to keep each
    # Cypher round-trip bounded.
    context: dict[str, dict] = {}
    for i in range(0, len(occ_ids), batch):
        rows = graph_context_for_occurrences(driver, occ_ids[i:i + batch])
        for row in rows:
            context[row["occ_id"]] = row
    (out_dir / "graph_context.json").write_text(
        json.dumps(context, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("wrote graph_context.json (%d occurrences)", len(context))

    # cites_edges: one flat query, aggregated in Python.
    with driver.session() as s:
        rows = [
            {"occ": r["occ"], "reg": r["reg"], "url": r["url"]}
            for r in s.run(
                "MATCH (o:Occurrence)-[:CITES]->(r:Regulation) "
                "RETURN o.id AS occ, r.id AS reg, o.source_url AS url"
            )
        ]
    edges = _aggregate_cites_edges(rows)
    (out_dir / "cites_edges.json").write_text(
        json.dumps(edges, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "wrote cites_edges.json (%d edges, %d regs, %d occurrences-with-cites)",
        len(rows), len(edges["reg_occs"]), len(edges["occ_cites"]),
    )
    return len(context), len(rows)


def run_parity_check(
    server: QdrantClient,
    qdrant_dir: pathlib.Path,
    collection: str,
    queries: list[str],
    *,
    k: int,
    min_overlap: int,
    strict: bool,
) -> bool:
    """Embed each query and compare server vs local-mode top-k doc_ids.

    Returns True if every query passes (set-overlap ≥ min_overlap). Degrades
    gracefully (returns True, logs a warning) if the embedder can't load on
    this host — the artifacts are still valid; only the live check is skipped.
    """
    try:
        from embed.bge_m3 import BGE_M3Embedder
        embedder = BGE_M3Embedder()
    except Exception as exc:  # noqa: BLE001
        logger.warning("parity check SKIPPED — embedder unavailable on host: %s", exc)
        return True

    local = QdrantClient(path=str(qdrant_dir))
    all_ok = True
    try:
        for q in queries:
            qv = embedder.embed([q])[0]
            server_ids = [c.record.doc_id for c in dense_search(server, collection, qv, k=k)]
            local_ids = [c.record.doc_id for c in dense_search(local, collection, qv, k=k)]
            rep = _compare_top_doc_ids(server_ids, local_ids)
            ok = rep["overlap"] >= min_overlap
            all_ok = all_ok and ok
            flag = "✓" if ok else "✗"
            logger.info(
                "  parity %s  overlap=%d/%d  exact_order=%s  q=%r",
                flag, rep["overlap"], rep["k"], rep["exact_order_match"], q,
            )
            if not rep["set_match"]:
                logger.info("      server-only: %s", rep["server_only"])
                logger.info("      local-only : %s", rep["local_only"])
    finally:
        local.close()

    if all_ok:
        logger.info("parity check PASSED (min_overlap=%d, k=%d)", min_overlap, k)
    else:
        msg = f"parity check FAILED — top-{k} doc_id overlap below {min_overlap} for some query"
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s (non-strict: continuing)", msg)
    return all_ok


def make_tarball(qdrant_dir: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    tar_path = out_dir / "qdrant_local.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    logger.info("tarring %s → %s", qdrant_dir.name, tar_path.name)
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(str(qdrant_dir), arcname="qdrant_local")
    logger.info("tarball: %s (%.1f MB)", tar_path, tar_path.stat().st_size / 1e6)
    return tar_path


def upload_artifacts(out_dir: pathlib.Path, tar_path: pathlib.Path, repo_id: str, prefix: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (write scope) required to upload; set it or drop --upload")
    api = HfApi(token=token)
    for path, name in [
        (tar_path, "qdrant_local.tar.gz"),
        (out_dir / "graph_context.json", "graph_context.json"),
        (out_dir / "cites_edges.json", "cites_edges.json"),
    ]:
        dest = f"{prefix}/{name}"
        logger.info("uploading %s → %s:%s", path.name, repo_id, dest)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type="dataset",
        )
    logger.info("upload complete → https://huggingface.co/datasets/%s/tree/main/%s", repo_id, prefix)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export self-contained Space artifacts (S44).")
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT,
                    help=f"local build dir (default: {DEFAULT_OUT})")
    ap.add_argument("--collection", default=None,
                    help="Qdrant collection (default: from QdrantConfig.from_env())")
    ap.add_argument("--dense-only", action="store_true",
                    help="export dense vectors only (skip named sparse)")
    ap.add_argument("--skip-graph", action="store_true", help="skip the Neo4j graph dumps")
    ap.add_argument("--skip-parity", action="store_true", help="skip the embed/search parity check")
    ap.add_argument("--parity-k", type=int, default=10, help="top-k for parity (default 10)")
    ap.add_argument("--parity-min-overlap", type=int, default=9,
                    help="min top-k doc_id overlap to pass (default 9; tolerates 1 HNSW swap)")
    ap.add_argument("--strict-parity", action="store_true",
                    help="raise (non-zero exit) if parity fails, instead of warning")
    ap.add_argument("--no-tar", action="store_true", help="skip building qdrant_local.tar.gz")
    ap.add_argument("--upload", action="store_true",
                    help="upload artifacts to the dataset repo (needs HF_TOKEN write)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = ap.parse_args(argv)

    cfg = QdrantConfig.from_env()
    collection = args.collection or cfg.collection
    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("export → %s  (collection=%s, host=%s:%s)", out_dir, collection, cfg.host, cfg.port)

    server = make_client(cfg)

    qdrant_dir, n_points = build_local_index(
        server, collection, out_dir, dense_only=args.dense_only
    )

    n_occ = n_edges = 0
    if not args.skip_graph:
        driver = make_driver()
        try:
            n_occ, n_edges = dump_graph_artifacts(driver, out_dir)
        finally:
            driver.close()
    else:
        logger.info("graph dumps SKIPPED (--skip-graph)")

    parity_ok = True
    if not args.skip_parity:
        parity_ok = run_parity_check(
            server, qdrant_dir, collection, DEFAULT_PARITY_QUERIES,
            k=args.parity_k, min_overlap=args.parity_min_overlap, strict=args.strict_parity,
        )
    else:
        logger.info("parity check SKIPPED (--skip-parity)")

    tar_path = None
    if not args.no_tar:
        tar_path = make_tarball(qdrant_dir, out_dir)

    if args.upload:
        if tar_path is None:
            raise RuntimeError("--upload needs the tarball; drop --no-tar")
        upload_artifacts(out_dir, tar_path, args.repo_id, args.prefix)
    else:
        logger.info("upload SKIPPED — re-run with --upload (and HF_TOKEN) when artifacts look good")

    logger.info(
        "DONE — %d points, %d occurrences-context, %d cites-edges, parity_ok=%s",
        n_points, n_occ, n_edges, parity_ok,
    )
    return 0 if parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
