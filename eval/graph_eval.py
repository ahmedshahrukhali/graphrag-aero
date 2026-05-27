"""Graph traversal evaluation.

Measures whether the knowledge graph returns grounded findings and regulations
for known occurrences — something pure vector retrieval cannot do. This is the
proof that the graph adds value.

Metric: TraversalHit@occ
  For each dataset item, call graph_context_for_occurrences([occ_id]).
  A "hit" requires ALL of:
    - at least one Finding node returned
    - at least one finding whose text contains a keyword from expect_finding_keywords
  Score = fraction of items that hit.

CLI::

    python -m eval.graph_eval              # real Neo4j
    python -m eval.graph_eval --json

Test entry: graph_evaluate(traversal_runner, dataset) takes
  traversal_runner: occ_id -> {findings, recommendations, direct_regs, acs}
so tests inject a stub driver without touching Neo4j.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent / "graph_dataset.jsonl"


@dataclass(frozen=True)
class GraphEvalItem:
    id: str
    query: str
    occ_id: str
    expect_regs: list[str]
    expect_finding_keywords: list[str]
    lang: str | None
    note: str


@dataclass
class GraphItemResult:
    id: str
    occ_id: str
    n_findings: int
    n_recs: int
    n_regs: int
    keyword_hit: bool
    score: float  # 1.0 = hit, 0.0 = miss


TraversalRunner = Callable[[str], dict]


def load_graph_dataset(path: Path) -> list[GraphEvalItem]:
    items: list[GraphEvalItem] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            d = json.loads(raw)
            items.append(GraphEvalItem(
                id=d["id"],
                query=d["query"],
                occ_id=d["occ_id"],
                expect_regs=d.get("expect_regs") or [],
                expect_finding_keywords=d.get("expect_finding_keywords") or [],
                lang=d.get("lang"),
                note=d.get("note", ""),
            ))
    if not items:
        raise ValueError(f"graph dataset {path} is empty")
    return items


def _keyword_hit(findings: list[dict], keywords: list[str]) -> bool:
    if not keywords:
        return True  # no constraint = pass
    joined = " ".join((f.get("text") or "") for f in findings).lower()
    return any(kw.lower() in joined for kw in keywords)


def _score_item(item: GraphEvalItem, ctx: dict) -> GraphItemResult:
    findings = ctx.get("findings") or []
    recs = ctx.get("recommendations") or []
    regs = list({r for r in (ctx.get("direct_regs") or [])
                 if r} | {f.get("cites_reg") for f in findings
                           if f.get("cites_reg")})
    kw_hit = _keyword_hit(findings, item.expect_finding_keywords)
    hit = bool(findings) and kw_hit
    return GraphItemResult(
        id=item.id, occ_id=item.occ_id,
        n_findings=len(findings), n_recs=len(recs), n_regs=len(regs),
        keyword_hit=kw_hit, score=1.0 if hit else 0.0,
    )


def graph_evaluate(
    traversal_runner: TraversalRunner,
    dataset: list[GraphEvalItem],
) -> dict:
    """Run traversal_runner for each item; return a metrics report.

    traversal_runner(occ_id) -> dict with findings/recommendations/direct_regs/acs.
    """
    results: list[GraphItemResult] = []
    for item in dataset:
        ctx = traversal_runner(item.occ_id)
        results.append(_score_item(item, ctx))

    n = len(results)
    score = sum(r.score for r in results) / n if n else 0.0
    return {
        "traversal_hit": score,
        "n": n,
        "items": [asdict(r) for r in results],
    }


def _real_traversal_runner() -> TraversalRunner:
    from graph.client import make_driver
    from graph.query import graph_context_for_occurrences

    driver = make_driver()

    def run(occ_id: str) -> dict:
        rows = graph_context_for_occurrences(driver, [occ_id])
        return rows[0] if rows else {}

    return run


def _format_text(report: dict) -> str:
    lines = [
        f"Graph traversal eval over {report['n']} occurrences",
        "=" * 44,
        f"  TraversalHit: {report['traversal_hit']:.4f}",
        "",
        "  Per-item:",
    ]
    for r in report["items"]:
        hit = "✓" if r["score"] == 1.0 else "✗"
        lines.append(
            f"  {hit} {r['id']} ({r['occ_id']})  "
            f"findings={r['n_findings']}  recs={r['n_recs']}  "
            f"regs={r['n_regs']}  kw_hit={r['keyword_hit']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None,
         traversal_runner: TraversalRunner | None = None) -> int:
    p = argparse.ArgumentParser(description="Graph traversal eval.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dataset = load_graph_dataset(args.dataset)
    runner = traversal_runner or _real_traversal_runner()
    report = graph_evaluate(runner, dataset)

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
