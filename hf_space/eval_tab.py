"""Eval Bench tab — run retrieval metrics against the live pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hf_space.api_client import ApiClient


# ─── inline metrics (no dependency on eval/) ────────────────────────────────

def _recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    exp = set(expected)
    return len(set(retrieved[:k]) & exp) / len(exp) if exp else 0.0


def _reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    exp = set(expected)
    for i, d in enumerate(retrieved, 1):
        if d in exp:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    exp = set(expected)
    seen: set[str] = set()
    deduped: list[str] = []
    for d in retrieved:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    actual = deduped[:k]
    dcg = sum(
        (1.0 if d in exp else 0.0) / math.log2(i + 2)
        for i, d in enumerate(actual)
    )
    ideal_rels = sorted(
        [1.0 if d in exp else 0.0 for d in actual], reverse=True,
    )
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# ─── eval dataset (embedded so the HF Space image doesn't need eval/) ───────

@dataclass(frozen=True)
class EvalItem:
    id: str
    query: str
    expected: list[str]
    lang: str | None


# Mirror of eval/dataset.jsonl (n=11, EN/FR/ZH). Kept embedded so the HF Space
# image needs no eval/ dir. `tags` from the source file are dropped (unused here).
EVAL_DATASET: list[EvalItem] = [
    EvalItem("q01", "Fox Harbour visual approach Astra SPX collision with trees",
             ["tsb/a00a0051"], "en"),
    EvalItem("q02", "Keystone Air Service Piper PA-31 fuel starvation water froze flapper valve",
             ["tsb/a00c0260"], "en"),
    EvalItem("q03", "Select Aviation College Cessna 150M mid-air collision Gatineau Airport",
             ["tsb/a23q0069"], "en"),
    EvalItem("q04", "Saint-Rémi Cessna 150G décrochage et collision avec le relief",
             ["tsb/a23q0041"], "fr"),
    EvalItem("q05", "安捷飛航訓練中心 DA-40NG 發動機失效迫降高雄外海",
             ["ttsb/3287_ttsb-aor-19-11-001"], "zh"),
    EvalItem("q06", "直昇機吊掛作業組員受傷致死 AS365N3",
             ["ttsb/3292_ttsb-aor-19-11-002"], "zh"),
    EvalItem("q07", "民用航空器维修计划和控制 CCAR-121",
             ["caac/P020151103346484825446"], "zh"),
    EvalItem("q08", "CAR 605.38 fuel quantity required Piper PA-31 fuel starvation",
             ["tsb/a00c0260"], "en"),
    EvalItem("q09", "C-FHGR C-FXLQ mid-air collision Gatineau Select Aviation Cessna 150M",
             ["tsb/a23q0069"], "en"),
    EvalItem("q10", "CCAR-121 飞行机组成员训练计划和检查要求 民用航空规章",
             ["caac/P020151103346484825446"], "zh"),
    EvalItem("q11", "CAR 602.01 VFR de nuit décrochage Cessna Saint-Rémi",
             ["tsb/a23q0041"], "fr"),
]


def build(client: "ApiClient") -> None:
    """Create the Eval Bench tab and wire its events."""
    import gradio as gr
    from hf_space.api_client import ApiError

    with gr.Tab("Eval"):
        gr.Markdown(
            "### Retrieval Evaluation Bench\n"
            "Run the curated eval dataset against the **live** retrieve + rerank pipeline. "
            "Reports Recall@k, MRR, and nDCG@k (binary relevance, doc-level)."
        )
        with gr.Row():
            topk_slider = gr.Slider(5, 50, value=10, step=5, label="Top-K")
            run_btn = gr.Button("Run evaluation", variant="primary")

        agg_md = gr.Markdown("_Click **Run evaluation** to start._")

        results_table = gr.Dataframe(
            headers=["id", "query", "lang", "expected", "top-1 hit",
                     "R@5", "R@10", "MRR", "nDCG@5", "nDCG@10"],
            datatype=["str", "str", "str", "str", "str",
                      "number", "number", "number", "number", "number"],
            label="Per-query results",
            interactive=False,
            wrap=True,
        )

        def do_eval(topk: int):
            topk = int(topk)
            rows: list[list] = []
            totals = {"r5": 0.0, "r10": 0.0, "mrr": 0.0, "nd5": 0.0, "nd10": 0.0}
            errors: list[str] = []

            for item in EVAL_DATASET:
                try:
                    resp = client.retrieve(
                        item.query, lang=item.lang, top_k=topk, ann_k=max(topk * 5, 50),
                    )
                    ids = [c.doc_id for c in resp.results]
                except ApiError as e:
                    errors.append(f"{item.id}: backend error ({e.status})")
                    rows.append([item.id, item.query[:50], item.lang or "",
                                 ", ".join(item.expected), "ERR",
                                 0, 0, 0, 0, 0])
                    continue
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{item.id}: {e}")
                    rows.append([item.id, item.query[:50], item.lang or "",
                                 ", ".join(item.expected), "ERR",
                                 0, 0, 0, 0, 0])
                    continue

                r5 = _recall_at_k(ids, item.expected, 5)
                r10 = _recall_at_k(ids, item.expected, 10)
                mrr = _reciprocal_rank(ids, item.expected)
                nd5 = _ndcg_at_k(ids, item.expected, 5)
                nd10 = _ndcg_at_k(ids, item.expected, 10)
                totals["r5"] += r5
                totals["r10"] += r10
                totals["mrr"] += mrr
                totals["nd5"] += nd5
                totals["nd10"] += nd10

                top1 = ids[0] if ids else "—"
                hit = "hit" if ids and ids[0] in set(item.expected) else ""
                rows.append([
                    item.id, item.query[:50], item.lang or "",
                    ", ".join(item.expected), f"{top1} {hit}".strip(),
                    round(r5, 3), round(r10, 3), round(mrr, 3),
                    round(nd5, 3), round(nd10, 3),
                ])

            n = len(EVAL_DATASET)
            agg = {k: v / n for k, v in totals.items()} if n else totals
            md = (
                f"**{n} queries** · Top-K = {topk}\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Recall@5 | {agg['r5']:.4f} |\n"
                f"| Recall@10 | {agg['r10']:.4f} |\n"
                f"| MRR | {agg['mrr']:.4f} |\n"
                f"| nDCG@5 | {agg['nd5']:.4f} |\n"
                f"| nDCG@10 | {agg['nd10']:.4f} |\n"
            )
            if errors:
                md += "\n**Errors:**\n" + "\n".join(f"- {e}" for e in errors)
            return md, rows

        run_btn.click(do_eval, [topk_slider], [agg_md, results_table])
