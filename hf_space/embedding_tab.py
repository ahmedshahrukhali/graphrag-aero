"""Embedding Space tab — interactive 3D scatter of the BGE-M3 vectors.

Loads a pre-computed projection from ``embedding_space.json`` (built by
``build_embedding_space.py``) and renders it as a Plotly 3D scatter, coloured
by corpus or language. This is the demo vehicle for the dual-corpus story:
because BGE-M3 shares one cross-lingual semantic space, same-topic docs from
different corpora (EN/TC vs ZH) land near each other — overlap = consistency,
distinct regions = the knowledge each corpus uniquely adds.

The Space loads the baked JSON statically — no Qdrant connection at runtime.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).with_name("embedding_space.json")

# Stable colours so corpora/languages keep their hue across re-renders.
_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777"]

_COLOR_FIELDS = {"corpus": "corpus", "language": "lang"}


def load_points(path: Path = _DATA_PATH) -> tuple[list[dict], str]:
    """Return ``(points, projection_method)`` from the baked JSON, or ``([], "")``
    if it hasn't been built yet."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("embedding_space.json not loaded: %s", e)
        return [], ""
    return data.get("points", []), data.get("projection", "")


def group_traces(points: list[dict], color_by: str) -> dict[str, list[dict]]:
    """Group points by the chosen field ("corpus" or "language").

    Returns ``{group_label: [point, ...]}`` in first-seen order — one Plotly
    trace per group, which gives the interactive legend (click to toggle).
    """
    field = _COLOR_FIELDS.get(color_by, "corpus")
    groups: dict[str, list[dict]] = {}
    for p in points:
        groups.setdefault(str(p.get(field, "?")), []).append(p)
    return groups


def _hover(p: dict) -> str:
    return (
        f"<b>{p.get('doc_id', '?')}</b> · p.{p.get('page', '?')}"
        f"<br>{p.get('corpus', '?')} / {p.get('lang', '?')}"
        f"<br>{p.get('snippet', '')}"
    )


def make_figure(points: list[dict], color_by: str) -> Any:
    """Build a Plotly 3D scatter Figure (one trace per group)."""
    import plotly.graph_objects as go  # lazy: keep module import cheap + testable

    fig = go.Figure()
    for i, (label, pts) in enumerate(group_traces(points, color_by).items()):
        fig.add_trace(go.Scatter3d(
            x=[p["x"] for p in pts], y=[p["y"] for p in pts], z=[p["z"] for p in pts],
            mode="markers",
            name=f"{label} ({len(pts)})",
            marker=dict(size=2, color=_PALETTE[i % len(_PALETTE)], opacity=0.7),
            text=[_hover(p) for p in pts],
            hoverinfo="text",
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
        legend=dict(title="", itemsizing="constant"),
        height=640,
    )
    return fig


def build(client: Any | None = None) -> None:
    """Create the Embedding Space tab and wire its events."""
    import gradio as gr

    points, method = load_points()

    with gr.Tab("Embedding Space"):
        if not points:
            gr.Markdown(
                "### Embedding Space\n"
                "_No projection built yet._ Run "
                "`python -m hf_space.build_embedding_space` against a populated "
                "Qdrant to generate `embedding_space.json`, then redeploy."
            )
            return

        gr.Markdown(
            f"### Embedding Space — {len(points):,} chunks ({method.upper()} → 3D)\n"
            "Each point is a chunk embedded by BGE-M3, projected to 3D. Same-topic "
            "docs cluster; because the model shares one cross-lingual space, "
            "different corpora overlap where they agree and spread out where each "
            "adds unique coverage. Drag to rotate; click legend entries to toggle."
        )
        color_by = gr.Radio(
            ["corpus", "language"], value="corpus", label="Colour by", scale=0,
        )
        plot = gr.Plot(value=make_figure(points, "corpus"))

        color_by.change(lambda cb: make_figure(points, cb), [color_by], [plot])
