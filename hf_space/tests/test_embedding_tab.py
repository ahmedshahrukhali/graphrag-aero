"""Tests for the Embedding Space tab helpers (no Gradio rendering)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_space.embedding_tab import group_traces, load_points, make_figure, _hover


def _pt(x, y, z, corpus, lang, doc="d", page=1, snip="s"):
    return {"x": x, "y": y, "z": z, "corpus": corpus, "lang": lang,
            "doc_id": doc, "page": page, "snippet": snip}


_POINTS = [
    _pt(0.1, 0.2, 0.3, "tsb", "en", "tsb/a1"),
    _pt(-0.1, 0.0, 0.5, "tsb", "fr", "tsb/a2"),
    _pt(0.9, -0.2, 0.1, "tc", "en", "tc/ac1"),
    _pt(0.4, 0.4, -0.4, "caac", "zh", "caac/c1"),
]


def test_load_points_missing_file_returns_empty(tmp_path: Path):
    assert load_points(tmp_path / "nope.json") == ([], "")


def test_load_points_reads_points_and_method(tmp_path: Path):
    f = tmp_path / "embedding_space.json"
    f.write_text(json.dumps({"projection": "umap", "n": 2, "points": _POINTS[:2]}),
                 encoding="utf-8")
    points, method = load_points(f)
    assert method == "umap"
    assert len(points) == 2


def test_group_traces_by_corpus():
    groups = group_traces(_POINTS, "corpus")
    assert set(groups) == {"tsb", "tc", "caac"}
    assert len(groups["tsb"]) == 2
    assert len(groups["caac"]) == 1


def test_group_traces_by_language():
    groups = group_traces(_POINTS, "language")
    assert set(groups) == {"en", "fr", "zh"}
    assert len(groups["en"]) == 2


def test_group_traces_defaults_to_corpus_on_unknown_field():
    # Unknown color_by falls back to corpus grouping (not a crash).
    groups = group_traces(_POINTS, "bogus")
    assert set(groups) == {"tsb", "tc", "caac"}


def test_hover_includes_doc_and_snippet():
    h = _hover(_pt(0, 0, 0, "tsb", "en", "tsb/a1", 5, "fuel exhaustion"))
    assert "tsb/a1" in h and "p.5" in h and "fuel exhaustion" in h


def test_make_figure_one_trace_per_group():
    pytest.importorskip("plotly")
    fig = make_figure(_POINTS, "corpus")
    # 3 corpora → 3 Scatter3d traces; each point count preserved.
    assert len(fig.data) == 3
    assert sum(len(tr.x) for tr in fig.data) == len(_POINTS)


def test_make_figure_empty_points():
    pytest.importorskip("plotly")
    fig = make_figure([], "corpus")
    assert len(fig.data) == 0
