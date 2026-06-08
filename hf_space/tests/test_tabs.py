"""Tests for the Corpus, Graph, Eval, and About tab helpers.

Pure-logic tests only — no Gradio rendering. The ``build()`` calls are
tested implicitly when ``make_app()`` constructs the Blocks tree inside
the Docker image; these tests verify the data-processing functions.
"""
from __future__ import annotations

import math

import pytest


# ── eval metrics (inlined in eval_tab, no eval/ dependency) ──────────────

from hf_space.eval_tab import _recall_at_k, _reciprocal_rank, _ndcg_at_k


class TestRecallAtK:
    def test_perfect_recall(self):
        assert _recall_at_k(["a", "b", "c"], ["a"], 5) == 1.0

    def test_miss(self):
        assert _recall_at_k(["x", "y", "z"], ["a"], 5) == 0.0

    def test_partial(self):
        assert _recall_at_k(["a", "x", "b"], ["a", "b", "c"], 5) == pytest.approx(2 / 3)

    def test_k_cutoff(self):
        assert _recall_at_k(["x", "x", "x", "x", "x", "a"], ["a"], 5) == 0.0

    def test_empty_expected(self):
        assert _recall_at_k(["a"], [], 5) == 0.0


class TestReciprocalRank:
    def test_first_hit(self):
        assert _reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_hit(self):
        assert _reciprocal_rank(["x", "a", "c"], ["a"]) == 0.5

    def test_no_hit(self):
        assert _reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


class TestNdcgAtK:
    def test_perfect(self):
        assert _ndcg_at_k(["a"], ["a"], 5) == pytest.approx(1.0)

    def test_zero(self):
        assert _ndcg_at_k(["x", "y"], ["a"], 5) == pytest.approx(0.0)

    def test_deduplication(self):
        # Duplicate doc_ids should be collapsed before scoring.
        val = _ndcg_at_k(["a", "a", "a", "a", "b"], ["a"], 5)
        assert val == pytest.approx(1.0)

    def test_position_matters(self):
        # Hit at position 1 scores higher than hit at position 3.
        v1 = _ndcg_at_k(["a", "x", "y"], ["a"], 5)
        v2 = _ndcg_at_k(["x", "y", "a"], ["a"], 5)
        assert v1 > v2


# ── graph rendering ──────────────────────────────────────────────────────

from hf_space.graph_tab import _render_graph


class TestRenderGraph:
    def test_with_findings(self):
        data = {
            "occ_id": "a13q0098",
            "occ_url": "https://example.com/a13q0098",
            "findings": [
                {"text": "Fuel tanks empty", "category": "cause",
                 "lang": "en", "source_doc_id": "tsb/a13q0098",
                 "page": 5, "cites_reg": "CAR 602.115"},
            ],
            "recommendations": [],
            "direct_regs": ["CAR 602.115"],
            "acs": [],
        }
        md = _render_graph(data)
        assert "a13q0098" in md
        assert "Findings (1)" in md
        assert "CAR 602.115" in md
        assert "Fuel tanks empty" in md

    def test_empty_graph(self):
        data = {
            "occ_id": "test",
            "occ_url": None,
            "findings": [],
            "recommendations": [],
            "direct_regs": [],
            "acs": [],
        }
        md = _render_graph(data)
        assert "isolated Occurrence node" in md

    def test_recommendations_render(self):
        data = {
            "occ_id": "x",
            "occ_url": None,
            "findings": [],
            "recommendations": [
                {"id": "A99-01", "text": "Install warning system",
                 "lang": "en", "source_doc_id": "tsb/x", "page": 10},
            ],
            "direct_regs": [],
            "acs": ["AC 500-001"],
        }
        md = _render_graph(data)
        assert "Recommendations (1)" in md
        assert "A99-01" in md
        assert "AC 500-001" in md


# ── about tab content ────────────────────────────────────────────────────

from hf_space.about_tab import _WHAT_MD, _HOW_MD


class TestAboutTabContent:
    def test_what_mentions_corpus_numbers(self):
        assert "77,179" in _WHAT_MD
        assert "TSB" in _WHAT_MD
        assert "Transport Canada" in _WHAT_MD

    def test_what_mentions_figure_chunks(self):
        assert "figure" in _WHAT_MD.lower()
        assert "Qwen2.5-VL" in _WHAT_MD

    def test_how_contains_pipeline_steps(self):
        for step in ("Qdrant", "reranker", "Neo4j", "LangGraph"):
            assert step in _HOW_MD, f"missing pipeline step: {step}"

    def test_how_lists_key_components(self):
        for component in ("pdfplumber", "PaddleOCR", "FlagEmbedding", "FastAPI", "Gradio", "auto-fallback"):
            assert component in _HOW_MD, f"missing stack component: {component}"

    def test_all_sections_are_nonempty_strings(self):
        for name, val in [("WHAT", _WHAT_MD), ("HOW", _HOW_MD)]:
            assert isinstance(val, str) and len(val) > 100, f"{name} content too short"
