"""pytest wrappers for eval/feedback_eval.py scenarios.

Each wrapper calls a standalone scenario from the module.  The scenarios
themselves contain asserts; pytest surfaces failures from those asserts.
All offline — in-memory Qdrant, no Postgres, no model weights.
"""
import pytest

pytest.importorskip("qdrant_client")

from eval.feedback_eval import (
    scenario_rejection_excludes_prior_chunks,
    scenario_excluded_hashes_absent_from_candidates,
    scenario_resolve_clears_row,
)


def test_rejection_excludes_prior_chunks():
    """§3: find_similar returns a rejected row and its chunk hashes."""
    scenario_rejection_excludes_prior_chunks()


def test_excluded_hashes_absent_from_candidates():
    """§3: retrieve node honours excluded_chunk_hashes from the feedback loop."""
    scenario_excluded_hashes_absent_from_candidates()


def test_resolve_clears_row():
    """§3: resolved rows are not returned by find_similar."""
    scenario_resolve_clears_row()
