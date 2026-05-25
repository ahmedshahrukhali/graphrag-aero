"""Turn LangGraph checkpoint history into a flat per-step trace.

Two trace surfaces exist:

1. ``state["trace"]`` — appended-to by each node as it executes (in-band).
2. ``trace_from_history(graph, config)`` — derived from the checkpointer's
   state history (out-of-band, includes the post-interrupt state).

(2) is richer because it captures the pause / resume boundary; the UI in P7
will render it as the HITL audit trail.
"""
from __future__ import annotations

from typing import Any


def trace_from_history(graph: Any, config: dict) -> list[dict]:
    """Return a chronological (oldest → newest) list of step summaries.

    Each entry::

        {
            "step":            int,         # 0-indexed
            "next":            tuple[str],  # nodes about to run, or () at end
            "values_summary":  {hop, n_candidates, draft_present, final_present},
        }
    """
    # ``get_state_history`` yields newest-first; we reverse.
    snapshots = list(graph.get_state_history(config))
    snapshots.reverse()
    out: list[dict] = []
    for i, snap in enumerate(snapshots):
        values = getattr(snap, "values", {}) or {}
        nxt = tuple(getattr(snap, "next", ()) or ())
        out.append({
            "step": i,
            "next": nxt,
            "values_summary": {
                "hop": values.get("hop"),
                "n_candidates": len(values.get("candidates", []) or []),
                "draft_present": bool(values.get("draft")),
                "final_present": bool(values.get("final")),
            },
        })
    return out
