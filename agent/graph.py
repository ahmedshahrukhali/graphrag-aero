"""Compile the LangGraph multi-hop agent.

Topology::

    START → retrieve → graph_expand → decide_continue
                                              │
                                ┌─────────────┴─────────────┐
                                │ low confidence + hops left│ otherwise
                                ↓                            ↓
                            retrieve                     synthesize
                                                              ↓
                                                          finalize → END

§3 (S27): HITL interrupt removed.  ``graph.invoke`` runs to END in one call
and returns the complete answer.  Rejected answers are recorded in the
``unaccepted_qa`` feedback store (see ``agent/feedback.py``) and applied to
the next similar query via ``excluded_chunk_hashes`` / ``rejected_prior`` in
``AgentState``.
"""
from __future__ import annotations

from typing import Any

from .nodes import (
    AgentDeps,
    finalize_node,
    make_decide_continue,
    make_graph_expand_node,
    make_retrieve_node,
    make_synthesize_node,
)
from .state import AgentState


def build_graph(deps: AgentDeps, *, checkpointer: Any) -> Any:
    """Build + compile the agent's StateGraph.

    Returns a compiled LangGraph object whose ``.invoke`` / ``.get_state`` /
    ``.update_state`` / ``.get_state_history`` methods drive the HITL flow.
    """
    from langgraph.graph import END, START, StateGraph  # type: ignore

    g: StateGraph = StateGraph(AgentState)
    g.add_node("retrieve", make_retrieve_node(deps))
    g.add_node("graph_expand", make_graph_expand_node(deps))
    g.add_node("synthesize", make_synthesize_node(deps))
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "graph_expand")
    g.add_conditional_edges(
        "graph_expand",
        make_decide_continue(deps),
        {"retrieve": "retrieve", "synthesize": "synthesize"},
    )
    g.add_edge("synthesize", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
