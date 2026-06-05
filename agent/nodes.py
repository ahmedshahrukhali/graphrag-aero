"""LangGraph nodes for the multi-hop agent.

Each node is a function ``(state: AgentState) -> dict`` returning a partial
state update; LangGraph merges it back into the graph state. Heavy deps
(retrieval pipeline, Neo4j driver, LLM) are bundled in :class:`AgentDeps` so
the same node functions are used in production and in tests with stubs.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from qdrant_client import QdrantClient

from embed.bge_m3 import DenseEmbedder

from graph.client import DriverLike
from graph.query import (
    graph_context_for_occurrences,
    recurring_context_for_occurrences,
)
from retrieve.pipeline import (
    DEFAULT_CHAR_BUDGET,
    DEFAULT_TOP_N_DOCS,
    anchored_retrieve,
    retrieve_and_rerank,
)
from retrieve.reranker import CrossEncoderReranker

from .llm import LLM
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .reformulate import reformulate
from .state import AgentState, scored_chunk_to_dict


logger = logging.getLogger(__name__)


# Calibrated S27 (2026-06-04): jargon-id eval queries (q08-q11) score <0.4 on
# hop 1; well-matched queries score >0.85.  The gap [0.4, 0.85] is clean;
# 0.65 is the midpoint.  Re-calibrate from live eval after corpus changes.
CONFIDENCE_THRESHOLD = 0.65


def _anchored_default() -> bool:
    return os.environ.get("RETRIEVE_ANCHORED", "1").lower() not in ("0", "false", "no")


def _broaden_default() -> int:
    """Fire the outward graph hop only when retrieval is this-narrow or narrower
    (distinct docs ≤ this). Env-overridable like RETRIEVE_ANCHORED."""
    try:
        return int(os.environ.get("GRAPH_BROADEN_WHEN_DOCS_LTE", "3"))
    except ValueError:
        return 3


@dataclass(frozen=True)
class AgentDeps:
    """Bundle of injected dependencies — keeps nodes pure-ish and testable."""

    embedder: DenseEmbedder
    reranker: CrossEncoderReranker
    qdrant: QdrantClient
    neo4j: DriverLike
    llm: LLM
    collection: str
    ann_k: int = 50
    top_k: int = 10
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    anchored: bool = field(default_factory=_anchored_default)
    top_n_docs: int = DEFAULT_TOP_N_DOCS
    char_budget: int = DEFAULT_CHAR_BUDGET
    # Outward graph hop (recurring patterns across other reports), gated by
    # retrieval concentration. See graph.query.recurring_context_for_occurrences.
    broaden_when_docs_lte: int = field(default_factory=_broaden_default)
    recurring_max_regs: int = 5
    recurring_max_siblings: int = 3
    recurring_max_reg_degree: int = 15


# ─── helpers ─────────────────────────────────────────────────────────────────

def _trace_entry(node: str, started: float, **extra) -> dict:
    entry = {"node": node, "elapsed_ms": int((time.time() - started) * 1000)}
    entry.update(extra)
    return entry


def _merge_candidates(
    existing: list[dict], new: list[dict], *, top_k: int,
) -> list[dict]:
    """Dedup by chunk_hash (keep first occurrence) and keep best ``top_k``."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in existing + new:
        h = c["chunk_hash"]
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    out.sort(
        key=lambda c: (
            -1.0 if c.get("rerank_score") is None else -float(c["rerank_score"]),
            -float(c.get("ann_score", 0.0)),
        )
    )
    return out[:top_k]


def _occurrence_ids_from(candidates: list[dict]) -> list[str]:
    ids: list[str] = []
    for c in candidates:
        doc_id = c.get("doc_id", "")
        if doc_id.startswith("tsb/"):
            ids.append(doc_id.split("/", 1)[1])
    return ids


# ─── nodes ───────────────────────────────────────────────────────────────────

def make_retrieve_node(deps: AgentDeps) -> Callable[[AgentState], dict]:
    def retrieve_node(state: AgentState) -> dict:
        started = time.time()
        current_hop = state.get("hop", 0)
        lang = state.get("lang")
        source = state.get("source")
        exclude_hashes = state.get("excluded_chunk_hashes") or []

        # §2: on hop N>1 expand the query with novel terms from prior candidates.
        if current_hop > 0 and state.get("candidates"):
            query = reformulate(state["query"], state["candidates"])
            logger.debug("reformulated query (hop %d): %r", current_hop + 1, query[:120])
        else:
            query = state["query"]

        if deps.anchored:
            # Anchored mode pools whole documents and lets the char budget +
            # reading-order sort govern, so we replace candidates wholesale
            # rather than merge/truncate to top_k (which would drop content
            # pages and scramble reading order).
            results = anchored_retrieve(
                query,
                embedder=deps.embedder,
                reranker=deps.reranker,
                client=deps.qdrant,
                collection=deps.collection,
                ann_k=deps.ann_k,
                top_k=deps.top_k,
                top_n_docs=deps.top_n_docs,
                char_budget=deps.char_budget,
                lang=lang,
                source=source,
                exclude_hashes=exclude_hashes if exclude_hashes else None,
            )
            merged = [scored_chunk_to_dict(r) for r in results]
            n_new = len(merged)
        else:
            results = retrieve_and_rerank(
                query,
                embedder=deps.embedder,
                reranker=deps.reranker,
                client=deps.qdrant,
                collection=deps.collection,
                ann_k=deps.ann_k,
                top_k=deps.top_k,
                lang=lang,
                source=source,
                exclude_hashes=exclude_hashes if exclude_hashes else None,
            )
            new = [scored_chunk_to_dict(r) for r in results]
            merged = _merge_candidates(state.get("candidates", []), new, top_k=deps.top_k)
            n_new = len(new)

        trace = list(state.get("trace", []))
        trace.append(_trace_entry(
            "retrieve", started,
            anchored=deps.anchored,
            hop=current_hop + 1,
            reformulated=(query != state["query"]),
            n_new=n_new, n_merged=len(merged),
            best_rerank=max((c["rerank_score"] or 0.0) for c in merged) if merged else None,
        ))
        return {
            "candidates": merged,
            "hop": current_hop + 1,
            "reformulated_query": query if query != state["query"] else None,
            "trace": trace,
        }
    return retrieve_node


def make_graph_expand_node(deps: AgentDeps) -> Callable[[AgentState], dict]:
    def graph_expand_node(state: AgentState) -> dict:
        started = time.time()
        candidates = state.get("candidates", [])
        ids = _occurrence_ids_from(candidates)
        rows = graph_context_for_occurrences(deps.neo4j, ids)
        trace = list(state.get("trace", []))
        trace.append(_trace_entry(
            "graph_expand", started,
            n_ids=len(set(ids)), n_rows=len(rows),
        ))

        # Outward second hop, gated by concentration: only broaden when the
        # retrieved context is narrow (few distinct docs) — exactly when the
        # answer would otherwise overstate breadth from 2-3 anchored docs.
        b_started = time.time()
        distinct_docs = len({c.get("doc_id", "") for c in candidates})
        fired = bool(ids) and distinct_docs <= deps.broaden_when_docs_lte
        recurring: list[dict] = []
        if fired:
            recurring = recurring_context_for_occurrences(
                deps.neo4j, ids,
                max_regs=deps.recurring_max_regs,
                max_siblings_per_reg=deps.recurring_max_siblings,
                max_reg_degree=deps.recurring_max_reg_degree,
            )
        n_siblings = sum(len(r.get("siblings", [])) for r in recurring)
        trace.append(_trace_entry(
            "graph_broaden", b_started,
            distinct_docs=distinct_docs, fired=fired,
            n_regs=len(recurring), n_siblings=n_siblings,
        ))
        return {
            "graph_context": rows,
            "recurring_context": recurring,
            "trace": trace,
        }
    return graph_expand_node


def make_decide_continue(deps: AgentDeps) -> Callable[[AgentState], str]:
    """Conditional edge function — returns next node *name*."""
    def decide_continue(state: AgentState) -> str:
        candidates = state.get("candidates", [])
        best = 0.0
        for c in candidates:
            score = c.get("rerank_score") or 0.0
            if score > best:
                best = score
        hop = state.get("hop", 0)
        max_hops = state.get("max_hops", 2)
        if hop < max_hops and best < deps.confidence_threshold:
            return "retrieve"
        return "synthesize"
    return decide_continue


def make_synthesize_node(deps: AgentDeps) -> Callable[[AgentState], dict]:
    def synthesize_node(state: AgentState) -> dict:
        started = time.time()
        # VRAM discipline: unload retrieval models before generating
        if hasattr(deps.embedder, "unload"):
            deps.embedder.unload()
        if hasattr(deps.reranker, "unload"):
            deps.reranker.unload()

        system = SYSTEM_PROMPT
        # §3: if a prior similar answer was rejected, tell the model to avoid
        # the same framing and cite different evidence.
        rejected_prior = state.get("rejected_prior")
        if rejected_prior:
            system = (
                f"{SYSTEM_PROMPT}\n\n"
                "NOTE: A previous answer to a similar question was explicitly "
                "rejected by the user. Avoid the same framing and cite "
                "different evidence than: " + rejected_prior[:400]
            )

        user = build_user_prompt(
            state["query"],
            state.get("candidates", []),
            state.get("graph_context", []),
            state.get("recurring_context", []),
        )
        draft = deps.llm.chat(system, user)
        trace = list(state.get("trace", []))
        trace.append(_trace_entry(
            "synthesize", started,
            prompt_chars=len(user), draft_chars=len(draft),
            has_rejected_prior=bool(rejected_prior),
        ))
        return {"draft": draft, "trace": trace}
    return synthesize_node


def finalize_node(state: AgentState) -> dict:
    """Copy ``draft`` to ``final``. The HITL gate is ``interrupt_before`` this
    node, so by the time we run, the caller has had a chance to edit the
    draft via ``graph.update_state``. We honour whatever ``draft`` says now."""
    draft = state.get("draft") or ""
    trace = list(state.get("trace", []))
    trace.append({"node": "finalize", "elapsed_ms": 0, "final_chars": len(draft)})
    return {"final": draft, "trace": trace}
