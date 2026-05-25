"""Prompt templates for gemma2:9b synthesis.

Kept separate from node logic so they're easy to iterate on without touching
graph orchestration. The synthesize node formats `{citations}` and
`{graph_context}` from the agent's state and calls ``LLM.chat(system, user)``.
"""
from __future__ import annotations

from typing import Sequence

from .state import ScoredChunkDict


SYSTEM_PROMPT = """\
You are an aerospace safety assistant grounded in Transport Canada Advisory
Circulars and TSB aviation investigation reports.

Rules:
- Answer ONLY from the citations provided. If the citations don't cover the
  question, say so explicitly — do not speculate.
- Cite each claim with [doc_id p.page] inline (e.g. [tsb/a00a0051 p.4]).
- Prefer findings, recommendations, and regulations over narrative.
- Match the language of the question (English or French).
- Be concise. 3–6 sentences unless the question genuinely needs more.
""".strip()


USER_TEMPLATE = """\
QUESTION:
{query}

GRAPH CONTEXT (related occurrences):
{graph_context}

CITATIONS:
{citations}

Answer the question using only the citations above.
""".strip()


def format_citations(candidates: Sequence[ScoredChunkDict], *, max_chars: int = 800) -> str:
    """Render top candidates as a numbered citation block for the prompt."""
    if not candidates:
        return "(no citations)"
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        section = f" §{c['section_title']}" if c.get("section_title") else ""
        snippet = c["text"].replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        lines.append(f"[{i}] {c['doc_id']} p.{c['page']}{section}\n    {snippet}")
    return "\n\n".join(lines)


def format_graph_context(rows: Sequence[dict]) -> str:
    if not rows:
        return "(none)"
    return "\n".join(
        f"- {r.get('id', '?')} ({r.get('lang', '?')}) {r.get('source_url') or ''}".rstrip()
        for r in rows
    )


def build_user_prompt(
    query: str,
    candidates: Sequence[ScoredChunkDict],
    graph_context: Sequence[dict],
) -> str:
    return USER_TEMPLATE.format(
        query=query,
        graph_context=format_graph_context(graph_context),
        citations=format_citations(candidates),
    )
