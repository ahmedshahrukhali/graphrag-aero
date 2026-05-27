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
- Answer from the citations and graph context provided. Synthesize across
  them — combine findings, recommendations, and regulations even when they
  come from different sources. Only say "not covered in the cited sources"
  for aspects genuinely absent from every source. Do not speculate.
- Cite each claim with [doc_id p.page] inline (e.g. [tsb/a00a0051 p.4]).
  Graph-context facts carry their own [doc p.page] — use those citations.
- Prefer findings, recommendations, and regulations over narrative.
- Match the language of the question (English or French).
- Be concise. 3–6 sentences unless the question genuinely needs more.
""".strip()


USER_TEMPLATE = """\
QUESTION:
{query}

GRAPH CONTEXT (structured facts extracted from occurrence reports — cite with [doc p.page]):
{graph_context}

CITATIONS (ranked text passages — cite with [doc_id p.page]):
{citations}

Answer the question using the graph context and citations above.
""".strip()


def format_citations(candidates: Sequence[ScoredChunkDict], *, max_chars: int = 2000) -> str:
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
    """Render traversal results as cited facts.

    Each Occurrence row exposes its findings, recommendations, and regulation /
    AC links as inline-cited lines so gemma can reference them with provenance.
    Bare {id, source_url, lang} rows (pre-extraction fallback) are rendered
    minimally so the prompt stays valid even before graph population.
    """
    if not rows:
        return "(none)"

    out: list[str] = []
    for row in rows:
        occ_id = row.get("occ_id") or row.get("id", "?")
        # Rich traversal row (post-extraction)
        findings = row.get("findings") or []
        recs = row.get("recommendations") or []
        direct_regs = row.get("direct_regs") or []
        acs_ref = row.get("acs") or []

        if findings or recs or direct_regs or acs_ref:
            out.append(f"Occurrence {occ_id}:")
            for f in findings:
                src = f.get("source_doc_id") or occ_id
                page = f.get("page", "?")
                cat = f.get("category", "finding")
                text = (f.get("text") or "").replace("\n", " ").strip()
                reg = f" [cites CAR {f['cites_reg']}]" if f.get("cites_reg") else ""
                out.append(f"  [{src} p.{page}] {cat}: {text}{reg}")
            for r in recs:
                src = r.get("source_doc_id") or occ_id
                page = r.get("page", "?")
                rid = f" ({r['id']})" if r.get("id") else ""
                text = (r.get("text") or "").replace("\n", " ").strip()
                out.append(f"  [{src} p.{page}] recommendation{rid}: {text}")
            if direct_regs:
                out.append(f"  cited regulations: {', '.join(direct_regs)}")
            if acs_ref:
                out.append(f"  referenced ACs: {', '.join(acs_ref)}")
        else:
            # Minimal fallback before extraction has run
            url = row.get("occ_url") or row.get("source_url") or ""
            out.append(f"- Occurrence {occ_id}  {url}".rstrip())

    return "\n".join(out)


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
