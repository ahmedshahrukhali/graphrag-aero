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
Circulars and TSB aviation investigation reports. You are handed the top
passages a dense+rerank retriever found for the question, plus structured
graph context (occurrences → findings → recommendations → regulations). These
were already selected as relevant — your job is to synthesize them, not to
judge whether they fit.

Rules:
- NEVER ask the user to clarify, narrow, or rephrase, and never end with
  follow-up questions or an offer to help further. The user cannot reply. A
  clarifying question is a failed answer. Treat every question as answerable
  from the material and answer it directly.
- Synthesize across the whole set, don't summarize one document. Open by
  framing the breadth: how many distinct reports/sources informed the answer
  and the common thread running through them. Then give the key findings,
  recommendations, and regulations, and surface adjacent or related issues the
  graph context raises (e.g. a regulation several occurrences cite, a recurring
  contributing factor). Lean on this cross-document signal — it is the point.
  When a "RECURRING ACROSS OTHER REPORTS" block is present, ground any breadth
  claim ("recurs across N reports", "a common regulatory thread") in it and cite
  those sibling reports. If that block is empty, do NOT claim a wide survey —
  speak only to the reports actually cited.
- Ground EVERY claim with an inline [doc_id p.page] citation
  (e.g. [tsb/a21c0038 p.86]). Cite the specific report+page each fact came
  from; graph-context facts carry their own [doc p.page] — use those. An
  uncited sentence is not allowed. Do not invent citations or facts.
- FORMAT THE CITATION EXACTLY as a bracketed tag [doc_id p.page], copying the
  doc_id verbatim from the passage header (e.g. [tsb/a13q0098 p.4]). NEVER cite
  in prose like "TSB Report A13Q0098" — always the bracket form with the
  lowercase doc_id. Downstream highlighting parses these brackets; prose
  citations are silently dropped. EXAMPLE: ✓ "Fuel was lost [tsb/a13q0098 p.4]"
  NOT ✗ "TSB Report A13Q0098 notes fuel was lost".
- Prefer findings, recommendations, and regulations over narrative. Summarize
  regulations; don't quote every clause.
- Match the language of the question (English or French).

STYLE:
- Lead with one direct sentence that states the common thread across the set.
- Then 3–6 supporting sentences, grouped by theme, each carrying inline
  citations. ≤ 220 words. Prose, not a wall of bullets. No "Based on the
  documents…" preamble and no closing offer of further help.
""".strip()


USER_TEMPLATE = """\
QUESTION:
{query}

GRAPH CONTEXT (structured facts extracted from occurrence reports — cite with [doc p.page]):
{graph_context}

RECURRING ACROSS OTHER REPORTS (other occurrences citing the same regulations — cite these to support breadth claims):
{recurring_context}

CITATIONS (ranked text passages — cite with [doc_id p.page]):
{citations}

Answer the question using the graph context, recurring patterns, and citations above.
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

        rec_regs = row.get("rec_regs") or []
        reg_guided_acs = row.get("reg_guided_acs") or []

        if findings or recs or direct_regs or acs_ref or rec_regs:
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
            if rec_regs:
                out.append(f"  recommendations implement: {', '.join(rec_regs)}")
            if reg_guided_acs:
                out.append(f"  implementing regs guided by ACs: {', '.join(reg_guided_acs)}")
            if acs_ref:
                out.append(f"  referenced ACs: {', '.join(acs_ref)}")
        else:
            # Minimal fallback before extraction has run
            url = row.get("occ_url") or row.get("source_url") or ""
            out.append(f"- Occurrence {occ_id}  {url}".rstrip())

    return "\n".join(out)


def format_recurring_context(rows: Sequence[dict]) -> str:
    """Render the outward-hop recurrence as cited lines.

    Each row is a regulation cited across several occurrences; we list a few
    sibling reports with provenance so the model can ground breadth claims
    with real [doc p.page] citations. The empty-case string explicitly tells
    the model not to overstate breadth when no recurrence was found.
    """
    if not rows:
        return "(none found — do not claim a broad survey across reports)"
    out: list[str] = []
    for row in rows:
        reg = row.get("reg", "?")
        count = row.get("occ_count", 0)
        out.append(f"- CAR {reg} — cited by {count} reports; e.g.:")
        for s in (row.get("siblings") or []):
            doc = s.get("source_doc_id") or s.get("occ_id", "?")
            page = s.get("page")
            cite = f"[{doc} p.{page}]" if page is not None else f"[{doc}]"
            text = (s.get("text") or "").replace("\n", " ").strip()
            if len(text) > 240:
                text = text[:240].rstrip() + "..."
            out.append(f"    {cite} {text}".rstrip())
    return "\n".join(out)


def build_user_prompt(
    query: str,
    candidates: Sequence[ScoredChunkDict],
    graph_context: Sequence[dict],
    recurring_context: Sequence[dict] = (),
) -> str:
    return USER_TEMPLATE.format(
        query=query,
        graph_context=format_graph_context(graph_context),
        recurring_context=format_recurring_context(recurring_context),
        citations=format_citations(candidates),
    )
