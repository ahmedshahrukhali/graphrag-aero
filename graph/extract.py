"""Entity extraction from chunk text.

Two layers:
  RegexExtractor  — deterministic, offline-safe. Pulls CAR/AC/TSB-rec citations
                    from every chunk and detects the 6 TSB section headers
                    (EN+FR) that signal findings/recommendations text.
  LLMExtractor    — uses the agent LLM Protocol to parse finding/recommendation
                    prose from section-bearing chunks. Only runs when a section
                    header is present; returns empty on all other chunks.
  HybridExtractor — runs RegexExtractor on every chunk; adds LLMExtractor on
                    section-bearing chunks. This is the production extractor.

All extractors implement the EntityExtractor Protocol so callers are decoupled.
Tests inject stubs for the LLM; the regex layer has no external deps.

ExtractedEntities structure
---------------------------
  regulations:      list[str]   canonical CAR numbers, e.g. "602.115"
  advisory_circulars: list[str] canonical AC numbers, e.g. "702-001"
  findings:         list[dict]  {text, category, lang}
                      category: "cause" | "risk" | "safety_action"
  recommendations:  list[dict]  {id: str|None, text: str, lang: str}
                      id is a TSB rec id (e.g. "A19-01") when present
  aircraft:         list[str]   (reserved — not populated yet)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol, Sequence

from embed.jsonl import ChunkRecord


logger = logging.getLogger(__name__)


# ─── data types ──────────────────────────────────────────────────────────────

class ExtractedEntities(dict):
    """Bag of extracted entities for a single chunk.

    Keys (all optional):
        regulations:        list[str]
        advisory_circulars: list[str]
        findings:           list[dict]   [{text, category, lang}]
        recommendations:    list[dict]   [{id, text, lang}]
        aircraft:           list[str]
    """


class EntityExtractor(Protocol):
    def extract(self, chunk: ChunkRecord) -> ExtractedEntities: ...


class NoopExtractor:
    def extract(self, chunk: ChunkRecord) -> ExtractedEntities:  # noqa: ARG002
        return ExtractedEntities()


def extract_all(
    extractor: EntityExtractor, chunks: Sequence[ChunkRecord],
) -> list[ExtractedEntities]:
    return [extractor.extract(c) for c in chunks]


# ─── patterns ────────────────────────────────────────────────────────────────

# CAR citations: "CAR 602.115", "CARs 507.20", "CAR507.20" — capture the number
_CAR = re.compile(r"\bCARs?\s*(\d{3}(?:\.\d+)+)", re.I)
# Bare part reference without "CAR" prefix — e.g. "section 602.115 of the CARs"
_CAR_SECTION = re.compile(r"\bsection\s+(\d{3}\.\d+)(?:\s+of\s+the\s+CARs?)?", re.I)

# AC citations: "AC 702-001", "AC702-001"
_AC = re.compile(r"\bAC\s*(\d{3}-\d{3})\b", re.I)

# TSB recommendation ids: "A99-03", "A19-01"
_TSB_REC_ID = re.compile(r"\b(A\d{2}-\d{2})\b")

# TSB section headers — appear mid-chunk; match from header to end of chunk
_SECTION_PATTERNS: dict[str, tuple[re.Pattern, str, str]] = {
    # (pattern, category, lang)
    "en_cause":   (re.compile(r"findings as to causes?(?:\s+and contributing factors)?", re.I),
                   "cause", "en"),
    "en_risk":    (re.compile(r"findings as to risk", re.I), "risk", "en"),
    "en_safety":  (re.compile(r"safety action\b", re.I), "safety_action", "en"),
    "fr_cause":   (re.compile(r"faits\s+établis\s+quant\s+aux\s+causes?", re.I),
                   "cause", "fr"),
    "fr_risk":    (re.compile(r"faits\s+établis\s+quant\s+aux\s+risques?", re.I),
                   "risk", "fr"),
    "fr_safety":  (re.compile(r"mesures\s+de\s+sécurité\b", re.I),
                   "safety_action", "fr"),
}

# Numbered / bulleted list items after a section header
_LIST_ITEM = re.compile(r"(?:^|\n)\s*(?:\d+\.|•|-|\*)\s+(.+?)(?=\n\s*(?:\d+\.|•|-|\*|\Z)|\Z)",
                        re.S)


# ─── regex extractor ─────────────────────────────────────────────────────────

class RegexExtractor:
    """Deterministic extraction of citations + section detection from any chunk."""

    def extract(self, chunk: ChunkRecord) -> ExtractedEntities:
        text = chunk.text
        ents = ExtractedEntities()

        # Regulations
        regs = list({m.group(1) for m in _CAR.finditer(text)})
        regs += [m.group(1) for m in _CAR_SECTION.finditer(text)
                 if m.group(1) not in regs]
        if regs:
            ents["regulations"] = regs

        # Advisory circulars
        acs = list({m.group(1) for m in _AC.finditer(text)})
        if acs:
            ents["advisory_circulars"] = acs

        # Section-bearing chunks: extract numbered list items as raw findings
        findings: list[dict] = []
        for _key, (rx, category, lang) in _SECTION_PATTERNS.items():
            m = rx.search(text)
            if not m:
                continue
            section_text = text[m.start():]
            for item in _LIST_ITEM.finditer(section_text):
                t = item.group(1).strip()
                if len(t) > 20:  # skip stubs
                    findings.append({"text": t, "category": category,
                                     "lang": chunk.lang})
        if findings:
            ents["findings"] = findings

        # TSB recommendation IDs cross-referenced in any chunk
        rec_ids = list({m.group(1) for m in _TSB_REC_ID.finditer(text)})
        if rec_ids:
            ents["recommendations"] = [{"id": rid, "text": "", "lang": chunk.lang}
                                        for rid in rec_ids]

        return ents


# ─── LLM extractor ───────────────────────────────────────────────────────────

_LLM_SYSTEM = (
    "You are a structured data extractor for aviation safety reports. "
    "Respond with valid JSON only — no prose, no markdown fences."
)

_LLM_PROMPT_TMPL = """\
Extract safety findings and recommendations from the following aviation report chunk.

Rules:
- findings: numbered statements under "Findings as to causes/risk" or "Safety Action" sections.
  Each finding is one self-contained sentence.
- recommendations: statements under recommendation sections. Include the TSB recommendation
  id (e.g. A19-01) if present in the text, else null.
- Only extract what is explicitly stated. Do not infer or add information.
- If a section is absent, return an empty list.

Return ONLY this JSON structure:
{{
  "findings": [{{"text": "...", "category": "cause|risk|safety_action"}}],
  "recommendations": [{{"id": "A19-01 or null", "text": "..."}}]
}}

CHUNK:
{text}
"""


def _is_section_bearing(text: str) -> bool:
    return any(rx.search(text) for _, (rx, _, _) in _SECTION_PATTERNS.items())


def _parse_llm_json(raw: str) -> dict:
    """Extract the first JSON object from raw LLM output."""
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a {...} block
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    logger.debug("LLMExtractor: could not parse JSON from response: %r", raw[:200])
    return {}


class LLMExtractor:
    """Uses the agent LLM Protocol to extract findings/recommendations from
    section-bearing chunks. Non-section chunks are skipped (returns empty).

    Callers must hold the LLM in memory before calling — this extractor is
    designed for offline batch extraction (upsert-graph), not query serving.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    def extract(self, chunk: ChunkRecord) -> ExtractedEntities:
        if not _is_section_bearing(chunk.text):
            return ExtractedEntities()

        prompt = _LLM_PROMPT_TMPL.format(text=chunk.text[:3000])
        try:
            raw = self._llm.chat(_LLM_SYSTEM, prompt)
        except Exception as exc:
            logger.warning("LLMExtractor: LLM call failed for %s p.%s: %s",
                           chunk.doc_id, chunk.page, exc)
            return ExtractedEntities()

        parsed = _parse_llm_json(raw)
        ents = ExtractedEntities()

        findings = []
        for item in parsed.get("findings") or []:
            if isinstance(item, dict) and item.get("text"):
                findings.append({
                    "text": str(item["text"]).strip(),
                    "category": str(item.get("category", "cause")),
                    "lang": chunk.lang,
                })
        if findings:
            ents["findings"] = findings

        recs = []
        for item in parsed.get("recommendations") or []:
            if isinstance(item, dict) and item.get("text"):
                rid = item.get("id")
                # Validate TSB rec id format; discard if malformed
                if rid and not _TSB_REC_ID.fullmatch(str(rid).strip()):
                    rid = None
                recs.append({
                    "id": str(rid).strip() if rid else None,
                    "text": str(item["text"]).strip(),
                    "lang": chunk.lang,
                })
        if recs:
            ents["recommendations"] = recs

        return ents


# ─── hybrid extractor ────────────────────────────────────────────────────────

class HybridExtractor:
    """Regex extraction on every chunk + LLM extraction on section chunks.

    Merges both results: LLM findings/recommendations take precedence over
    regex list-item findings (higher quality prose); regex citations
    (regulations, advisory_circulars, rec IDs) are always kept.
    """

    def __init__(self, llm) -> None:
        self._regex = RegexExtractor()
        self._llm = LLMExtractor(llm)

    def extract(self, chunk: ChunkRecord) -> ExtractedEntities:
        rx = self._regex.extract(chunk)
        ents = ExtractedEntities(rx)  # start from regex results

        if _is_section_bearing(chunk.text):
            llm_ents = self._llm.extract(chunk)
            # LLM findings override regex list items only when non-empty.
            # Empty LLM response (bad JSON, LLM skip) keeps regex findings.
            if llm_ents.get("findings"):
                ents["findings"] = llm_ents["findings"]
            if llm_ents.get("recommendations"):
                # Merge: LLM recs have text; regex recs have id only
                # Combine by id dedup, preferring LLM text
                llm_by_id: dict = {}
                for r in llm_ents["recommendations"]:
                    if r.get("id"):
                        llm_by_id[r["id"]] = r
                    else:
                        llm_by_id[f"__anon_{len(llm_by_id)}"] = r
                rx_recs = {r["id"]: r for r in ents.get("recommendations", [])
                           if r.get("id")}
                merged = {**rx_recs, **llm_by_id}
                ents["recommendations"] = list(merged.values())

        return ents
