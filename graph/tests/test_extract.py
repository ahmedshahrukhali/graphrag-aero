"""Tests for graph/extract.py — RegexExtractor, LLMExtractor, HybridExtractor."""
import json

from embed.jsonl import ChunkRecord
from graph.extract import (
    ExtractedEntities,
    HybridExtractor,
    LLMExtractor,
    NoopExtractor,
    RegexExtractor,
    _is_section_bearing,
    extract_all,
)


def _rec(text: str, *, doc_id: str = "tsb/a01", lang: str = "en",
         page: int = 1) -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id, source_url=None, section_title="", page=page,
        bbox=[0.0, 0.0, 0.0, 0.0], chunk_hash="0" * 64, lang=lang, text=text,
    )


# ─── NoopExtractor / extract_all ─────────────────────────────────────────────

def test_noop_returns_empty_bag():
    ent = NoopExtractor().extract(_rec("x"))
    assert ent == ExtractedEntities()
    assert isinstance(ent, dict)


def test_extract_all_runs_one_per_chunk():
    out = extract_all(NoopExtractor(), [_rec("a"), _rec("b"), _rec("c")])
    assert len(out) == 3
    assert all(isinstance(e, ExtractedEntities) for e in out)


# ─── _is_section_bearing ─────────────────────────────────────────────────────

def test_section_bearing_en_cause():
    assert _is_section_bearing("Findings as to Causes and Contributing Factors\n1. The pilot")


def test_section_bearing_en_risk():
    assert _is_section_bearing("3.2 Findings as to Risk\n1. CVR capacity")


def test_section_bearing_fr_cause():
    assert _is_section_bearing("Faits établis quant aux causes et aux facteurs contribuants")


def test_section_bearing_fr_risk():
    assert _is_section_bearing("Faits établis quant aux risques\n1. Si les pilotes")


def test_section_bearing_safety_action():
    assert _is_section_bearing("Safety Action Taken\nTransport Canada has reviewed")


def test_section_bearing_negative():
    assert not _is_section_bearing("The aircraft departed at 08:00. Weather was VMC.")


# ─── RegexExtractor ──────────────────────────────────────────────────────────

def test_regex_extracts_car_citation():
    rx = RegexExtractor()
    ents = rx.extract(_rec("The pilot violated CAR 602.115(a) on fuel reserves."))
    assert "regulations" in ents
    assert "602.115" in ents["regulations"]


def test_regex_extracts_car_section_reference():
    rx = RegexExtractor()
    ents = rx.extract(_rec("Section 602.115 of the CARs requires sufficient fuel."))
    assert "regulations" in ents
    assert "602.115" in ents["regulations"]


def test_regex_extracts_ac_citation():
    rx = RegexExtractor()
    ents = rx.extract(_rec("Refer to AC 700-027 for guidance on fatigue."))
    assert "advisory_circulars" in ents
    assert "700-027" in ents["advisory_circulars"]


def test_regex_extracts_tsb_rec_id():
    rx = RegexExtractor()
    ents = rx.extract(_rec("TSB recommendation A19-01 was issued following this finding."))
    assert "recommendations" in ents
    assert any(r["id"] == "A19-01" for r in ents["recommendations"])


def test_regex_extracts_multiple_cars():
    rx = RegexExtractor()
    ents = rx.extract(_rec("CAR 602.88 and CAR 602.115 both apply here."))
    assert set(ents["regulations"]) >= {"602.88", "602.115"}


def test_regex_deduplicates_citations():
    rx = RegexExtractor()
    ents = rx.extract(_rec("CAR 602.115, see also CAR 602.115 and CAR 602.115."))
    assert ents["regulations"].count("602.115") == 1


def test_regex_no_citations_returns_empty():
    rx = RegexExtractor()
    ents = rx.extract(_rec("The weather was clear with 15 miles visibility."))
    assert ents == ExtractedEntities()


def test_regex_extracts_finding_from_section():
    text = (
        "Findings as to Causes and Contributing Factors\n"
        "1. The fuel selector valve was in the off position.\n"
        "2. The pilot failed to complete the pre-flight checklist.\n"
    )
    rx = RegexExtractor()
    ents = rx.extract(_rec(text))
    assert "findings" in ents
    assert len(ents["findings"]) == 2
    assert ents["findings"][0]["category"] == "cause"
    assert ents["findings"][0]["lang"] == "en"


def test_regex_extracts_fr_finding():
    text = (
        "Faits établis quant aux risques\n"
        "1. Si des vols VFR de nuit sont effectués sans équipement minimum, "
        "les pilotes courent un risque de désorientation.\n"
    )
    rx = RegexExtractor()
    ents = rx.extract(_rec(text, lang="fr"))
    assert "findings" in ents
    assert ents["findings"][0]["category"] == "risk"
    assert ents["findings"][0]["lang"] == "fr"


# ─── LLMExtractor ────────────────────────────────────────────────────────────

class StubLLM:
    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def chat(self, system, user):
        self.calls.append((system, user))
        return self.response


def test_llm_extractor_skips_nonsection_chunks():
    llm = StubLLM()
    ex = LLMExtractor(llm)
    ents = ex.extract(_rec("Weather was VMC. No issues encountered."))
    assert ents == ExtractedEntities()
    assert llm.calls == []  # no LLM call on non-section chunk


def test_llm_extractor_parses_finding_json():
    resp = json.dumps({
        "findings": [{"text": "Fuel tanks were empty on arrival.", "category": "cause"}],
        "recommendations": [],
    })
    llm = StubLLM(resp)
    ex = LLMExtractor(llm)
    text = "Findings as to Causes and Contributing Factors\n1. Fuel tanks were empty."
    ents = ex.extract(_rec(text))
    assert len(ents["findings"]) == 1
    assert "Fuel tanks" in ents["findings"][0]["text"]


def test_llm_extractor_parses_recommendation_with_id():
    resp = json.dumps({
        "findings": [],
        "recommendations": [{"id": "A19-01", "text": "Install TAWS on all aircraft."}],
    })
    llm = StubLLM(resp)
    ex = LLMExtractor(llm)
    text = "Safety Action Taken\nRecommendation A19-01 was issued."
    ents = ex.extract(_rec(text))
    assert len(ents["recommendations"]) == 1
    assert ents["recommendations"][0]["id"] == "A19-01"


def test_llm_extractor_rejects_malformed_rec_id():
    resp = json.dumps({
        "findings": [],
        "recommendations": [{"id": "NOT_A_REC_ID", "text": "Some action."}],
    })
    llm = StubLLM(resp)
    ex = LLMExtractor(llm)
    text = "Safety Action Taken\nSome action was taken."
    ents = ex.extract(_rec(text))
    assert ents["recommendations"][0]["id"] is None


def test_llm_extractor_handles_markdown_fenced_json():
    resp = "```json\n" + json.dumps({
        "findings": [{"text": "Engine failed.", "category": "cause"}],
        "recommendations": [],
    }) + "\n```"
    llm = StubLLM(resp)
    ex = LLMExtractor(llm)
    text = "Findings as to Causes and Contributing Factors\n1. Engine failed."
    ents = ex.extract(_rec(text))
    assert len(ents["findings"]) == 1


def test_llm_extractor_handles_bad_json_gracefully():
    llm = StubLLM("this is not JSON at all")
    ex = LLMExtractor(llm)
    text = "Findings as to Risk\n1. Some risk."
    ents = ex.extract(_rec(text))
    assert ents == ExtractedEntities()  # bad parse → empty, no crash


def test_llm_extractor_handles_llm_exception():
    class BrokenLLM:
        def chat(self, *_):
            raise RuntimeError("Ollama down")

    ex = LLMExtractor(BrokenLLM())
    text = "Findings as to Risk\n1. CVR insufficient."
    ents = ex.extract(_rec(text))
    assert ents == ExtractedEntities()


# ─── HybridExtractor ─────────────────────────────────────────────────────────

def test_hybrid_combines_regex_citations_with_llm_findings():
    text = (
        "Findings as to Causes and Contributing Factors\n"
        "1. The pilot failed to maintain fuel awareness per CAR 602.88.\n"
        "AC 700-027 provides guidance.\n"
    )
    llm_resp = json.dumps({
        "findings": [{"text": "Pilot failed to maintain fuel awareness.", "category": "cause"}],
        "recommendations": [],
    })
    llm = StubLLM(llm_resp)
    ex = HybridExtractor(llm)
    ents = ex.extract(_rec(text))
    # LLM findings should be present
    assert "findings" in ents
    assert any("fuel" in f["text"].lower() for f in ents["findings"])
    # Regex citations preserved
    assert "602.88" in ents.get("regulations", [])
    assert "700-027" in ents.get("advisory_circulars", [])


def test_hybrid_llm_findings_override_regex_list_items():
    text = (
        "Findings as to Causes\n"
        "1. Short list item A.\n"
        "2. Short list item B.\n"
    )
    llm_resp = json.dumps({
        "findings": [{"text": "LLM richer finding text here.", "category": "cause"}],
        "recommendations": [],
    })
    ex = HybridExtractor(StubLLM(llm_resp))
    ents = ex.extract(_rec(text))
    texts = [f["text"] for f in ents["findings"]]
    assert any("LLM richer" in t for t in texts)
    # Regex list items should not appear (LLM overrode them)
    assert not any("Short list item" in t for t in texts)


def test_hybrid_no_llm_call_on_nonsection_chunk():
    llm = StubLLM()
    ex = HybridExtractor(llm)
    ex.extract(_rec("Normal narrative text. No section headers here."))
    assert llm.calls == []
