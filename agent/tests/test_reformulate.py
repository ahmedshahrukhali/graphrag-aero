"""Tests for agent.reformulate — offline, no model weights."""
import pytest

from agent.reformulate import reformulate, _tokenise, _query_tokens


# ─── _tokenise ───────────────────────────────────────────────────────────────

def test_tokenise_basic():
    toks = _tokenise("Fuel exhaustion forced landing")
    assert "fuel" in toks and "exhaustion" in toks and "forced" in toks


def test_tokenise_regulation_code():
    toks = _tokenise("CAR 602.115 fuel quantity check")
    assert "car" in toks
    assert "602.115" in toks


def test_tokenise_aircraft_registration():
    toks = _tokenise("aircraft C-FHGR collided")
    assert "c-fhgr" in toks


def test_tokenise_cjk():
    toks = _tokenise("飞行机组 CCAR-121")
    assert any(len(t) >= 1 for t in toks)


# ─── reformulate ─────────────────────────────────────────────────────────────

def _cand(text: str) -> dict:
    return {"text": text, "doc_id": "tsb/x", "chunk_hash": "abc"}


def test_reformulate_returns_query_when_no_candidates():
    q = "fuel exhaustion forced landing"
    assert reformulate(q, []) == q


def test_reformulate_appends_novel_terms():
    q = "fuel exhaustion forced landing"
    cands = [_cand("The flapper valve froze at low temperature. CAR 602.115 applies.")]
    result = reformulate(q, cands)
    assert result.startswith(q)
    # At least one novel token from the candidate should appear.
    assert len(result) > len(q)
    extra = result[len(q):].strip().split()
    assert len(extra) >= 1


def test_reformulate_does_not_repeat_query_terms():
    q = "fuel exhaustion forced landing"
    cands = [_cand("fuel exhaustion forced landing again fuel")]
    result = reformulate(q, cands)
    # Tokens already in the query must not appear in the expansion.
    extra_tokens = set(result[len(q):].strip().lower().split())
    query_tokens = {"fuel", "exhaustion", "forced", "landing"}
    assert extra_tokens.isdisjoint(query_tokens)


def test_reformulate_respects_top_n():
    q = "engine failure"
    cands = [_cand("flange crack propagation fatigue metallurgy fractography inspection")]
    result = reformulate(q, cands, top_n=3)
    extra = result[len(q):].strip().split()
    assert len(extra) <= 3


def test_reformulate_filters_short_tokens():
    q = "engine failure"
    cands = [_cand("it is a the de le")]  # all below min_len=3 or stop-words
    result = reformulate(q, cands, min_len=3)
    assert result == q


def test_reformulate_no_expansion_when_candidates_match_query():
    q = "fuel exhaustion forced landing canadian aviation regulation"
    # Everything in the candidate overlaps with the query.
    cands = [_cand("fuel exhaustion forced landing")]
    result = reformulate(q, cands)
    assert result == q


def test_reformulate_aviation_identifier_expansion():
    q = "engine failure after takeoff"
    cands = [_cand("The crew noted cracking in flange 43-B. See AC 700-001. Registration C-GABC.")]
    result = reformulate(q, cands)
    assert result.startswith(q)
    # Identifiers like "ac" or "c-gabc" or "700-001" should appear.
    extra = result[len(q):]
    assert any(tok in extra for tok in ["ac", "700-001", "c-gabc", "43-b", "flange"])
