from ingestion.processing.dedup import Dedup, chunk_hash, normalize_for_hash


def test_normalize_collapses_whitespace_and_lowercases():
    assert normalize_for_hash("  Hello\nWORLD\t  ") == "hello world"


def test_chunk_hash_is_stable_across_whitespace_and_case():
    a = chunk_hash("Findings: Stall at low altitude.")
    b = chunk_hash("  findings:    stall at low altitude. ")
    assert a == b


def test_chunk_hash_differs_for_different_text():
    assert chunk_hash("alpha") != chunk_hash("beta")


def test_dedup_keeps_first_and_drops_repeats():
    d = Dedup()
    h = chunk_hash("boilerplate")
    assert d.is_new(h) is True
    assert d.is_new(h) is False
    assert d.is_new(h) is False
    assert len(d) == 1


def test_dedup_independent_across_instances():
    d1, d2 = Dedup(), Dedup()
    h = chunk_hash("x")
    assert d1.is_new(h) is True
    # A fresh Dedup hasn't seen it.
    assert d2.is_new(h) is True
