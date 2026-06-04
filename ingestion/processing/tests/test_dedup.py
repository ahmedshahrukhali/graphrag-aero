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


def test_dedup_thread_safe_exactly_one_winner():
    """Concurrent is_new calls for the same hash: exactly one thread sees True."""
    import threading

    d = Dedup()
    h = chunk_hash("shared boilerplate")
    winners: list[bool] = []
    lock = threading.Lock()

    def _probe():
        result = d.is_new(h)
        with lock:
            winners.append(result)

    threads = [threading.Thread(target=_probe) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert winners.count(True) == 1
    assert winners.count(False) == 19
