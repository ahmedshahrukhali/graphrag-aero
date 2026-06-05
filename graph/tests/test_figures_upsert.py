"""Tests for upsert_figures in graph/upsert.py — offline, FakeDriver only."""
from __future__ import annotations

from ingestion.processing.figures import FigureRecord
from graph.upsert import upsert_figures


class FakeSession:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def run(self, cypher, **params):
        self.log.append((cypher, dict(params)))
        return iter(())


class FakeDriver:
    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def session(self, **kwargs):
        return FakeSession(self.runs)

    def close(self):
        pass


def _fig(doc_id="tsb/a13q0098", page=3, bbox=None, caption="Fuel gauge.", ocr_text="130 GAL"):
    if bbox is None:
        bbox = [10.0, 20.0, 200.0, 150.0]
    return FigureRecord(doc_id=doc_id, page=page, bbox=bbox,
                        caption=caption, ocr_text=ocr_text)


# ─── upsert_figures ───────────────────────────────────────────────────────────

def test_upsert_figures_returns_count():
    driver = FakeDriver()
    figs = [_fig(), _fig("tsb/a00a0051", page=1)]
    n = upsert_figures(driver, figs)
    assert n == 2


def test_upsert_figures_empty_is_noop():
    driver = FakeDriver()
    n = upsert_figures(driver, [])
    assert n == 0
    # No Cypher should be executed for an empty list
    assert driver.runs == []


def test_upsert_figures_runs_figure_cypher():
    driver = FakeDriver()
    upsert_figures(driver, [_fig()])
    cyphers = [r[0] for r in driver.runs]
    # Must execute the figure MERGE
    assert any("Figure" in c for c in cyphers)


def test_upsert_figures_runs_has_figure_edge_for_tsb():
    driver = FakeDriver()
    upsert_figures(driver, [_fig("tsb/a13q0098")])
    cyphers = [r[0] for r in driver.runs]
    # Must wire the HAS_FIGURE edge
    assert any("HAS_FIGURE" in c for c in cyphers)


def test_upsert_figures_no_has_figure_for_tc():
    """TC docs don't have Occurrence nodes → no HAS_FIGURE edge should be attempted."""
    driver = FakeDriver()
    upsert_figures(driver, [_fig("tc/AC_702-001_ISSUE-1")])
    cyphers = [r[0] for r in driver.runs]
    # Figure node is still upserted…
    assert any("Figure" in c for c in cyphers)
    # …but HAS_FIGURE is NOT attempted for a TC doc
    assert not any("HAS_FIGURE" in c for c in cyphers)


def test_upsert_figures_node_row_shape():
    driver = FakeDriver()
    fig = _fig()
    upsert_figures(driver, [fig])
    # Find the UPSERT_FIGURE run
    fig_run = next(
        (r for r in driver.runs if "fig.doc_id" in r[0] or "fig.caption" in r[0]),
        None,
    )
    assert fig_run is not None
    rows = fig_run[1]["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == fig.figure_id
    assert row["doc_id"] == "tsb/a13q0098"
    assert row["page"] == 3
    assert row["bbox"] == [10.0, 20.0, 200.0, 150.0]
    assert row["caption"] == "Fuel gauge."
    assert row["ocr_text"] == "130 GAL"


def test_upsert_figures_idempotency_same_id():
    """Two FigureRecords with the same (doc_id, page, bbox) produce the same id."""
    fig1 = _fig(page=5, bbox=[0.0, 0.0, 100.0, 100.0])
    fig2 = _fig(page=5, bbox=[0.0, 0.0, 100.0, 100.0])
    assert fig1.figure_id == fig2.figure_id


def test_upsert_figures_different_pages_different_ids():
    fig1 = _fig(page=1, bbox=[0.0, 0.0, 100.0, 100.0])
    fig2 = _fig(page=2, bbox=[0.0, 0.0, 100.0, 100.0])
    assert fig1.figure_id != fig2.figure_id


def test_upsert_figures_deduplicates_has_figure_links():
    """Same (occ_id, fig_id) must not produce duplicate link rows."""
    driver = FakeDriver()
    fig = _fig()
    # Pass the same figure twice
    upsert_figures(driver, [fig, fig])
    # Find the HAS_FIGURE run
    link_run = next(
        (r for r in driver.runs if "HAS_FIGURE" in r[0]),
        None,
    )
    assert link_run is not None
    # Dedup: only one link row despite two identical FigureRecords
    assert len(link_run[1]["rows"]) == 1


def test_upsert_figures_mixed_tsb_and_tc():
    driver = FakeDriver()
    tsb_fig = _fig("tsb/a001")
    tc_fig = _fig("tc/AC_702-001_ISSUE-1", bbox=[5.0, 5.0, 200.0, 200.0])
    n = upsert_figures(driver, [tsb_fig, tc_fig])
    assert n == 2
    cyphers = [r[0] for r in driver.runs]
    # HAS_FIGURE only for TSB
    has_figure_runs = [r for r in driver.runs if "HAS_FIGURE" in r[0]]
    assert len(has_figure_runs) == 1
    assert has_figure_runs[0][1]["rows"][0]["occ_id"] == "a001"
