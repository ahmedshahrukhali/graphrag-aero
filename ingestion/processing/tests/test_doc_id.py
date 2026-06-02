from pathlib import Path

import pytest

from ingestion.processing.doc_id import doc_ref_for_path


def test_tsb_en_doc_ref():
    ref = doc_ref_for_path(Path("data/corpus/en/tsb/a00a0110.pdf"))
    assert ref.doc_id == "tsb/a00a0110"
    assert ref.source == "tsb"
    assert ref.stem == "a00a0110"
    assert ref.lang == "en"
    assert ref.source_url == (
        "https://www.bst-tsb.gc.ca/sites/default/files/"
        "rapports-reports/aviation/a00a0110/eng/a00a0110.pdf"
    )


def test_tsb_fr_doc_ref_uses_fra_path():
    ref = doc_ref_for_path(Path("data/corpus/fr/tsb/a00a0110.pdf"))
    assert ref.lang == "fr"
    assert "/fra/" in ref.source_url


def test_tc_doc_ref_has_no_source_url():
    ref = doc_ref_for_path(Path("data/corpus/en/tc/AC_100-001_e08_20210622.pdf"))
    assert ref.doc_id == "tc/AC_100-001_e08_20210622"
    assert ref.source == "tc"
    assert ref.lang == "en"
    # We can't reconstruct the TC URL from the stem alone; that's documented.
    assert ref.source_url is None


def test_corpus_tag_mirrors_source_for_tsb_tc():
    """WS-0: corpus is a first-class tag; == source for the EN/FR TSB+TC corpus
    (the ZH axis will mint 'caac')."""
    assert doc_ref_for_path(Path("data/corpus/en/tsb/a00a0110.pdf")).corpus == "tsb"
    assert doc_ref_for_path(Path("data/corpus/en/tc/AC_100-001_e08.pdf")).corpus == "tc"


def test_zh_ttsb_doc_ref():
    ref = doc_ref_for_path(Path("data/corpus/zh/ttsb/asc-aor-2001.pdf"))
    assert ref.doc_id == "ttsb/asc-aor-2001"
    assert ref.source == "ttsb"
    assert ref.lang == "zh"
    assert ref.corpus == "ttsb"
    # No URL builder yet (TC-style) — the scraper adds it later.
    assert ref.source_url is None


def test_zh_caac_doc_ref():
    ref = doc_ref_for_path(Path("data/corpus/zh/caac/ac-121-fs.pdf"))
    assert ref.doc_id == "caac/ac-121-fs"
    assert ref.source == "caac"
    assert ref.lang == "zh"
    assert ref.corpus == "caac"
    assert ref.source_url is None


def test_rejects_non_pdf():
    with pytest.raises(ValueError):
        doc_ref_for_path(Path("data/corpus/en/tsb/a00a0110.txt"))


def test_rejects_unknown_source():
    with pytest.raises(ValueError):
        doc_ref_for_path(Path("data/corpus/en/faa/anything.pdf"))
