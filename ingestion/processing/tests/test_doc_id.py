from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.acquisition import caac as caac_mod
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


def test_zh_ttsb_doc_ref_legacy_stem_no_url():
    # Stem without a numeric media-id prefix (e.g. legacy ASC naming) — graceful None.
    ref = doc_ref_for_path(Path("data/corpus/zh/ttsb/asc-aor-2001.pdf"))
    assert ref.doc_id == "ttsb/asc-aor-2001"
    assert ref.source == "ttsb"
    assert ref.lang == "zh"
    assert ref.corpus == "ttsb"
    assert ref.source_url is None


def test_zh_ttsb_doc_ref_with_media_url():
    # stem = "{media_id}_{name}" → reconstructable URL.
    ref = doc_ref_for_path(Path("data/corpus/zh/ttsb/9234_安捷b-86002調查報告.pdf"))
    assert ref.source_url == "https://www.ttsb.gov.tw/media/9234/安捷b-86002調查報告.pdf"


def test_zh_ttsb_doc_ref_ascii_name():
    ref = doc_ref_for_path(Path("data/corpus/zh/ttsb/3059_00_general.pdf"))
    assert ref.source_url == "https://www.ttsb.gov.tw/media/3059/00_general.pdf"


def test_zh_caac_doc_ref_not_in_seed():
    # Stem not in the seed → None (same as before).
    with patch.object(caac_mod, "load_seed_file", return_value=[]):
        ref = doc_ref_for_path(Path("data/corpus/zh/caac/ac-121-fs.pdf"))
    assert ref.doc_id == "caac/ac-121-fs"
    assert ref.source == "caac"
    assert ref.lang == "zh"
    assert ref.corpus == "caac"
    assert ref.source_url is None


def test_zh_caac_doc_ref_resolves_seed_url():
    seed_url = "https://www.caac.gov.cn/XXGK/XXGK/MHXH/201511/P020151103346484825446.pdf"
    with patch.object(caac_mod, "load_seed_file", return_value=[seed_url]):
        ref = doc_ref_for_path(Path("data/corpus/zh/caac/P020151103346484825446.pdf"))
    assert ref.source_url == seed_url


def test_rejects_non_pdf():
    with pytest.raises(ValueError):
        doc_ref_for_path(Path("data/corpus/en/tsb/a00a0110.txt"))


def test_rejects_unknown_source():
    with pytest.raises(ValueError):
        doc_ref_for_path(Path("data/corpus/en/faa/anything.pdf"))
