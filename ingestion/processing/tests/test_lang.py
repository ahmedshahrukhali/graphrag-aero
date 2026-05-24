from pathlib import Path

import pytest

from ingestion.processing.lang import lang_for_path


def test_lang_for_path_en():
    assert lang_for_path(Path("data/corpus/en/tsb/a23h0001.pdf")) == "en"


def test_lang_for_path_fr():
    assert lang_for_path(Path("data/corpus/fr/tc/AC_100-001_f08.pdf")) == "fr"


def test_lang_for_path_is_case_insensitive():
    assert lang_for_path(Path("DATA/CORPUS/EN/TSB/x.pdf")) == "en"


def test_lang_for_path_refuses_unknown():
    with pytest.raises(ValueError):
        lang_for_path(Path("data/corpus/de/tsb/x.pdf"))
