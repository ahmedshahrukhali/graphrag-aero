"""Language inference from corpus path.

We rely on the on-disk layout (`data/corpus/en/...` vs `data/corpus/fr/...`)
rather than language detection — aviation jargon trips language detectors and
the acquisition layer already partitioned by language at download time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

Lang = Literal["en", "fr", "zh"]


def lang_for_path(path: Path) -> Lang:
    """Return 'en', 'fr' or 'zh' based on whichever appears as a path segment.

    'zh' covers the Chinese corpus (TTSB Traditional + CAAC Simplified); the
    script variant is resolved downstream per-source, not from the path.
    Raises ``ValueError`` if none is present — we refuse to guess.
    """
    parts = [p.lower() for p in Path(path).parts]
    if "en" in parts:
        return "en"
    if "fr" in parts:
        return "fr"
    if "zh" in parts:
        return "zh"
    raise ValueError(f"cannot infer language from path: {path}")
