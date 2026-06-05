"""Query reformulation for multi-hop retrieval.

Hop N>1 would otherwise re-run the identical query → identical ANN hits (the
``make_decide_continue`` loop was a no-op before §2).  This module extracts
high-salience, low-overlap terms from the top-ranked hop-1 candidates and
appends them to the original query, so hop 2 explores different index regions.

Algorithm (no ML required — offline-safe, zero model weight downloads):
  1. Pool the text of all candidates.
  2. Tokenise (word-level) + count term frequencies.
  3. Weight longer tokens higher as a specificity proxy.
  4. Penalise tokens already present in the original query.
  5. Return ``"{original} {top-N novel tokens}"``.

Aviation corpora are identifier-dense (regulation codes, aircraft registrations,
doc numbers) so longer tokens tend to be the most discriminating ones.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


# Common English / French function words that carry no retrieval signal.
_STOP = frozenset(
    "a an the and or of in to for on at by is was were are with from as its "
    "it this that these those been be have has had will would could should "
    "may also such more than while after before during between when where "
    "which who what how not no can all any some but if so les des du la le "
    "un une et ou de en au aux est sont".split()
)

# Matches whole tokens including hyphens and slashes (aviation codes like
# "AC 700-001", "CAR 602.115", "C-FHGR") and CJK ranges.
_TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ɏ一-鿿㐀-䶿0-9]"
    r"[A-Za-zÀ-ɏ一-鿿㐀-䶿0-9\-./]*"
)


def _tokenise(text: str) -> list[str]:
    return [m.lower() for m in _TOKEN_RE.findall(text)]


def _query_tokens(query: str) -> frozenset[str]:
    return frozenset(_tokenise(query))


def reformulate(
    query: str,
    candidates: Sequence[dict],
    *,
    top_n: int = 6,
    min_len: int = 3,
) -> str:
    """Return an expanded query: ``query`` + novel high-salience terms from ``candidates``.

    ``candidates`` is a list of ScoredChunkDict-shaped dicts (must have ``"text"``).
    Returns ``query`` unchanged when no useful expansion terms are found.

    ``top_n``  maximum expansion tokens to append.
    ``min_len`` minimum token character length (filters noise like "the", "de").
    """
    if not candidates:
        return query

    existing = _query_tokens(query)
    tf: Counter[str] = Counter()
    for c in candidates:
        for tok in _tokenise(c.get("text", "")):
            if tok in _STOP:
                continue
            if len(tok) < min_len:
                continue
            if tok in existing:
                continue
            # Weight by length as a specificity proxy: "ac700-001" > "fuel".
            tf[tok] += len(tok)

    novel = [tok for tok, _ in tf.most_common(top_n)]
    if not novel:
        return query
    return f"{query} {' '.join(novel)}"
