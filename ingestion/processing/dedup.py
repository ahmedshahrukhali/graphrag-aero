"""Cross-document chunk deduplication.

TSB reports share boilerplate ("About the investigation", standard disclaimers,
the TSB mission statement). Identical chunks add zero retrieval signal and
inflate the corpus. We drop any chunk whose ``chunk_hash`` has already been
emitted in this run.
"""
from __future__ import annotations

import hashlib
import re
import threading


_WS = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Collapse whitespace, strip, lowercase — so trivial reformatting collides."""
    return _WS.sub(" ", text).strip().lower()


def chunk_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


class Dedup:
    """In-memory set of chunk hashes seen so far in this run."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._seen)

    def is_new(self, h: str) -> bool:
        """Return True the first time ``h`` is seen; False every subsequent call."""
        with self._lock:
            if h in self._seen:
                return False
            self._seen.add(h)
            return True
