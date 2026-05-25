"""Map ``chunk_hash`` (sha256 hex) → deterministic UUID for Qdrant point IDs.

Qdrant accepts unsigned-int or UUID point IDs; sha256 hex isn't either. We take
the first 128 bits of the hash and wrap them as a UUID. Deterministic, so a
re-run upserts in place instead of producing duplicates.
"""
from __future__ import annotations

import uuid


def point_id_for(chunk_hash: str) -> str:
    """``chunk_hash`` is sha256 hex (64 chars). Returns a stable UUID string."""
    if len(chunk_hash) < 32:
        raise ValueError(f"chunk_hash too short: {len(chunk_hash)} chars (need ≥32)")
    return str(uuid.UUID(bytes=bytes.fromhex(chunk_hash[:32])))
