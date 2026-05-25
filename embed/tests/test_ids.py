"""Tests for chunk_hash → UUID derivation."""
import uuid

import pytest

from embed.ids import point_id_for


def test_deterministic():
    h = "335f935eefb0bb072144a334b96c1f8b51fdbaa4d65e588863d34c24a94c3373"
    assert point_id_for(h) == point_id_for(h)


def test_returns_valid_uuid_string():
    h = "0b3678381a5ddaf500b17b9fd1804f9370ab9b79878c5f4d9eed82cade72db30"
    pid = point_id_for(h)
    # Will raise if not a valid UUID.
    uuid.UUID(pid)


def test_different_hashes_collide_only_on_first_128_bits():
    # Same first 32 hex chars, different tail → same UUID (expected; we only
    # use the first 128 bits). This isn't a problem in practice because sha256
    # collisions are astronomically rare on either half; we just document it.
    a = "a" * 32 + "0" * 32
    b = "a" * 32 + "f" * 32
    assert point_id_for(a) == point_id_for(b)


def test_distinct_for_distinct_prefixes():
    a = "0" * 32 + "ffff" * 8
    b = "1" * 32 + "ffff" * 8
    assert point_id_for(a) != point_id_for(b)


def test_too_short_raises():
    with pytest.raises(ValueError):
        point_id_for("abc")
