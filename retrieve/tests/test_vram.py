"""Tests for ModelSession. No torch dependency — cleanup is parametrised."""
import sys
import types

import pytest

from retrieve.vram import ModelSession, wait_for_free_vram


def _fake_torch(is_avail: bool, free_mib: int, total_mib: int = 8192):
    """A minimal fake ``torch`` exposing the cuda calls the barrier uses."""
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: is_avail,
        mem_get_info=lambda: (free_mib * 1024 * 1024, total_mib * 1024 * 1024),
    )
    return mod


def test_enter_loads_and_returns_model():
    sentinel = object()
    with ModelSession(lambda: sentinel) as m:
        assert m is sentinel


def test_property_before_enter_raises():
    s = ModelSession(lambda: object())
    with pytest.raises(RuntimeError):
        _ = s.model


def test_property_inside_with_returns_model():
    sentinel = object()
    s = ModelSession(lambda: sentinel)
    with s:
        assert s.model is sentinel


def test_exit_runs_cleanup():
    cleaned = []
    with ModelSession(lambda: "m", cleanup=lambda: cleaned.append(1)):
        pass
    assert cleaned == [1]


def test_exit_runs_cleanup_even_on_exception():
    cleaned = []
    with pytest.raises(RuntimeError):
        with ModelSession(lambda: "m", cleanup=lambda: cleaned.append(1)):
            raise RuntimeError("boom")
    assert cleaned == [1]


def test_exit_releases_model_reference():
    s = ModelSession(lambda: "m", cleanup=lambda: None)
    with s:
        pass
    # After exit, accessing .model should raise (model was released).
    with pytest.raises(RuntimeError):
        _ = s.model


def test_default_cleanup_is_noop_without_torch_or_gpu():
    """Default cleanup must not raise even when torch is absent / no CUDA."""
    # Just verifies the context manager runs end-to-end with the default hook.
    with ModelSession(lambda: object()):
        pass


def test_wait_for_free_vram_true_when_enough_free(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, free_mib=7500))
    assert wait_for_free_vram(7000, timeout_s=1.0) is True


def test_wait_for_free_vram_noop_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, free_mib=0))
    assert wait_for_free_vram(7000, timeout_s=1.0) is True


def test_wait_for_free_vram_times_out_when_busy(monkeypatch):
    # GPU stays busy (only 1 GiB free) → barrier gives up and returns False.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, free_mib=1024))
    assert wait_for_free_vram(7000, timeout_s=0.15, poll_s=0.05) is False
