"""Tests for ModelSession. No torch dependency — cleanup is parametrised."""
import pytest

from retrieve.vram import ModelSession


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
