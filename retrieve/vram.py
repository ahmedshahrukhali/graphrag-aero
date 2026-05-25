"""Sequential VRAM discipline — a context manager that loads a model on enter
and releases the CUDA cache on exit.

3060Ti (8 GB) has just barely enough headroom for BGE-M3 (~0.5 GB) +
reranker-v2-m3 (~0.5 GB) + gemma2:9b Q4_K_M (~5.5 GB). Interactive callers
(the FastAPI backend in P6) keep models loaded across requests; batch / eval
callers (P5) wrap each stage in ``with ModelSession(...)`` to free VRAM
between stages.

Tests don't depend on torch — the cleanup hook is parametrisable.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Generic, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


def _empty_cuda_cache() -> None:
    """Best-effort CUDA cache release. No-op if torch isn't installed or no GPU."""
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelSession(Generic[T]):
    """Load a model with ``factory()`` on enter; drop it + free CUDA on exit.

    ``cleanup`` defaults to :func:`_empty_cuda_cache` but can be overridden in
    tests to assert that exit ran (without needing torch installed).
    """

    def __init__(
        self,
        factory: Callable[[], T],
        *,
        cleanup: Callable[[], None] = _empty_cuda_cache,
        name: str | None = None,
    ) -> None:
        self._factory = factory
        self._cleanup = cleanup
        self._name = name or factory.__name__ if hasattr(factory, "__name__") else "model"
        self._model: T | None = None

    @property
    def model(self) -> T:
        if self._model is None:
            raise RuntimeError(f"ModelSession({self._name}): not entered")
        return self._model

    def __enter__(self) -> T:
        logger.info("loading model: %s", self._name)
        self._model = self._factory()
        return self._model

    def __exit__(self, exc_type, exc, tb) -> None:
        logger.info("unloading model: %s", self._name)
        self._model = None
        self._cleanup()
        # Don't suppress exceptions.
        return None
