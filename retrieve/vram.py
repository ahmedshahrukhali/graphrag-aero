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
    """Best-effort GPU memory release. No-op if torch isn't installed or no GPU.

    ``empty_cache()`` only returns blocks the allocator already considers free,
    so we ``gc.collect()`` first to drop the just-unloaded model's tensors —
    otherwise the ~4 GB of BGE-M3 + reranker stays resident and, on the 8 GB
    3060Ti, collides with Ollama's ~6.9 GB at the generate step and crashes the
    WSL GPU VM. This is the load-bearing half of the sequential-VRAM discipline.
    """
    import gc

    gc.collect()
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wait_for_free_vram(
    min_free_mib: int = 7000,
    *,
    timeout_s: float = 30.0,
    poll_s: float = 0.5,
) -> bool:
    """Block until the GPU reports >= ``min_free_mib`` free, or ``timeout_s``.

    This is the cross-process **barrier** that makes VRAM occupancy sequential.
    The agent's compute is already sequential (retrieve → its text feeds the
    LLM), but the *memory* isn't: after unloading BGE-M3 + reranker, the backend
    process's GPU memory isn't reclaimed by the driver instantly. Calling Ollama
    immediately let its ~6.9 GB allocation overlap the not-yet-freed ~4 GB →
    momentary >8 GB → WSL GPU VM crash. So after unloading we wait *here* until
    the driver confirms the GPU is actually free, then generate.

    Returns True once free (or if torch/CUDA is unavailable — nothing to wait
    for). Returns False on timeout (caller proceeds anyway, best-effort).
    """
    import time

    try:
        import torch  # type: ignore
    except ImportError:
        return True
    if not torch.cuda.is_available():
        return True

    need = min_free_mib * 1024 * 1024
    deadline = time.monotonic() + timeout_s
    while True:
        free, _total = torch.cuda.mem_get_info()
        if free >= need:
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "wait_for_free_vram: only %d MiB free after %.0fs (need %d MiB) — proceeding anyway",
                free // (1024 * 1024), timeout_s, min_free_mib,
            )
            return False
        time.sleep(poll_s)


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
