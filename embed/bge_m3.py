"""BGE-M3 dense + sparse embedder wrapper.

CLAUDE.md locks: BGE-M3 (BAAI/bge-m3) via FlagEmbedding, dense dim=1024,
multilingual EN+FR+ZH. Sparse output (lexical weights) can be requested
alongside dense for hybrid Qdrant collections.

Real model loading is lazy and import-guarded so the test suite stays offline.
``DenseEmbedder`` Protocol is unchanged — tests pass a stub with the same shape.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol, Sequence


logger = logging.getLogger(__name__)


DENSE_DIM = 1024
DEFAULT_MAX_LENGTH = 8192

# Sparse weight type: {token_id: importance_score}
SparseWeights = dict[int, float]


class DenseEmbedder(Protocol):
    """Anything with ``embed(texts) -> list of 1024-vectors`` satisfies this."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class BGE_M3Embedder:
    """Wraps ``FlagEmbedding.BGEM3FlagModel`` for dense + optional sparse encoding."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_fp16: bool = True,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        # Lazy import — keeps ``import embed.bge_m3`` cheap and offline-safe.
        from FlagEmbedding import BGEM3FlagModel  # type: ignore

        name = model_name or os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
        # device: None → FlagEmbedding auto-selects (GPU if present); "cpu" pins
        # to CPU. The backend pins to CPU so BGE-M3 doesn't co-reside on the 8 GB
        # GPU with Ollama (their sum crashes the WSL GPU VM); the batch embed job
        # leaves this None to keep GPU speed. Env EMBED_DEVICE overrides.
        if device is None:
            device = os.environ.get("EMBED_DEVICE") or None
        if device == "cpu":
            use_fp16 = False  # fp16 is a GPU optimization — invalid/slow on CPU.
        logger.info("loading BGE-M3 (%s, fp16=%s, device=%s)", name, use_fp16, device or "auto")
        self._model = BGEM3FlagModel(name, use_fp16=use_fp16, devices=device)
        self._max_length = max_length
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # ``out["dense_vecs"]`` is a numpy array of shape (N, 1024).
        vecs = out["dense_vecs"]
        return [list(map(float, row)) for row in vecs]

    def embed_sparse(self, texts: Sequence[str]) -> list[SparseWeights]:
        """Return BGE-M3 lexical weights (sparse) for each text.

        ``out["lexical_weights"]`` from FlagEmbedding is a list of dicts
        ``{token_id: score}``. Scores are float32 importance weights; zero
        entries are omitted, so the dict is already sparse.
        """
        if not texts:
            return []
        out = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        raw: list[dict] = out["lexical_weights"]
        return [{int(k): float(v) for k, v in w.items()} for w in raw]


class HuggingFaceEmbedder:
    """Calls Hugging Face Inference API for dense embeddings.
    Used as an automatic fallback when local PyTorch is unavailable.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model = model_name or os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient  # type: ignore
            token = os.environ.get("HF_TOKEN")
            if not token:
                logger.warning("HF_TOKEN not set; HuggingFaceEmbedder will use unauthenticated limits")
            logger.info("connecting to HF Inference API: %s", self._model)
            self._client = InferenceClient(model=self._model, token=token)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        out = client.feature_extraction(list(texts))
        import numpy as np
        arr = np.array(out)
        if arr.ndim == 3:
            # Pooling: use CLS token (first token) if unpooled
            arr = arr[:, 0, :]
        return [list(map(float, row)) for row in arr]

    def embed_sparse(self, texts: Sequence[str]) -> list[SparseWeights]:
        # The Inference API does not natively expose the lexical weights for BGE-M3.
        # Return empty sparse weights to fall back to dense-only retrieval.
        return [{} for _ in texts]


def get_embedder(device: str | None = None, **kwargs) -> DenseEmbedder:
    """Auto-fallback factory: local PyTorch if available or forced, else HF API."""
    try:
        import torch
        if device == "cpu" or torch.cuda.is_available():
            return BGE_M3Embedder(device=device, **kwargs)
    except ImportError:
        pass
    logger.info("Local GPU unavailable or PyTorch missing, falling back to HuggingFaceEmbedder")
    return HuggingFaceEmbedder()
