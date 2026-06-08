"""Cross-encoder reranker — BAAI/bge-reranker-v2-m3 via FlagEmbedding.

CLAUDE.md locks the reranker model. The cross-encoder scores (query, passage)
pairs jointly; far more accurate than dense cosine similarity but quadratic in
pair count — that's why P3 only reranks the ANN top-K (default 50), not the
full corpus.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from embed.jsonl import ChunkRecord


logger = logging.getLogger(__name__)


DEFAULT_MAX_LENGTH = 512


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieval candidate with its scores.

    ``ann_score`` is Qdrant cosine similarity from the dense ANN step.
    ``rerank_score`` is the cross-encoder logit; ``None`` until reranked.
    Final ordering uses ``rerank_score`` when present, else ``ann_score``.
    """
    record: ChunkRecord
    ann_score: float
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.ann_score


class CrossEncoderReranker(Protocol):
    """Anything with ``score(query, passages) -> list[float]`` is reranker-shaped."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class BGE_RerankerV2M3:
    """Wraps ``FlagEmbedding.FlagReranker`` for ``bge-reranker-v2-m3``."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        use_fp16: bool = True,
        max_length: int = DEFAULT_MAX_LENGTH,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        from FlagEmbedding import FlagReranker  # type: ignore

        name = model_name or os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        # device: None → auto (GPU if present); "cpu" pins to CPU. The backend
        # pins to CPU so the reranker doesn't share the 8 GB GPU with Ollama
        # (the sum crashes the WSL GPU VM). Env RERANK_DEVICE overrides.
        if device is None:
            device = os.environ.get("RERANK_DEVICE") or None
        
        # Disable fp16 globally to prevent the 'expected scalar type Float but found Half' PyTorch bug
        # that occurs when model.half() is called inside FlagReranker on specific Transformers versions.
        use_fp16 = False
        
        logger.info("loading reranker (%s, fp16=%s, device=%s)", name, use_fp16, device or "auto")
        self._model = FlagReranker(name, use_fp16=use_fp16, devices=device)
        self._max_length = max_length
        self._normalize = normalize

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        out = self._model.compute_score(
            pairs, max_length=self._max_length, normalize=self._normalize,
        )
        # FlagReranker returns a float for a single pair, list[float] otherwise.
        if isinstance(out, (int, float)):
            return [float(out)]
        return [float(x) for x in out]


def rerank(
    query: str,
    candidates: Sequence[ScoredChunk],
    reranker: CrossEncoderReranker,
    *,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Score ``candidates`` with the cross-encoder; return them ordered by
    ``rerank_score`` desc. Original ``ann_score`` is preserved on each result.

    ``top_k`` caps the returned list (post-rerank); ``None`` returns all.
    """
    if not candidates:
        return []
    scores = reranker.score(query, [c.record.text for c in candidates])
    if len(scores) != len(candidates):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(candidates)} candidates")
    rescored = [replace(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
    rescored.sort(key=lambda c: c.rerank_score, reverse=True)
    if top_k is not None:
        rescored = rescored[:top_k]
    return rescored


class HuggingFaceReranker:
    """Calls Hugging Face Inference API for cross-encoder reranking.
    Used as an automatic fallback when local PyTorch is unavailable.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model = model_name or os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient  # type: ignore
            token = os.environ.get("HF_TOKEN")
            if not token:
                logger.warning("HF_TOKEN not set; HuggingFaceReranker will use unauthenticated limits")
            logger.info("connecting to HF Inference API: %s", self._model)
            self._client = InferenceClient(model=self._model, token=token)
        return self._client

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        
        # Inference API for text classification expects `inputs: "text"` or pairs depending on model.
        # But bge-reranker requires pairs. We will send raw HTTP post as hf_hub InferenceClient
        # doesn't elegantly support pair classification payloads natively for all rerankers.
        import requests
        url = f"https://api-inference.huggingface.co/models/{self._model}"
        headers = {}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        payload = {"inputs": {"source_sentence": query, "sentences": list(passages)}}
        resp = requests.post(url, headers=headers, json=payload)
        
        if resp.status_code != 200:
            logger.error("HF reranker failed: %s", resp.text)
            return [0.0] * len(passages)
            
        data = resp.json()
        
        # Depending on the endpoint, it could be a list of floats or list of dicts.
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], float):
            return data
            
        # Fallback naive parse if the format is [{"label": "...", "score": 0.99}, ...]
        try:
            return [float(x.get("score", 0.0)) if isinstance(x, dict) else float(x) for x in data]
        except Exception:
            logger.error("Failed to parse HF reranker response: %s", data)
            return [0.0] * len(passages)


def get_reranker(device: str | None = None, **kwargs) -> CrossEncoderReranker:
    """Auto-fallback factory: local PyTorch if available or forced, else HF API."""
    try:
        import torch
        if device == "cpu" or torch.cuda.is_available():
            return BGE_RerankerV2M3(device=device, **kwargs)
    except ImportError:
        pass
    logger.info("Local GPU unavailable or PyTorch missing, falling back to HuggingFaceReranker")
    return HuggingFaceReranker()
