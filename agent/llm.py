"""LLM wrapper.

The agent uses qwen3:4b via Ollama (set by OLLAMA_MODEL; CLAUDE.md locks this
as the single generation LLM). Ollama runs as a sibling docker service and
exposes an HTTP API; ``ollama-python`` is the thin client.

``LLM`` Protocol lets tests stub generation without importing ollama.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Iterator, Protocol


logger = logging.getLogger(__name__)

# Strip a leading reasoning block some thinking models still emit (e.g. an empty
# <think></think>) even with /no_think, so it never leaks into the cited answer.
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


# Ollama's default num_ctx (4096) silently truncates the citation
# block — the synthesize prompt runs ~4.7k tokens, so the model never sees the
# last citations and can't synthesize. 8192 fits the whole prompt.
def _default_options() -> dict:
    return {
        "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
        "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
    }


class LLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...

    def chat_stream(self, system: str, user: str) -> Iterator[str]: ...


class OllamaLLM:
    """Calls ``ollama.Client(host).chat(model=..., messages=[...])``.

    Lazy import — tests pass a stub LLM and never load ``ollama``.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        *,
        options: dict | None = None,
    ) -> None:
        # Lazy import. Resolved at first .chat() call so tests can monkeypatch
        # the ``ollama`` module before any real network call.
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._model = model or os.environ.get("OLLAMA_MODEL", "qwen3:4b")
        self._options = options if options is not None else _default_options()
        # keep_alive=0 → Ollama unloads the model from VRAM immediately after
        # generating, instead of holding it ~5 min. Load-bearing on the 8 GB
        # 3060Ti: it frees the GPU before the next query's BGE-M3 + reranker
        # load, so the LLM and retrieval models are never co-resident (their sum
        # crashes the WSL GPU VM). Override with OLLAMA_KEEP_ALIVE (e.g. "5m").
        self._keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "0")
        # Qwen3 models "think" (emit a reasoning block) by default, which is slow
        # and, for our citation-heavy synthesis, hurts format adherence. Default
        # off via the "/no_think" soft switch (Qwen3 honours it; harmless to
        # other models). Set OLLAMA_THINK=1 to re-enable reasoning.
        self._think = os.environ.get("OLLAMA_THINK", "0") == "1"
        self._client = None

    def _user_content(self, user: str) -> str:
        return user if self._think else f"{user}\n\n/no_think"

    def _ensure_client(self):
        if self._client is None:
            from ollama import Client  # type: ignore

            logger.info("connecting to ollama: %s (model=%s)", self._host, self._model)
            self._client = Client(host=self._host)
        return self._client

    def chat(self, system: str, user: str) -> str:
        client = self._ensure_client()
        resp = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": self._user_content(user)},
            ],
            options=self._options,
            keep_alive=self._keep_alive,
        )
        # Ollama's response shape: {"message": {"role": "assistant", "content": ...}, ...}
        return _THINK_RE.sub("", resp["message"]["content"] or "")

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        """Yield assistant content chunks as Ollama emits them.

        Uses the same chat call with stream=True; each yielded item is a
        partial content string suitable for piping into Streamlit's
        ``st.write_stream`` or any incremental writer.
        """
        client = self._ensure_client()
        for chunk in client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": self._user_content(user)},
            ],
            options=self._options,
            keep_alive=self._keep_alive,
            stream=True,
        ):
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                yield piece


class HuggingFaceLLM:
    """Calls Hugging Face Inference API for text generation.
    Used as an automatic fallback when Ollama is unavailable.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self._client = None
        self._think = os.environ.get("OLLAMA_THINK", "0") == "1"

    def _user_content(self, user: str) -> str:
        return user if self._think else f"{user}\n\n/no_think"

    def _ensure_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient  # type: ignore
            token = os.environ.get("HF_TOKEN")
            if not token:
                logger.warning("HF_TOKEN not set; HuggingFaceLLM will use unauthenticated limits")
            logger.info("connecting to HF Inference API: %s", self._model)
            self._client = InferenceClient(model=self._model, token=token)
        return self._client

    def chat(self, system: str, user: str) -> str:
        client = self._ensure_client()
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": self._user_content(user)},
            ],
            max_tokens=2048,
        )
        return _THINK_RE.sub("", resp.choices[0].message.content or "")

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        client = self._ensure_client()
        for chunk in client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": self._user_content(user)},
            ],
            max_tokens=2048,
            stream=True,
        ):
            piece = chunk.choices[0].delta.content or ""
            if piece:
                yield piece


def get_llm() -> LLM:
    """Auto-fallback factory: ping local Ollama, fallback to Hugging Face API if dead."""
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=1.0)
        return OllamaLLM()
    except Exception:
        logger.info("Ollama unreachable at %s, falling back to HuggingFaceLLM", host)
        return HuggingFaceLLM()
