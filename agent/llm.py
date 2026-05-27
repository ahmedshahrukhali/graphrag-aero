"""LLM wrapper.

The agent uses gemma2:9b via Ollama (CLAUDE.md locks this). Ollama runs as a
sibling docker service and exposes an HTTP API; ``ollama-python`` is the thin
client.

``LLM`` Protocol lets tests stub generation without importing ollama.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol


logger = logging.getLogger(__name__)


# Ollama's default num_ctx (4096 for gemma2) silently truncates the citation
# block — the synthesize prompt runs ~4.7k tokens, so the model never sees the
# last citations and can't synthesize. 8192 fits the whole prompt.
def _default_options() -> dict:
    return {
        "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
        "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
    }


class LLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...


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
        self._model = model or os.environ.get("OLLAMA_MODEL", "gemma2:9b")
        self._options = options if options is not None else _default_options()
        self._client = None

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
                {"role": "user", "content": user},
            ],
            options=self._options,
        )
        # Ollama's response shape: {"message": {"role": "assistant", "content": ...}, ...}
        return resp["message"]["content"]
