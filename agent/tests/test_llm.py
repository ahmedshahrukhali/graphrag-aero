"""Tests for OllamaLLM — fake ``ollama`` module monkeypatched into sys.modules."""
import sys
import types

import pytest

from agent.llm import OllamaLLM


class _FakeClient:
    def __init__(self, host=None):
        self.host = host
        self.calls: list[dict] = []

    def chat(self, model, messages, options):
        self.calls.append({"model": model, "messages": messages, "options": options})
        return {"message": {"role": "assistant", "content": "hello from gemma"}}


def _install_fake_ollama(monkeypatch: pytest.MonkeyPatch) -> dict:
    holder = {"client": None}

    class FakeOllamaModule(types.ModuleType):
        Client = staticmethod(None)  # set below

    def Client(host=None):  # noqa: N802 — match real lib's class name
        holder["client"] = _FakeClient(host=host)
        return holder["client"]

    mod = FakeOllamaModule("ollama")
    mod.Client = Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", mod)
    return holder


def test_chat_calls_ollama_client_with_expected_shape(monkeypatch: pytest.MonkeyPatch):
    holder = _install_fake_ollama(monkeypatch)
    llm = OllamaLLM(host="http://x:11434", model="gemma2:9b")
    out = llm.chat(system="be brief", user="what is X?")
    assert out == "hello from gemma"
    fake: _FakeClient = holder["client"]
    assert fake.host == "http://x:11434"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["model"] == "gemma2:9b"
    assert call["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "what is X?"},
    ]


def test_env_defaults(monkeypatch: pytest.MonkeyPatch):
    _install_fake_ollama(monkeypatch)
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    llm = OllamaLLM()
    llm.chat("s", "u")
    fake = sys.modules["ollama"].Client.__self__ if hasattr(sys.modules["ollama"].Client, "__self__") else None
    # Easier check: ask the llm what it stored.
    assert llm._host == "http://env-host:11434"
    assert llm._model == "env-model"


def test_options_passed_through(monkeypatch: pytest.MonkeyPatch):
    holder = _install_fake_ollama(monkeypatch)
    llm = OllamaLLM(options={"temperature": 0.1})
    llm.chat("s", "u")
    # Caller-supplied options are used verbatim (no defaults merged in).
    assert holder["client"].calls[0]["options"] == {"temperature": 0.1}


def test_default_options_set_num_ctx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)
    holder = _install_fake_ollama(monkeypatch)
    llm = OllamaLLM()  # no options → defaults apply
    llm.chat("s", "u")
    opts = holder["client"].calls[0]["options"]
    assert opts["num_ctx"] == 8192
    assert opts["temperature"] == 0.2


def test_num_ctx_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
    holder = _install_fake_ollama(monkeypatch)
    llm = OllamaLLM()
    llm.chat("s", "u")
    assert holder["client"].calls[0]["options"]["num_ctx"] == 16384


def test_default_model_is_qwen3(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    holder = _install_fake_ollama(monkeypatch)
    llm = OllamaLLM()
    llm.chat("s", "u")
    assert holder["client"].calls[0]["model"] == "qwen3:8b"


def test_client_lazy_imported(monkeypatch: pytest.MonkeyPatch):
    """Importing OllamaLLM must not import ollama; only .chat() triggers the import."""
    monkeypatch.delitem(sys.modules, "ollama", raising=False)
    llm = OllamaLLM()
    assert "ollama" not in sys.modules
    _install_fake_ollama(monkeypatch)
    llm.chat("s", "u")
    assert "ollama" in sys.modules
