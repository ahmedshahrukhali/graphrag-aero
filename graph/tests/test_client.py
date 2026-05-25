"""Tests for Neo4jConfig + the make_driver lazy-import surface."""
import pytest

from graph.client import Neo4jConfig


def test_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://x:7687")
    monkeypatch.setenv("NEO4J_USER", "u")
    monkeypatch.setenv("NEO4J_PASSWORD", "p")
    cfg = Neo4jConfig.from_env()
    assert cfg == Neo4jConfig(uri="bolt://x:7687", user="u", password="p")


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch):
    for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    cfg = Neo4jConfig.from_env()
    assert cfg.uri.startswith("bolt://")
    assert cfg.user == "neo4j"
