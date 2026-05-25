"""Tests for schema init — FakeDriver captures executed cypher."""
from graph.schema import CONSTRAINTS, INDEXES, init_schema


class FakeSession:
    def __init__(self, log: list[tuple[str, dict]]):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def run(self, cypher: str, **params):
        self.log.append((cypher, dict(params)))
        return iter(())


class FakeDriver:
    def __init__(self):
        self.statements: list[tuple[str, dict]] = []

    def session(self, **kwargs):
        return FakeSession(self.statements)

    def close(self) -> None:
        pass


def test_init_schema_runs_all_statements():
    d = FakeDriver()
    n = init_schema(d)
    assert n == len(CONSTRAINTS) + len(INDEXES)
    assert len(d.statements) == n


def test_init_schema_includes_all_labels():
    d = FakeDriver()
    init_schema(d)
    cypher_blob = "\n".join(s for s, _ in d.statements)
    for label in ("Occurrence", "Aircraft", "Finding", "Recommendation", "Regulation", "AC"):
        assert label in cypher_blob, f"missing constraint for {label}"


def test_init_schema_idempotent_cypher():
    """All schema statements use IF NOT EXISTS so re-running is safe."""
    for stmt in CONSTRAINTS + INDEXES:
        assert "IF NOT EXISTS" in stmt
