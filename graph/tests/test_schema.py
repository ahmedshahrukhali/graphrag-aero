"""Tests for schema init — FakeDriver captures executed cypher."""
from graph.schema import CONSTRAINTS, INDEXES, MIGRATIONS, init_schema


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
    assert n == len(CONSTRAINTS) + len(INDEXES) + len(MIGRATIONS)
    assert len(d.statements) == n


def test_init_schema_includes_all_labels():
    d = FakeDriver()
    init_schema(d)
    cypher_blob = "\n".join(s for s, _ in d.statements)
    for label in ("Document", "Occurrence", "Aircraft", "Finding",
                  "Recommendation", "Regulation", "AC"):
        assert label in cypher_blob, f"missing constraint/migration for {label}"


def test_init_schema_ddl_idempotent():
    """DDL statements (CONSTRAINTS + INDEXES) use IF NOT EXISTS."""
    for stmt in CONSTRAINTS + INDEXES:
        assert "IF NOT EXISTS" in stmt


def test_init_schema_migrations_are_where_guarded():
    """Migrations are WHERE-guarded so re-running is a no-op."""
    for stmt in MIGRATIONS:
        assert "WHERE" in stmt, f"migration not WHERE-guarded: {stmt[:60]}"


def test_document_constraint_present():
    assert any("Document" in s for s in CONSTRAINTS)


def test_finding_source_index_present():
    assert any("finding_source" in s for s in INDEXES)
