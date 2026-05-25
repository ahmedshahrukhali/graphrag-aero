# graph/ — P4

Neo4j knowledge graph for the aerospace corpus. P4 ships the schema +
Occurrence-only population; richer entity extraction (Aircraft, Finding,
Recommendation, Regulation, AC) is a follow-up task with the hooks in
`extract.py` ready.

## Schema

```
(:Occurrence)-[:INVOLVES]->(:Aircraft)
(:Occurrence)-[:HAS_FINDING]->(:Finding)
(:Finding)-[:LED_TO]->(:Recommendation)
(:Recommendation)-[:CITES]->(:Regulation)
(:Regulation)-[:GUIDED_BY]->(:AC)
```

Each node label has a uniqueness constraint on `.id`. The `Occurrence` node
also has an index on `.lang` for cross-lingual filtering.

## Install (host)

```bash
pip install -r graph/requirements.txt
pip install -r graph/requirements-dev.txt   # + pytest
```

## Run

```bash
# bring up Neo4j first
docker compose up -d neo4j

# apply schema (idempotent)
python -m agent.run init-schema

# populate Occurrence nodes from data/chunks
python -m agent.run upsert-graph --in data/chunks
```

The CLI lives in `agent/run.py` so all graph + agent operations share one
entry point. Direct module imports are fine for programmatic use:

```python
from graph.client import make_driver
from graph.schema import init_schema
from graph.upsert import upsert_occurrences_from_chunks

driver = make_driver()
init_schema(driver)
upsert_occurrences_from_chunks(driver, Path("data/chunks"))
driver.close()
```

## Environment

Reads from `.env` (or process env):

| variable          | default                  |
|-------------------|--------------------------|
| `NEO4J_URI`       | `bolt://localhost:7687`  |
| `NEO4J_USER`      | `neo4j`                  |
| `NEO4J_PASSWORD`  | `please_change_me`       |

## Tests

```bash
pytest graph/tests -q
```

All offline. A FakeDriver / FakeSession captures executed cypher; the real
`neo4j` driver is lazy-imported and never loaded by the test suite.
