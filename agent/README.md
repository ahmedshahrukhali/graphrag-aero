# agent/ — P4

LangGraph multi-hop agent: dense retrieve → graph expand → optional refetch → qwen3:4b synthesis → HITL pre-finalize gate.

## Install

```bash
pip install -r agent/requirements-dev.txt
```

## CLI

```bash
docker compose up -d qdrant neo4j postgres ollama
docker compose exec ollama ollama pull qwen3:4b

python -m agent.run init-schema
python -m agent.run upsert-graph --in data/chunks
python -m agent.run query "fuel exhaustion forced landing" --thread demo-1
python -m agent.run resume demo-1                   # accept draft as-is
python -m agent.run resume demo-1 --draft-file edited.txt
```

## Tests

```bash
pytest agent/tests -q
```

Fully offline: MemorySaver + stubbed embedder/reranker/LLM/Neo4j driver.

## Topology

```
START → retrieve → graph_expand → decide_continue
                                       │
                          ┌────────────┴────────────┐
                          │ hop<max AND best<0.5    │ otherwise
                          ↓                         ↓
                       retrieve                 synthesize
                                                    ↓
                                                (INTERRUPT) ← HITL
                                                    ↓
                                                 finalize → END
```

Compiled with `interrupt_before=["finalize"]`. Callers inspect `state["draft"]`, optionally `graph.update_state(config, {"draft": "..."})`, then `graph.invoke(None, config)` to finalize.

## Env

`NEO4J_URI/USER/PASSWORD`, `POSTGRES_DSN`, `OLLAMA_HOST/MODEL`, `QDRANT_*`, `EMBED_MODEL`, `RERANK_MODEL`. All defaults already in `.env.example`.
