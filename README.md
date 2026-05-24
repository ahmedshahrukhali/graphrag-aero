# Graph RAG over Aerospace Documents

End-to-end Graph RAG over Transport Canada + Transportation Safety Board (TSB) aviation documents (EN + FR): PDF ingestion → dense retrieval + rerank → multi-agent graph reasoning (LangGraph + Neo4j) → evaluation, OpenTelemetry tracing, a Next.js UI, and a Gradio Hugging Face Space demo. Fully local LLM inference via Ollama.

This repo is built **with Claude Code running on your machine.** See `CLAUDE.md` (the agent's instructions) and `MANIFEST.md` (decisions + progress).

## Prerequisites
- Docker + Docker Compose
- Node.js 18+ (for Claude Code) and `npm i -g @anthropic-ai/claude-code`
- An NVIDIA GPU is recommended but not required (CPU works; uncomment the GPU block in `docker-compose.yml` to use a GPU)

## Quick start
```bash
# 1. configure
cp .env.example .env        # then edit secrets + pick OLLAMA_MODEL

# 2. start backing services
make infra                  # qdrant, neo4j, postgres, ollama, otel-collector

# 3. pull the local LLM
make models

# 4. add documents
#    drop Transport Canada / TSB PDFs into data/corpus/
#    or run the P1b acquisition script (see ingestion/acquisition/README.md)

# 5. drive the build with Claude Code
claude                      # in this repo; it reads CLAUDE.md and works phase by phase
```

Once the app phases (P6/P7) are built:
```bash
make up                     # backend :8080, frontend :3000
make ingest                 # process data/corpus
make eval                   # Recall@k / nDCG / MRR
```

## Layout
| Path | Phase | What lives here |
|------|-------|-----------------|
| `ingestion/` | P1 | PDF load, OCR, tables, chunk, dedup |
| `backend/` | P2,P3,P6 | embeddings, retrieval/rerank, FastAPI, agents |
| `eval/` | P5 | retrieval metrics |
| `frontend/` | P7 | Next.js + TS UI |
| `hf_space/` | P8 | Gradio multimodal demo |
| `otel/` | — | OpenTelemetry collector config |
| `data/corpus/` | — | your source PDFs (gitignored) |

## Notes
- Open questions to settle as you reach the relevant phase are tracked in `MANIFEST.md` (LLM choice, languages, HITL target).
- Tests are designed to run **without** downloading model weights (models are mocked), so CI stays fast and offline.
