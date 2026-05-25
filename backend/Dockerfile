# P6 backend image. Sits on top of the agent runtime — it imports from
# embed/, retrieve/, graph/, and agent/, so we copy all of them. Single
# uvicorn worker by design: the BGE-M3 / reranker / Ollama VRAM budget is
# sequential, not parallel.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
COPY retrieve/requirements.txt /tmp/retrieve-requirements.txt
COPY embed/requirements.txt /tmp/embed-requirements.txt
COPY graph/requirements.txt /tmp/graph-requirements.txt
COPY agent/requirements.txt /tmp/agent-requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-cache BGE-M3 + reranker so cold-start doesn't hit HuggingFace.
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)" \
 && python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)"

COPY embed    /app/embed
COPY retrieve /app/retrieve
COPY graph    /app/graph
COPY agent    /app/agent
COPY backend  /app/backend

ENV PYTHONPATH=/app

EXPOSE 8080

# --workers 1 is load-bearing: sequential VRAM discipline (embed → rerank →
# Ollama) breaks if two requests run in parallel on the same GPU.
CMD ["uvicorn", "backend.app:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
