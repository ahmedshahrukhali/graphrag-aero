# P3 retrieve image — torch + FlagEmbedding (BGE-M3 dense + reranker-v2-m3)
# + qdrant-client. Same dep family as embed/Dockerfile; we pre-cache both
# model weights so cold runs don't need HuggingFace network access.

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

# Re-use embed/'s requirements so we don't drift on pin versions.
COPY embed/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-cache BGE-M3 (query encoder) and the cross-encoder reranker.
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)" \
 && python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)"

# We import from both embed/ (DenseEmbedder, qdrant config) and retrieve/.
COPY embed /app/embed
COPY retrieve /app/retrieve

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "retrieve.run"]
