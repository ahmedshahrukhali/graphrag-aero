# P2 embed image — torch + FlagEmbedding (BGE-M3) + qdrant-client.
# Separate from the P1 ingestion image because paddle (P1's OCR) and torch
# fight over CUDA versions; combining them is a known dependency-resolution
# nightmare. Agent runtime (P3+) shares this stack and can reuse this base.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# torch wheels need libgomp; FlagEmbedding's tokenizer download is HTTPS only.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY embed/requirements.txt /tmp/requirements.txt
# Install CUDA-enabled torch first against PyTorch's index; subsequent pip
# installs see torch already satisfied and skip the CPU wheel from PyPI.
# Wheels bundle CUDA runtime libs, so python:3.11-slim base is fine.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 "torch>=2.2,<3" \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-cache the BGE-M3 model weights (~1 GB fp16) so the first run on machines
# without network completes successfully. Comment this out if you'd rather
# pay the download cost on first use and keep the image smaller.
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)"

COPY embed /app/embed

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "embed.run"]
CMD ["--in", "/app/data/chunks"]
