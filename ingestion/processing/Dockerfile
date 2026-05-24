# P1 ingestion image — isolated from agent runtime because PaddleOCR + torch
# conflict with the agent stack. Pre-downloads the BGE-M3 tokenizer artifact
# (no model weights) so runtime has no HuggingFace network call.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# System deps pdfplumber needs (libgl for image rendering when OCR is used).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ─── core stage: text-only ingestion ─────────────────────────────────────────
COPY ingestion/processing/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-cache the BGE-M3 tokenizer artifact (no model weights — just the
# tokenizer config + vocab). Future runtime calls hit the local cache.
RUN python -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('BAAI/bge-m3')"

# Acquisition module is imported by doc_id.py for URL reconstruction.
COPY ingestion/acquisition/requirements.txt /tmp/acq-requirements.txt
RUN pip install --no-cache-dir -r /tmp/acq-requirements.txt

COPY ingestion /app/ingestion

# Run inside /app; the data volume mounts at /app/data via compose.
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "ingestion.processing.run"]
CMD ["--in", "/app/data/corpus", "--out", "/app/data/chunks"]

# ─── optional OCR stage: install paddle on top ───────────────────────────────
FROM base AS ocr
COPY ingestion/processing/requirements-ocr.txt /tmp/requirements-ocr.txt
RUN pip install --no-cache-dir -r /tmp/requirements-ocr.txt
