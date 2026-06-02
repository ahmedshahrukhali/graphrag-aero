# P1 ingestion image — isolated from agent runtime because PaddleOCR + torch
# conflict with the agent stack. Pre-downloads the BGE-M3 tokenizer artifact
# (no model weights) so runtime has no HuggingFace network call.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# System deps for image rendering (libgl, libglib) + PaddleOCR's paddle core,
# which dynamically loads libgomp (GNU OpenMP) — absent from python:3.11-slim,
# so `import paddle` fails with "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
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
# GPU PaddleOCR. paddlepaddle-gpu 3.x bundles its CUDA libs in the wheel
# (torch-style), so it runs on this slim base with no CUDA toolkit / cuDNN /
# cublas in the image — install it from the paddle index, CUDA build cu126.
RUN pip install --no-cache-dir paddlepaddle-gpu==3.0.0 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
COPY ingestion/processing/requirements-ocr.txt /tmp/requirements-ocr.txt
RUN pip install --no-cache-dir -r /tmp/requirements-ocr.txt
