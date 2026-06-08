# P8 HuggingFace Space image. The Space SDK is set to "docker" in the
# README's YAML frontmatter; HF will build this Dockerfile and expose
# port 7860. Locally, ``docker compose --profile hf-space up`` runs the
# same image against the local backend.
#
# Single-process, CPU-only — the Space loads NO ML models. All inference
# goes to the FastAPI backend at $BACKEND_URL.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PORT=7860

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY hf_space/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY hf_space /app/hf_space

ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["python", "-m", "hf_space.app"]
