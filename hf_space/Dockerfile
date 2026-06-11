# Local-dev image for the `hf-space` compose service (CPU-only Gradio shell).
#
# NOTE: this builds the LOCAL Gradio shell only — it is NOT what the HF Space runs.
# The Space uses the gradio SDK (see hf_space/space_root/README.md) and is deployed
# via scripts/deploy_space.ps1 (HfApi upload). ``docker compose --profile hf-space up``
# builds this image to reproduce the Space UI against the local backend. Never deployed.
#
# Single-process, CPU-only — loads NO ML models. All inference goes to $BACKEND_URL.

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
