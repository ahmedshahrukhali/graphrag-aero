---
title: GraphRAG Aero
emoji: 🛩️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# hf_space/ — P8

Gradio shell over the FastAPI backend. Cross-lingual (EN+FR) Q&A over
Transport Canada Advisory Circulars and TSB aviation investigation
reports, with **PDF page snippets rendered server-side** as the
multimodal output element.

## What it does

1. User submits a question + lang/source filters + max_hops.
2. The Space calls `POST /retrieve` and `POST /query` on the backend.
3. The agent's draft + the per-node trace are rendered in the UI; the
   draft sits in an editable textbox (this is the HITL gate).
4. For each retrieved chunk, the Space downloads the source PDF, renders
   the relevant page via `pdfplumber.Page.to_image(...)`, and overlays
   the chunk's `bbox` as a translucent rectangle. The thumbnails appear
   inline in a Gradio gallery.
5. User clicks **Finalize** → the Space POSTs `/resume/{thread_id}` with
   the edited draft (or empty body if unchanged) → final answer renders.

The Space loads **no ML models**. Every model call goes to the backend.

## Configure

```bash
cp hf_space/.env.example hf_space/.env
# point BACKEND_URL at a reachable backend (default http://localhost:8080)
```

For a real HuggingFace Spaces deployment, set `BACKEND_URL` as a Space
secret pointing at a publicly-reachable backend.

## Run

Local Python:

```bash
pip install -r hf_space/requirements.txt
BACKEND_URL=http://localhost:8080 python -m hf_space.app
# → http://localhost:7860
```

Local Docker:

```bash
docker compose --profile hf-space up --build hf-space
```

HuggingFace Spaces: push this directory to a Space configured with the
`docker` SDK (the YAML frontmatter at the top of this file is what HF
reads). Set `BACKEND_URL` as a Space secret.

## Layout

- [app.py](app.py) — Gradio Blocks app: handlers `on_ask`, `on_finalize`,
  `on_healthz`. State machine mirrors `frontend/app/page.tsx`.
- [api_client.py](api_client.py) — sync httpx client + frozen
  dataclasses mirroring `backend/schemas.py`. Raises `ApiError` on
  non-2xx.
- [pdf_render.py](pdf_render.py) — `render_page_with_bbox(pdf_url, page,
  bbox, dpi=120)` returns a PIL image of the PDF page with the bbox
  highlighted. `bbox_to_pixels` is the pure helper (pdfplumber points →
  rendered pixels). LRU-cached.
- [tests/](tests/) — offline: api_client uses `httpx.MockTransport`,
  pdf_render uses a mocked pdfplumber + in-memory PIL.

## Tests

```bash
pytest hf_space/tests/
```

No network. No model weights. 12 tests.

## Out of scope

- CLIP / Whisper / any extra input modality (text-only input by design).
- Self-hosting the model stack inside the Space — that's what the
  FastAPI backend is for. The Space is intentionally a shell.
- Auth — single-user demo.
