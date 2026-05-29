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
2. The Space streams `GET /query/stream` (SSE): `status → sources → token×N → done`.
   Status lines fill a "🧠 Thinking" accordion; the `sources` event fills a
   "📑 Sources" accordion (chunk snippets); tokens stream into the answer.
3. The streamed draft is the HITL gate — Accept / Edit / Discard appear once
   `done` arrives.
4. **After** the answer finishes, the Space renders each source PDF page and
   appends them as a zoomable `gr.Gallery` message **inline in the chat**
   (see "Design notes" for why inline, not a side panel). Highlights are
   **anchored to the answer's citations**, not the stored chunk bbox.
5. Accept finalizes; Edit opens the draft for correction; Discard drops the turn.

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
  bbox, *, dpi=100, draw_bbox=True, locate_text=None)` returns a PIL image
  of the PDF page. `bbox_to_pixels` is the pure helper (pdfplumber points →
  rendered pixels). `search_page_bbox(page, needle)` locates a cited span via
  `pdfplumber.Page.search` for the citation-anchored highlight. LRU-cached.
- [tests/](tests/) — offline: api_client uses `httpx.MockTransport`,
  pdf_render uses a mocked pdfplumber + in-memory PIL.

## Tests

```bash
pytest hf_space/tests/
```

No network. No model weights.

## Design notes — S15 pivots

Two deliberate pivots landed in S15. Both are reversible; the revert paths
are recorded here so a future session doesn't have to reverse-engineer them.

### Pivot 1 — citation-anchored highlights (replaces stored-bbox draw)

**Why.** The chunk `bbox` stored in Qdrant is mis-placed: `ingestion`'s
`_join_pages` char→bbox alignment desyncs (extract_text injects layout
whitespace absent from the char list), so most chars get `bbox=None` and the
per-chunk union collapses to the first line / top-left corner. Drawing it
produced boxes the user confirmed were useless.

**What we do now.** At render time we parse the answer's citations
(`[doc p.N] "quoted span"`, via `_parse_citations` / `_CITATION_RE` in
`app.py`), then `pdfplumber.Page.search` for that span on the page and box the
**matched span**. **No box if the span isn't found** (no misleading rectangle).
The broken stored bbox is bypassed — no re-ingestion needed.

**Revert.** The legacy stored-bbox path is *still in `pdf_render.py`* — call
`render_page_with_bbox(url, page, bbox, locate_text=None)` and it draws the
stored bbox as before. To go back fully, have `_gallery_items` pass
`draw_bbox=True` without a `locate_text`/`citations` argument. (The real fix
lives upstream in `ingestion/processing/chunk.py::_join_pages` — fixing the
alignment there would make the stored bbox trustworthy again.)

### Pivot 2 — inline page gallery (replaces the right "Sources" sidebar)

**Why.** The right `gr.Sidebar` (Pages / Chunks tabs) was too narrow: the
lightbox blew out the layout, chunk text was gated behind a "click a message"
placeholder that rendered nothing useful, and it ate horizontal space the chat
needed. Clicking empty chat space fired the (now-removed) `on_select` and left
dead buffer.

**What we do now.** No right panel. After the answer streams, source pages are
appended as one zoomable `gr.Gallery` **message in the chat** (`_gallery_message`
in `app.py`, `allow_preview=True` for the lightbox, `.pdf-inline` CSS caps it at
~46vh so it sits beside the answer). Rendering happens *after* the `done` yield
so it never delays tokens or the HITL row. Chunk snippets moved into the
"📑 Sources" accordion (`_chunks_md`). The bbox toggle re-renders by stripping
and re-appending the trailing gallery message (`on_toggle_bbox`).

**Revert.** `git revert` the S15 inline-gallery commit to restore the right
`gr.Sidebar(position="right")` with Pages/Chunks tabs + the lazy `on_select`
render path. (The two pivots are in one commit; reverting brings back both the
old sidebar *and* the stored-bbox draw — split them by hand if you only want
one back.)

## Out of scope

- CLIP / Whisper / any extra input modality (text-only input by design).
- Self-hosting the model stack inside the Space — that's what the
  FastAPI backend is for. The Space is intentionally a shell.
- Auth — single-user demo.
