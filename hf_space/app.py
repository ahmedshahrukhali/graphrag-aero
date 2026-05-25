"""Gradio Blocks app — the HuggingFace Space surface.

The Space is a thin shell: every model call goes to the FastAPI backend
at ``$BACKEND_URL``. The only locally-computed pieces are the PDF page
images with bbox highlights (pdf_render.render_page_with_bbox), which is
the "multimodal" element of this demo.

State flow mirrors frontend/app/page.tsx:
    idle → asking → paused → resuming → done
``gr.State`` carries the per-session blob (thread_id, retrieve hits,
draft from the agent) so handlers stay pure functions.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import gradio as gr

from hf_space.api_client import ApiClient, ApiError, RetrieveResponse, make_client
from hf_space.pdf_render import PdfRenderError, render_page_with_bbox


logger = logging.getLogger(__name__)


# ─── handlers ────────────────────────────────────────────────────────────────

def _lang_param(choice: str) -> str | None:
    return None if choice == "all" else choice


def _gallery_items(retrieve: RetrieveResponse) -> list[tuple[Any, str]]:
    """For each retrieved chunk, render its PDF page with the bbox
    highlighted. Failed renders fall through to a text-only caption (no
    image) so a broken PDF URL doesn't blank the whole gallery."""
    items: list[tuple[Any, str]] = []
    for c in retrieve.results:
        caption = (
            f"#{c.rank} · {c.doc_id} · p.{c.page} · "
            f"rerank={'—' if c.rerank_score is None else f'{c.rerank_score:.3f}'}"
        )
        if not c.source_url:
            continue
        try:
            img = render_page_with_bbox(c.source_url, c.page, c.bbox)
            items.append((img, caption))
        except PdfRenderError as e:
            logger.warning("pdf render failed for %s: %s", c.doc_id, e)
    return items


def _trace_rows(trace: list[dict]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for step in trace:
        node = step.get("node", "")
        elapsed = step.get("elapsed_ms", 0)
        extras = "  ".join(
            f"{k}={v!r}" for k, v in step.items() if k not in ("node", "elapsed_ms")
        )
        rows.append([node, elapsed, extras])
    return rows


def _chunks_md(retrieve: RetrieveResponse) -> str:
    if not retrieve.results:
        return "_no chunks retrieved._"
    out: list[str] = []
    for c in retrieve.results:
        snippet = " ".join(c.text.split())[:280]
        section = f" · § {c.section_title}" if c.section_title else ""
        out.append(
            f"**#{c.rank} · `{c.doc_id}` · p.{c.page}{section}** "
            f"(rerank={c.rerank_score if c.rerank_score is None else f'{c.rerank_score:.3f}'})\n\n"
            f"{snippet}…\n"
        )
    return "\n---\n".join(out)


def make_app(api: ApiClient | None = None) -> gr.Blocks:
    """Build the Gradio app. Tests pass a stubbed ``api`` to drive
    handlers without a real backend."""
    client = api or make_client()

    def on_ask(query: str, lang: str, source: str, max_hops: int, state: dict):
        if not query.strip():
            return state, gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ enter a query."

        thread_id = str(uuid.uuid4())
        try:
            retrieve = client.retrieve(
                query.strip(),
                lang=_lang_param(lang),
                source=_lang_param(source),
                top_k=10,
            )
            paused = client.query(query.strip(), thread_id, max_hops=max_hops)
        except ApiError as e:
            return state, gr.update(), gr.update(), gr.update(), gr.update(), _fmt_error(e)

        new_state = {
            "thread_id": paused.thread_id,
            "draft": paused.draft or "",
            "retrieve": retrieve,
            "trace": paused.trace,
        }
        return (
            new_state,
            gr.update(value=paused.draft or "", visible=True),
            gr.update(value=_gallery_items(retrieve), visible=True),
            gr.update(value=_chunks_md(retrieve), visible=True),
            gr.update(value=_trace_rows(paused.trace), visible=True),
            f"paused at HITL · thread_id `{paused.thread_id}` · {paused.n_candidates} candidates",
        )

    def on_finalize(edited_draft: str, state: dict):
        if not state or "thread_id" not in state:
            return state, gr.update(), gr.update(), "⚠️ no active session."

        original = state.get("draft") or ""
        body_draft = edited_draft if edited_draft != original else None
        try:
            done = client.resume(state["thread_id"], draft=body_draft)
        except ApiError as e:
            return state, gr.update(), gr.update(), _fmt_error(e)

        return (
            state,
            gr.update(value=done.final or "_(no answer)_", visible=True),
            gr.update(value=_trace_rows(done.trace), visible=True),
            f"finalised · thread_id `{done.thread_id}`"
            + (" · draft was edited by user" if body_draft is not None else ""),
        )

    def on_healthz(_state: dict):
        try:
            h = client.healthz()
        except ApiError as e:
            return f"❌ backend unreachable ({e.status})"
        except Exception as e:  # noqa: BLE001
            return f"❌ backend unreachable ({e})"
        flag = "✅" if h.ok else "⚠️"
        return (
            f"{flag} backend: qdrant={h.qdrant.ok} · neo4j={h.neo4j.ok} · ollama={h.ollama.ok}"
        )

    with gr.Blocks(title="GraphRAG Aero") as app:
        gr.Markdown(
            "# GraphRAG Aero — TC + TSB aviation safety\n"
            "Ask a question in **English or French**. The agent retrieves passages from "
            "Transport Canada Advisory Circulars and TSB investigation reports, drafts an "
            "answer, and pauses at the HITL gate so you can edit before finalising."
        )

        state = gr.State({})
        with gr.Row():
            status = gr.Markdown("_ready._")
        with gr.Row():
            health_btn = gr.Button("Check backend", scale=0)
            health_md = gr.Markdown("")
            health_btn.click(on_healthz, inputs=[state], outputs=[health_md])

        with gr.Row():
            query = gr.Textbox(
                label="Question",
                placeholder="e.g. fuel exhaustion forced landing",
                lines=2,
                scale=4,
            )
        with gr.Row():
            lang = gr.Radio(["all", "en", "fr"], value="all", label="Language")
            source = gr.Radio(["all", "tsb", "tc"], value="all", label="Source")
            max_hops = gr.Slider(1, 5, value=2, step=1, label="Max hops")
            ask_btn = gr.Button("Ask agent", variant="primary")

        with gr.Row():
            with gr.Column(scale=3):
                draft = gr.Textbox(
                    label="Draft (HITL gate — edit before finalising)",
                    lines=10,
                    visible=False,
                    interactive=True,
                )
                finalize_btn = gr.Button("Finalize", visible=True)
                final_md = gr.Markdown(visible=False)
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Cited PDF snippets",
                    columns=1,
                    rows=2,
                    height="auto",
                    visible=False,
                )
        chunks_md = gr.Markdown(visible=False)

        gr.Markdown("### Agent trace")
        trace = gr.Dataframe(
            headers=["node", "elapsed_ms", "extras"],
            datatype=["str", "number", "str"],
            interactive=False,
            visible=False,
        )

        ask_btn.click(
            on_ask,
            inputs=[query, lang, source, max_hops, state],
            outputs=[state, draft, gallery, chunks_md, trace, status],
        )
        finalize_btn.click(
            on_finalize,
            inputs=[draft, state],
            outputs=[state, final_md, trace, status],
        )

    return app


def _fmt_error(e: ApiError) -> str:
    detail = ""
    if isinstance(e.detail, dict) and "detail" in e.detail:
        detail = f" — {e.detail['detail']}"
    return f"❌ backend error ({e.status}){detail}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = make_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
