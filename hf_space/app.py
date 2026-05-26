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


def _patch_gradio_client_bool_schema() -> None:
    """Work around a gradio_client 4.x bug: ``_json_schema_to_python_type``
    assumes every schema is a dict, but a JSON schema can legally be a bool
    (e.g. ``additionalProperties: true``). When one reaches ``get_type`` it
    does ``if "const" in schema`` on a bool → ``TypeError``, which crashes
    ``get_api_info()`` at launch and stops the server from binding. Short-
    circuit bool schemas the way upstream later did. Idempotent + defensive.
    """
    try:
        import gradio_client.utils as gcu
    except Exception:  # noqa: BLE001
        return

    orig = gcu._json_schema_to_python_type

    def safe(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return orig(schema, defs)

    gcu._json_schema_to_python_type = safe


_patch_gradio_client_bool_schema()


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

    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo").set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_shadow="*shadow_drop_lg",
    )

    with gr.Blocks(
        title="GraphRAG Aero",
        theme=theme,
        css=".gradio-container{max-width:1200px!important;margin:auto}"
            " .hero h1{margin-bottom:.25em} .hero p{color:#475569;margin-top:0}"
            " .pill{display:inline-block;padding:2px 8px;border-radius:999px;"
            "background:#eff6ff;color:#1e40af;font-size:12px;margin-right:6px}",
    ) as app:
        with gr.Row(elem_classes="hero"):
            gr.Markdown(
                "# 🛩️ GraphRAG Aero\n"
                "**Multilingual Graph RAG over Transport Canada Advisory Circulars + "
                "TSB aviation investigation reports.** Ask in English or French; the "
                "agent retrieves cited passages, drafts an answer, and pauses at a "
                "Human-in-the-Loop gate so you can edit before finalising.\n\n"
                "<span class='pill'>BGE-M3 + reranker-v2-m3</span>"
                "<span class='pill'>Qdrant 54k chunks</span>"
                "<span class='pill'>Neo4j knowledge graph</span>"
                "<span class='pill'>gemma2:9b · Ollama</span>"
                "<span class='pill'>EN + FR</span>"
            )

        state = gr.State({})

        with gr.Row():
            health_md = gr.Markdown("_checking backend…_")
            health_btn = gr.Button("↻ Refresh", scale=0, size="sm")
            health_btn.click(on_healthz, inputs=[state], outputs=[health_md])

        with gr.Row():
            query = gr.Textbox(
                label="Your question",
                placeholder="e.g. fuel exhaustion forced landing",
                lines=2,
                scale=4,
                autofocus=True,
            )
        with gr.Row():
            lang = gr.Radio(["all", "en", "fr"], value="all", label="Language")
            source = gr.Radio(["all", "tsb", "tc"], value="all", label="Source")
            max_hops = gr.Slider(1, 5, value=2, step=1, label="Max hops")
            ask_btn = gr.Button("🔍 Ask agent", variant="primary", size="lg")

        gr.Examples(
            examples=[
                ["fuel exhaustion forced landing", "en", "tsb", 2],
                ["engine failure after takeoff", "en", "tsb", 2],
                ["carburetor icing", "en", "all", 2],
                ["VFR flight into IMC", "en", "tsb", 2],
                ["alimentation en carburant", "fr", "tsb", 2],
                ["atterrissage forcé moteur en panne", "fr", "tsb", 2],
            ],
            inputs=[query, lang, source, max_hops],
            label="Try a sample query",
        )

        status = gr.Markdown("_ready._")

        with gr.Row():
            with gr.Column(scale=3):
                draft = gr.Textbox(
                    label="✏️ Draft answer — edit before finalising (HITL gate)",
                    lines=10,
                    visible=False,
                    interactive=True,
                )
                finalize_btn = gr.Button("✅ Finalize", variant="primary", visible=True)
                final_md = gr.Markdown(visible=False, label="Final answer")
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="📄 Cited PDF snippets (bbox highlighted)",
                    columns=1,
                    rows=2,
                    height="auto",
                    visible=False,
                    object_fit="contain",
                )

        with gr.Accordion("📚 Cited passages (text)", open=False):
            chunks_md = gr.Markdown("_run a query to see cited passages._")

        with gr.Accordion("🔬 Agent trace", open=False):
            trace = gr.Dataframe(
                headers=["node", "elapsed_ms", "extras"],
                datatype=["str", "number", "str"],
                interactive=False,
                wrap=True,
            )

        with gr.Accordion("ℹ️ About", open=False):
            gr.Markdown(
                "- **Corpus:** ~54k chunks across 1,860 docs from TSB aviation "
                "investigation reports (1991–present, EN + FR) and Transport Canada "
                "Advisory Circulars.\n"
                "- **Pipeline:** Qdrant ANN over BGE-M3 dense embeddings → "
                "bge-reranker-v2-m3 cross-encoder → multi-hop LangGraph agent → "
                "HITL gate → gemma2:9b synthesis.\n"
                "- **Multimodal:** for every cited chunk the Space fetches the source "
                "PDF and renders the page with the chunk's bounding box highlighted.\n"
                "- **Source code:** https://github.com/ahmedshahrukhali/graphrag-aero\n"
                "- This Space is a thin client; all inference happens on a backend "
                "exposed via `BACKEND_URL`."
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

        app.load(on_healthz, inputs=[state], outputs=[health_md])

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
