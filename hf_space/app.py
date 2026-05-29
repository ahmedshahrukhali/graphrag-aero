"""Gradio Blocks app — the HuggingFace Space surface (Gradio 5.x).

Three-zone layout:
    LEFT  Sidebar — brand + New chat + lang/source filters + recent + samples + health
    CENTER Column  — gr.Chatbot(type="messages") + HITL row + composer
    RIGHT Sidebar  — Document gallery + Chunks tabs (per-message artifacts)

The Space remains a thin shell over the FastAPI backend at $BACKEND_URL.
The only locally-computed work is PDF-page-with-bbox rendering.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

import gradio as gr

from hf_space.api_client import ApiClient, ApiError, RetrievedChunk, RetrieveResponse, make_client
from hf_space.pdf_render import PdfRenderError, render_page_with_bbox


logger = logging.getLogger(__name__)


def _patch_gradio_client_bool_schema() -> None:
    """Harmless no-op on Gradio 5; kept for safety during transition."""
    try:
        import gradio_client.utils as gcu
    except Exception:  # noqa: BLE001
        return
    orig = getattr(gcu, "_json_schema_to_python_type", None)
    if orig is None:
        return

    def safe(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return orig(schema, defs)

    gcu._json_schema_to_python_type = safe


_patch_gradio_client_bool_schema()


# ─── pure helpers ────────────────────────────────────────────────────────────

def _lang_param(choice: str) -> str | None:
    return None if choice in ("all",) else choice


def _sources_to_retrieve(sources: list[dict], query: str) -> RetrieveResponse:
    """Adapt the ``sources`` list from /query/stream into a RetrieveResponse."""
    chunks: list[RetrievedChunk] = []
    for i, s in enumerate(sources, start=1):
        bbox = s.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        chunks.append(RetrievedChunk(
            rank=i,
            doc_id=s.get("doc_id", ""),
            source_url=s.get("source_url"),
            section_title=s.get("section_title", ""),
            page=int(s.get("page", 0)),
            bbox=tuple(bbox),  # type: ignore[arg-type]
            lang=s.get("lang", ""),
            text=s.get("text", ""),
            ann_score=float(s.get("ann_score", 0.0)),
            rerank_score=None if s.get("rerank_score") is None else float(s["rerank_score"]),
        ))
    return RetrieveResponse(query=query, results=chunks)


def _gallery_items(retrieve: RetrieveResponse) -> list[tuple[Any, str]]:
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


def _chunks_md(retrieve: RetrieveResponse) -> str:
    if not retrieve.results:
        return "_no chunks retrieved._"
    out: list[str] = []
    for c in retrieve.results:
        snippet = " ".join(c.text.split())[:280]
        section = f" · § {c.section_title}" if c.section_title else ""
        score = "—" if c.rerank_score is None else f"{c.rerank_score:.3f}"
        out.append(
            f"**#{c.rank} · `{c.doc_id}` · p.{c.page}{section}** (rerank={score})\n\n"
            f"{snippet}…\n"
        )
    return "\n---\n".join(out)


def _sources_content(retrieve: RetrieveResponse) -> str:
    """Compact markdown for the sources accordion inside gr.Chatbot."""
    if not retrieve.results:
        return "_no sources_"
    lines: list[str] = []
    for c in retrieve.results:
        score = "—" if c.rerank_score is None else f"{c.rerank_score:.3f}"
        lines.append(f"- `{c.doc_id}` p.{c.page} · rerank={score}")
    return "\n".join(lines)


def _recent_samples(history: list[dict]) -> list[list[str]]:
    if not history:
        return []
    return [[h.get("query", "")] for h in history]


def _push_history(history: list[dict] | None, query: str, thread_id: str) -> list[dict]:
    h = list(history or [])
    h = [x for x in h if x.get("query") != query]
    h.insert(0, {"query": query, "thread_id": thread_id, "ts": datetime.utcnow().isoformat()})
    return h[:12]


SAMPLE_QUERIES: list[tuple[str, str, str, int]] = [
    ("fuel exhaustion forced landing", "en", "tsb", 2),
    ("engine failure after takeoff", "en", "tsb", 2),
    ("carburetor icing", "en", "all", 2),
    ("VFR flight into IMC", "en", "tsb", 2),
    ("alimentation en carburant", "fr", "tsb", 2),
    ("approach procedures helicopter", "en", "tc", 2),
]


def _sample_rows() -> list[list[str]]:
    return [[q] for q, _, _, _ in SAMPLE_QUERIES]


def _fmt_error(e: ApiError) -> str:
    detail = ""
    if isinstance(e.detail, dict) and "detail" in e.detail:
        detail = f" — {e.detail['detail']}"
    return f"❌ backend error ({e.status}){detail}"


# ─── minimal CSS ─────────────────────────────────────────────────────────────

_CSS = """
.gradio-container textarea {
  color: #0f172a !important;
  -webkit-text-fill-color: #0f172a !important;
}
"""


# ─── app ─────────────────────────────────────────────────────────────────────

def make_app(api: ApiClient | None = None) -> gr.Blocks:
    """Build the Gradio app. Tests pass a stubbed ``api``."""
    client = api or make_client()

    # ── handlers ──────────────────────────────────────────────────────────

    def on_ask(
        query_text: str,
        lang_v: str,
        source_v: str,
        max_hops_v: int,
        history: list[dict] | None,
        sess: dict,
        artifacts: dict,
    ):
        """Streaming generator. Yields (chat, sess, artifacts, hitl_row, history, recent)."""
        q = (query_text or "").strip()
        if not q:
            return

        thread_id = str(uuid.uuid4())
        IDX_THINK, IDX_SRC, IDX_ANS = 1, 2, 3

        chat_list: list[dict] = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "",
             "metadata": {"title": "🧠 Thinking…", "status": "pending"}},
            {"role": "assistant", "content": "_retrieving sources…_",
             "metadata": {"title": "📑 Sources (0)"}},
            {"role": "assistant", "content": ""},
        ]

        nop6 = (gr.update(),) * 6

        def _yield(chat=None, s=None, a=None, hitl=None, hist=None, rec=None):
            return (
                chat if chat is not None else gr.update(),
                s if s is not None else gr.update(),
                a if a is not None else gr.update(),
                hitl if hitl is not None else gr.update(),
                hist if hist is not None else gr.update(),
                rec if rec is not None else gr.update(),
            )

        yield _yield(chat=chat_list, hitl=gr.update(visible=False))

        text_buf: list[str] = []
        final_thread_id = thread_id
        sources_done = False
        artifacts = dict(artifacts)

        try:
            for ev in client.query_stream(q, thread_id, max_hops=max_hops_v):
                et, data = ev.get("event"), ev.get("data") or {}

                if et == "status":
                    msg = data.get("msg", "")
                    prev = chat_list[IDX_THINK].get("content") or ""
                    chat_list[IDX_THINK] = {
                        **chat_list[IDX_THINK],
                        "content": (prev + f"\n- {msg}").lstrip(),
                    }
                    yield _yield(chat=list(chat_list))

                elif et == "sources":
                    raw_sources = list(data.get("sources") or [])
                    retrieve = _sources_to_retrieve(raw_sources, q)
                    chat_list[IDX_SRC] = {
                        "role": "assistant",
                        "content": _sources_content(retrieve),
                        "metadata": {"title": f"📑 Sources ({len(retrieve.results)})"},
                    }
                    artifacts[IDX_SRC] = {
                        "gallery": _gallery_items(retrieve),
                        "chunks_md": _chunks_md(retrieve),
                    }
                    sources_done = True
                    yield _yield(chat=list(chat_list), a=dict(artifacts))

                elif et == "token":
                    text_buf.append(data.get("text", ""))
                    chat_list[IDX_ANS] = {**chat_list[IDX_ANS], "content": "".join(text_buf)}
                    yield _yield(chat=list(chat_list))

                elif et == "done":
                    final_thread_id = data.get("thread_id") or thread_id
                    n_steps = (chat_list[IDX_THINK].get("content") or "").count("- ")
                    chat_list[IDX_THINK] = {
                        "role": "assistant",
                        "content": chat_list[IDX_THINK].get("content") or "",
                        "metadata": {
                            "title": f"🧠 Thought ({n_steps} step{'s' if n_steps != 1 else ''})",
                            "status": "done",
                        },
                    }
                    # Fallback: if backend didn't send a separate sources event.
                    if not sources_done and data.get("sources"):
                        retrieve = _sources_to_retrieve(data["sources"], q)
                        chat_list[IDX_SRC] = {
                            "role": "assistant",
                            "content": _sources_content(retrieve),
                            "metadata": {"title": f"📑 Sources ({len(retrieve.results)})"},
                        }
                        artifacts[IDX_SRC] = {
                            "gallery": _gallery_items(retrieve),
                            "chunks_md": _chunks_md(retrieve),
                        }
                    new_history = _push_history(history, q, final_thread_id)
                    new_sess = {
                        **sess,
                        "thread_id": final_thread_id,
                        "draft": "".join(text_buf),
                        "query": q,
                    }
                    yield _yield(
                        chat=list(chat_list),
                        s=new_sess,
                        a=dict(artifacts),
                        hitl=gr.update(visible=True),
                        hist=new_history,
                        rec=gr.update(samples=_recent_samples(new_history)),
                    )
                    return

        except ApiError as e:
            chat_list.append({"role": "assistant", "content": _fmt_error(e)})
            yield _yield(chat=list(chat_list))
            return
        except Exception as e:  # noqa: BLE001
            chat_list.append({"role": "assistant", "content": f"❌ stream failed ({e})"})
            yield _yield(chat=list(chat_list))
            return

    def on_accept(chat: list[dict], sess: dict):
        new_sess = {k: v for k, v in sess.items() if k != "draft"}
        return chat, new_sess, gr.update(visible=False)

    def on_edit(sess: dict):
        draft = sess.get("draft", "")
        return gr.update(value=draft, visible=True), gr.update(visible=True)

    def on_save_edit(edited: str, chat: list[dict], sess: dict):
        new_chat = list(chat)
        for i in range(len(new_chat) - 1, -1, -1):
            m = new_chat[i]
            if m.get("role") == "assistant" and not m.get("metadata"):
                new_chat[i] = {**m, "content": edited}
                break
        return new_chat, {**sess, "draft": edited}, gr.update(visible=False), gr.update(visible=False)

    def on_cancel_edit():
        return gr.update(visible=False), gr.update(visible=False)

    def on_discard(chat: list[dict], sess: dict):
        new_chat = list(chat)
        while new_chat and new_chat[-1].get("role") == "assistant":
            new_chat.pop()
        if new_chat and new_chat[-1].get("role") == "user":
            new_chat.pop()
        new_sess = {k: v for k, v in sess.items() if k != "draft"}
        return new_chat, new_sess, gr.update(visible=False), gr.update(visible=False)

    def on_new(_sess):
        return [], {}, {}, gr.update(visible=False), gr.update(visible=False)

    def on_select(evt: gr.SelectData, artifacts: dict):
        idx = evt.index
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        art = artifacts.get(idx) or {}
        return (
            art.get("gallery") or [],
            art.get("chunks_md") or "_click a message to see chunks._",
        )

    def on_pick_sample(evt: gr.SelectData):
        idx = evt.index
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        if 0 <= idx < len(SAMPLE_QUERIES):
            q, lang_v, source_v, hops_v = SAMPLE_QUERIES[idx]
            return q, lang_v, source_v, hops_v
        return gr.update(), gr.update(), gr.update(), gr.update()

    def on_pick_recent(history: list[dict] | None, evt: gr.SelectData):
        if not history:
            return gr.update()
        idx = evt.index
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        if 0 <= idx < len(history):
            return gr.update(value=history[idx].get("query", ""))
        return gr.update()

    def on_healthz(_dummy=None):
        try:
            h = client.healthz()
        except ApiError as e:
            return f"❌ backend unreachable ({e.status})"
        except Exception as e:  # noqa: BLE001
            return f"❌ backend unreachable ({e})"
        flag = "🟢" if h.ok else "🟠"
        bits = [
            f"{name}={'ok' if c.ok else 'down'}"
            for name, c in (("qdrant", h.qdrant), ("neo4j", h.neo4j), ("ollama", h.ollama))
        ]
        return f"{flag} backend · " + " · ".join(bits)

    def on_load(history: list[dict] | None):
        return gr.update(samples=_recent_samples(history or [])), on_healthz()

    # ── layout ────────────────────────────────────────────────────────────

    with gr.Blocks(
        title="GraphRAG Aero",
        theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"),
        css=_CSS,
        fill_height=True,
    ) as app:
        sess      = gr.State({})
        artifacts = gr.State({})
        try:
            history = gr.BrowserState([])  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            history = gr.State([])

        # ── LEFT SIDEBAR ──────────────────────────────────────────────────
        with gr.Sidebar(position="left", open=True):
            gr.Markdown("## 🛩️ GraphRAG Aero")
            new_btn = gr.Button("＋ New chat", variant="primary")
            gr.HTML("<hr>")
            gr.Markdown("### Recent")
            recent = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[],
                label=None,
                type="values",
                samples_per_page=10,
            )
            gr.Markdown("### Sample queries")
            samples = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=_sample_rows(),
                label=None,
                type="values",
                samples_per_page=10,
            )
            gr.HTML("<hr>")
            lang = gr.Radio(["all", "en", "fr"], value="all", label="Lang")
            source = gr.Radio(["all", "tsb", "tc"], value="all", label="Corpus")
            with gr.Accordion("Advanced", open=False):
                max_hops = gr.Slider(1, 5, value=2, step=1, label="Max hops")
            gr.HTML("<hr>")
            health_md = gr.Markdown("_checking backend…_")

        # ── CENTER ────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            chat = gr.Chatbot(
                type="messages",
                show_copy_button=True,
                height="62vh",
                label=None,
                show_label=False,
            )
            with gr.Row(visible=False) as hitl_row:
                accept_btn  = gr.Button("✅ Accept",  variant="primary",   scale=0)
                edit_btn    = gr.Button("✏️ Edit",    variant="secondary", scale=0)
                discard_btn = gr.Button("✕ Discard", variant="secondary", scale=0)
            edit_box = gr.Textbox(
                label="Edit answer", lines=5, visible=False,
                placeholder="Edit the draft answer…",
            )
            with gr.Row(visible=False) as save_row:
                save_btn   = gr.Button("Save",   variant="primary",   scale=0)
                cancel_btn = gr.Button("Cancel", variant="secondary", scale=0)
            with gr.Row():
                query = gr.Textbox(
                    placeholder="Ask in English or French…",
                    lines=1,
                    max_lines=4,
                    scale=8,
                    show_label=False,
                    autofocus=True,
                    container=False,
                )
                ask_btn  = gr.Button("Send ↑", variant="primary",   scale=0)
                stop_btn = gr.Button("⏹ Stop",  variant="secondary", scale=0)

        # ── RIGHT SIDEBAR ─────────────────────────────────────────────────
        with gr.Sidebar(position="right", open=False):
            gr.Markdown("### Sources")
            with gr.Tabs():
                with gr.Tab("Pages"):
                    doc_gallery = gr.Gallery(
                        label=None,
                        show_label=False,
                        columns=1,
                        height=420,
                        object_fit="contain",
                    )
                with gr.Tab("Chunks"):
                    chunks_md_view = gr.Markdown("_click a message to see chunks._")

        # ── wiring ────────────────────────────────────────────────────────
        ask_outputs = [chat, sess, artifacts, hitl_row, history, recent]
        ask_event_a = ask_btn.click(
            on_ask,
            inputs=[query, lang, source, max_hops, history, sess, artifacts],
            outputs=ask_outputs,
        )
        ask_event_b = query.submit(
            on_ask,
            inputs=[query, lang, source, max_hops, history, sess, artifacts],
            outputs=ask_outputs,
        )

        stop_btn.click(None, cancels=[ask_event_a, ask_event_b])

        accept_btn.click(on_accept, [chat, sess], [chat, sess, hitl_row])
        edit_btn.click(on_edit, [sess], [edit_box, save_row])
        save_btn.click(on_save_edit, [edit_box, chat, sess], [chat, sess, edit_box, save_row])
        cancel_btn.click(on_cancel_edit, None, [edit_box, save_row])
        discard_btn.click(on_discard, [chat, sess], [chat, sess, hitl_row, save_row])

        new_btn.click(on_new, [sess], [chat, sess, artifacts, hitl_row, save_row])

        chat.select(on_select, [artifacts], [doc_gallery, chunks_md_view])
        recent.select(on_pick_recent, [history], [query])
        samples.select(on_pick_sample, None, [query, lang, source, max_hops])
        app.load(on_load, [history], [recent, health_md])

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = make_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
