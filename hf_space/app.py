"""Gradio Blocks app — the HuggingFace Space surface.

Three-pane layout (Tier-2 redesign):
    LEFT rail   — brand + "New question" + recent queries (BrowserState) +
                  feature shortcuts + health badge
    CENTER      — chat-style conversation (user bubble → draft card with
                  HITL editor → final answer card) + sticky composer
    RIGHT rail  — Sources / Trace / Logs tabs

The Space remains a thin shell over the FastAPI backend at $BACKEND_URL.
The only locally-computed work is the PDF-page-with-bbox rendering.
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
    """Work around gradio_client 4.x bug with bool JSON schemas — see prior
    history (`fix(hf-space): unbreak Gradio startup`)."""
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


# ─── pure helpers ────────────────────────────────────────────────────────────

def _lang_param(choice: str) -> str | None:
    return None if choice == "all" else choice


def _sources_to_retrieve(sources: list[dict], query: str) -> RetrieveResponse:
    """Adapt the ``sources`` list from /query/stream's done event into the
    same shape /retrieve returns, so the existing render helpers (gallery,
    chunks_md, logs) work unchanged. Backend sources don't carry ``rank``;
    we assign it by position."""
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
        score = "—" if c.rerank_score is None else f"{c.rerank_score:.3f}"
        out.append(
            f"**#{c.rank} · `{c.doc_id}` · p.{c.page}{section}** (rerank={score})\n\n"
            f"{snippet}…\n"
        )
    return "\n---\n".join(out)


def _logs_text(query: str, retrieve: RetrieveResponse, trace: list[dict], *, paused: bool) -> str:
    """Synthesize a pseudo-log view from the agent trace + retrieve hits.
    The backend doesn't stream real logs; this is a derived inspector view."""
    now = datetime.utcnow().strftime("%H:%M:%S")
    lines = [
        f"{now} INFO  agent.run started  query={query!r}",
        f"{now} INFO  retrieve k={len(retrieve.results)}",
    ]
    for c in retrieve.results[:5]:
        score = "—" if c.rerank_score is None else f"{c.rerank_score:.3f}"
        lines.append(f"{now} INFO    #{c.rank} {c.doc_id} p.{c.page} rerank={score}")
    for step in trace:
        node = step.get("node", "?")
        ms = step.get("elapsed_ms", 0)
        extras = " ".join(
            f"{k}={v!r}" for k, v in step.items() if k not in ("node", "elapsed_ms")
        )
        lines.append(f"{now} INFO  step {node:<14} elapsed={ms}ms  {extras}")
    if paused:
        lines.append(f"{now} WARN  interrupt before finalize — HITL gate")
        lines.append(f"{now} INFO  draft returned to client")
    return "\n".join(lines)


def _topbar(query: str, lang: str, source: str, max_hops: int, *, n_chunks: int = 0, status: str = "") -> str:
    bits = [f"**{query}**" if query else "_New conversation_"]
    if query:
        chips = [f"`{lang}`", f"`{source}`", f"`{max_hops}-hops`"]
        if n_chunks:
            chips.append(f"{n_chunks} chunks")
        bits.append("  ·  ".join(chips))
        if status:
            bits.append(f"_{status}_")
    return "  ·  ".join(bits)


def _recent_samples(history: list[dict]) -> list[list[str]]:
    """gr.Dataset samples — one row per recent query."""
    if not history:
        return []
    return [[h.get("query", "")] for h in history]


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


def _push_history(history: list[dict] | None, query: str, thread_id: str) -> list[dict]:
    h = list(history or [])
    h = [x for x in h if x.get("query") != query]  # de-dupe
    h.insert(0, {"query": query, "thread_id": thread_id, "ts": datetime.utcnow().isoformat()})
    return h[:12]


# ─── CSS ─────────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --rail-bg: #0f172a; --rail-ink: #e2e8f0; --rail-muted: #94a3b8;
  --rail-border: #1e293b; --rail-hover: #1e293b;
  --border: #e5e7eb; --ink: #0f172a; --muted: #64748b;
  --accent: #2563eb; --accent-soft: #dbeafe; --accent-ink: #1e40af;
  --ok: #16a34a; --warn: #d97706;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }
body, .gradio-container { background: #f8fafc; }

/* THREE-PANE LAYOUT */
#layout { display: grid !important; grid-template-columns: 260px 1fr 360px !important; gap: 0 !important; min-height: 100vh; }
#layout > * { min-width: 0; }

/* ── LEFT RAIL ── */
.left-rail {
  background: var(--rail-bg) !important; color: var(--rail-ink) !important;
  border-right: 1px solid var(--rail-border); padding: 14px 10px !important;
  display: flex; flex-direction: column; min-height: 100vh;
}
.left-rail .gr-markdown, .left-rail .gr-markdown * { color: var(--rail-ink) !important; }
.left-rail .brand-block { padding: 4px 8px 14px; font-size: 15.5px; font-weight: 600; }
.left-rail .brand-block .logo { font-size: 22px; margin-right: 6px; }
.left-rail .section-title { font-size: 10.5px !important; text-transform: uppercase; letter-spacing: .08em;
   color: var(--rail-muted) !important; padding: 12px 10px 4px; font-weight: 600; }
.left-rail .features-block .feature {
  padding: 7px 10px; border-radius: 7px; cursor: default; font-size: 13px;
  color: var(--rail-muted) !important; display: flex; align-items: center; gap: 9px;
}
.left-rail .features-block .feature:hover { background: var(--rail-hover); color: var(--rail-ink) !important; }
.left-rail button.newq, .left-rail .newq button {
  width: 100%; padding: 9px 12px !important; background: var(--accent) !important; color: white !important;
  border: 0 !important; border-radius: 9px !important; font-weight: 600 !important; font-size: 13.5px !important;
}
.left-rail button.newq:hover, .left-rail .newq button:hover { background: #1d4ed8 !important; }
.left-rail .recent-list .gr-dataset, .left-rail .recent-list table { background: transparent !important; border: 0 !important; }
.left-rail .recent-list td, .left-rail .recent-list tr {
  background: transparent !important; color: var(--rail-ink) !important; border: 0 !important;
  font-size: 12.5px !important; padding: 6px 10px !important; cursor: pointer;
}
.left-rail .recent-list tr:hover td { background: var(--rail-hover) !important; }
.left-rail .footer-block { margin-top: auto; padding: 12px 4px 4px; border-top: 1px solid var(--rail-border); }
.left-rail .footer-block, .left-rail .footer-block * { font-size: 12px !important; color: #86efac !important; }

/* compact filter pills in left rail */
.left-rail .filter-row { display: flex; align-items: center; gap: 8px; padding: 4px 10px 6px; }
.left-rail .filter-row .icon { font-size: 13px; opacity: .8; }
.left-rail .filter-pills { padding: 0 !important; background: transparent !important; border: 0 !important; }
.left-rail .filter-pills .gr-form, .left-rail .filter-pills .wrap, .left-rail .filter-pills fieldset {
  background: transparent !important; border: 0 !important; padding: 0 !important; gap: 4px !important;
}
.left-rail .filter-pills label { background: transparent !important; padding: 3px 8px !important; border-radius: 999px !important; border: 1px solid var(--rail-border) !important;
  font-size: 11.5px !important; color: var(--rail-muted) !important; cursor: pointer; }
.left-rail .filter-pills label:has(input:checked) { background: #1e3a8a !important; border-color: #2563eb !important; color: #dbeafe !important; font-weight: 600; }
.left-rail .filter-pills input[type=radio] { display: none !important; }
.left-rail .filter-pills span { color: inherit !important; font-size: inherit !important; }

/* ── CENTER ── */
.center-pane {
  background: white !important; border-right: 1px solid var(--border);
  display: flex !important; flex-direction: column;
  padding: 0 !important; min-height: 100vh;
}
.center-pane .topbar { border-bottom: 1px solid var(--border); padding: 12px 22px !important; background: white !important; }
.center-pane .topbar .gr-markdown { font-size: 14px !important; color: var(--muted); }
.center-pane .topbar .gr-markdown strong { color: var(--ink); font-weight: 600; }
.center-pane .messages { flex: 1; padding: 22px 28px !important; overflow-y: auto; gap: 14px !important; }

.center-pane .msg-user {
  align-self: flex-end !important; background: var(--accent) !important; color: white !important;
  padding: 10px 16px !important; border-radius: 14px 14px 4px 14px !important;
  font-size: 14.5px !important; max-width: 72%; margin-left: auto;
}
.center-pane .msg-user, .center-pane .msg-user * { color: white !important; }

.center-pane .draft-card {
  background: #fffbeb !important; border: 1px solid #fde68a !important; border-radius: 14px !important;
  padding: 14px 18px !important; max-width: 100%;
}
.center-pane .draft-card .agent-tag {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; color: #92400e !important; font-weight: 600;
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px;
}
.center-pane .draft-card .agent-tag .dot { width: 7px; height: 7px; border-radius: 999px; background: var(--warn); display: inline-block; }
.center-pane .draft-card textarea {
  background: white !important; border-radius: 8px !important; border: 1px solid #fef3c7 !important;
  color: #111827 !important; -webkit-text-fill-color: #111827 !important;
  opacity: 1 !important; font-size: 14px !important; line-height: 1.55 !important;
}

.center-pane .final-card {
  background: #f0fdf4 !important; border: 1px solid #bbf7d0 !important; border-radius: 14px !important;
  padding: 16px 22px !important; font-size: 14.5px !important; line-height: 1.7;
}
.center-pane .final-card .agent-tag {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 11px; color: #166534 !important; font-weight: 600;
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: 8px;
}
.center-pane .final-card .agent-tag .dot { width: 7px; height: 7px; border-radius: 999px; background: var(--ok); display: inline-block; }

.center-pane .composer { border-top: 1px solid var(--border); padding: 14px 22px 18px !important; background: white; }
.center-pane .composer-row { background: #f8fafc; border: 1px solid var(--border); border-radius: 14px !important; padding: 6px 6px 6px 14px !important; gap: 6px !important; }
.center-pane .composer-row textarea {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  color: #111827 !important; -webkit-text-fill-color: #111827 !important; opacity: 1 !important;
}
.center-pane .composer-row button { border-radius: 10px !important; min-width: 60px !important; }

/* ghost stop button — subdued, inline with Send */
.center-pane .composer-row .ghost-btn button, .center-pane .composer-row button.ghost-btn {
  background: white !important; color: var(--muted) !important;
  border: 1px solid var(--border) !important; font-weight: 500 !important;
}
.center-pane .composer-row .ghost-btn button:hover, .center-pane .composer-row button.ghost-btn:hover {
  background: #fef2f2 !important; color: #b91c1c !important; border-color: #fecaca !important;
}

/* HITL explainer card */
.center-pane .hitl-explainer {
  background: #f8fafc; border: 1px solid var(--border); border-left: 3px solid var(--accent);
  border-radius: 10px; padding: 12px 16px; font-size: 13px; color: var(--ink); line-height: 1.5;
}
.center-pane .hitl-explainer b { color: var(--accent-ink); }

/* discard button — secondary outline next to Finalize */
.center-pane .draft-card button.discard, .center-pane .draft-card .discard button {
  background: white !important; color: var(--muted) !important;
  border: 1px solid var(--border) !important; font-weight: 500 !important;
}

/* ── RIGHT RAIL ── */
.right-rail { background: white !important; padding: 0 !important; min-height: 100vh; border-left: 1px solid var(--border); }
.right-rail .tabs { padding: 0 !important; }
.right-rail .tab-nav button { font-size: 13px !important; padding: 12px 14px !important; }
.right-rail .gallery > * { background: transparent !important; }

/* status text */
.center-pane .status { font-size: 12px !important; color: var(--muted) !important; }
.center-pane .status code { background: #f1f5f9; padding: 1px 6px; border-radius: 4px; font-size: 11.5px; }
"""


# ─── app ─────────────────────────────────────────────────────────────────────

def make_app(api: ApiClient | None = None) -> gr.Blocks:
    """Build the Gradio app. Tests pass a stubbed ``api`` to drive
    handlers without a real backend."""
    client = api or make_client()

    def on_ask(query: str, lang: str, source: str, max_hops: int, history: list[dict] | None):
        """Streaming chat handler. Yields partial UI updates as SSE events
        arrive from /query/stream. The right-rail gallery + trace + sources
        populate only at the end (from the ``done`` event), so we don't
        block tokens on a separate /retrieve round-trip."""
        q = (query or "").strip()
        N_OUTPUTS = 14
        nop = (gr.update(),) * N_OUTPUTS

        def _emit(**overrides):
            """Build a 14-tuple with the named overrides; everything else nop."""
            keys = [
                "history", "recent", "topbar", "status", "user_msg",
                "hitl_explainer", "draft_card", "draft", "final_card",
                "gallery", "chunks_md", "trace", "logs", "sess",
            ]
            return tuple(overrides.get(k, gr.update()) for k in keys)

        if not q:
            yield _emit(status=gr.update(value="⚠️ enter a query.", visible=True))
            return

        thread_id = str(uuid.uuid4())

        # Initial UI flip: show the user bubble, status line, and an empty
        # draft card so tokens have a place to land.
        yield _emit(
            user_msg=gr.update(value=q, visible=True),
            topbar=gr.update(value=_topbar(q, lang, source, max_hops, status="streaming…"), visible=True),
            status=gr.update(value="_starting…_", visible=True),
            draft_card=gr.update(visible=True),
            draft=gr.update(value="", visible=True),
            final_card=gr.update(visible=False),
            hitl_explainer=gr.update(visible=False),
        )

        text_buf: list[str] = []
        sources: list[dict] = []
        trace_steps: list[dict] = []
        final_thread_id = thread_id

        try:
            for ev in client.query_stream(q, thread_id, max_hops=max_hops):
                et, data = ev.get("event"), ev.get("data") or {}
                if et == "status":
                    yield _emit(status=gr.update(value=f"_{data.get('msg','')}_", visible=True))
                elif et == "token":
                    text_buf.append(data.get("text", ""))
                    yield _emit(draft=gr.update(value="".join(text_buf), visible=True))
                elif et == "done":
                    sources = list(data.get("sources") or [])
                    trace_steps = list(data.get("trace") or [])
                    final_thread_id = data.get("thread_id") or thread_id
        except ApiError as e:
            yield _emit(status=gr.update(value=_fmt_error(e), visible=True))
            return
        except Exception as e:  # noqa: BLE001
            yield _emit(status=gr.update(value=f"❌ stream failed ({e})", visible=True))
            return

        # Stream finished — populate right-rail panels and reveal HITL gate.
        retrieve = _sources_to_retrieve(sources, q)
        new_history = _push_history(history, q, final_thread_id)
        n = len(retrieve.results)
        yield _emit(
            history=new_history,
            recent=gr.update(samples=_recent_samples(new_history)),
            topbar=gr.update(value=_topbar(q, lang, source, max_hops, n_chunks=n, status=f"draft ready · thread `{final_thread_id[:8]}…`"), visible=True),
            status=gr.update(value=f"_{n} sources · draft ready, edit and finalize_", visible=True),
            hitl_explainer=gr.update(visible=True),
            draft_card=gr.update(visible=True),
            draft=gr.update(value="".join(text_buf), visible=True),
            final_card=gr.update(visible=False),
            gallery=gr.update(value=_gallery_items(retrieve)),
            chunks_md=gr.update(value=_chunks_md(retrieve)),
            trace=gr.update(value=_trace_rows(trace_steps)),
            logs=gr.update(value=_logs_text(q, retrieve, trace_steps, paused=True)),
            sess={
                "thread_id": final_thread_id,
                "draft": "".join(text_buf),
                "retrieve": retrieve,
                "query": q,
                "trace": trace_steps,
            },
        )

    def on_finalize(edited_draft: str, sess: dict):
        """Commit the (possibly edited) draft as the final answer.

        The streaming /query/stream path doesn't go through the LangGraph
        HITL checkpoint, so there's no backend state to /resume — finalising
        is a UI-only promotion of draft→final. The agent trace already in
        the session is unchanged; we just append a finalize marker so the
        right-rail trace makes sense."""
        if not sess or "thread_id" not in sess:
            return (
                gr.update(), gr.update(), gr.update(),
                gr.update(value="⚠️ no active session."),
                gr.update(), gr.update(),
            )

        original = sess.get("draft") or ""
        body = edited_draft if isinstance(edited_draft, str) else ""
        edited_note = " · draft was edited by user" if body != original else ""

        retrieve = sess.get("retrieve") or RetrieveResponse("", [])
        # Synthesise a finalize trace step locally — backend already returned
        # the trace through the stream's done event; we just close it out.
        # Pull whatever trace was in the dataframe by re-deriving from sess.
        trace_steps = list(sess.get("trace") or [])
        trace_steps.append({"node": "finalize", "elapsed_ms": 0, "final_chars": len(body)})
        logs = _logs_text(sess.get("query", ""), retrieve, trace_steps, paused=False)
        thread = sess.get("thread_id", "?")[:8]
        return (
            gr.update(visible=True),
            gr.update(value=body or "_(no answer)_"),
            gr.update(visible=False),
            gr.update(value=f"_finalised · thread `{thread}…`{edited_note}_"),
            gr.update(value=_trace_rows(trace_steps)),
            gr.update(value=logs),
        )

    def on_new(_sess):
        """Reset to blank state. Used by '+ New question' (left rail) and
        the Discard button next to Finalize. Also resets lang/source/max_hops."""
        return (
            "",                                        # query box cleared
            "all", "all", 2,                          # lang/source/max_hops defaults
            gr.update(value="", visible=False),       # user msg hidden
            gr.update(visible=False),                  # HITL explainer hidden
            gr.update(visible=False),                  # draft card hidden
            gr.update(value="", visible=False),       # draft textbox cleared
            gr.update(visible=False),                  # final card hidden
            gr.update(value=""),                       # final markdown cleared
            gr.update(value="", visible=False),       # topbar hidden again
            gr.update(value="", visible=False),       # status hidden
            {},                                         # session blob cleared
        )

    def on_pick_sample(evt: gr.SelectData):
        """Clicking a sample query in the left rail populates the inputs
        without auto-submitting."""
        idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else 0)
        if 0 <= idx < len(SAMPLE_QUERIES):
            q, lang_v, source_v, hops_v = SAMPLE_QUERIES[idx]
            return q, lang_v, source_v, hops_v
        return gr.update(), gr.update(), gr.update(), gr.update()

    def on_pick_recent(history: list[dict], evt: gr.SelectData):
        if not history:
            return gr.update()
        idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else 0)
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
        bits = []
        for name, c in (("qdrant", h.qdrant), ("neo4j", h.neo4j), ("ollama", h.ollama)):
            bits.append(f"{name}={'ok' if c.ok else 'down'}")
        return f"{flag} backend · " + " · ".join(bits)

    def on_load(history: list[dict] | None):
        return gr.update(samples=_recent_samples(history or [])), on_healthz()

    theme = gr.themes.Default(primary_hue="blue", neutral_hue="slate")

    with gr.Blocks(title="GraphRAG Aero", theme=theme, css=_CSS) as app:
        # state
        sess = gr.State({})
        try:
            history = gr.BrowserState([])  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            # Older Gradio: fall back to ephemeral state (per-session, not persistent).
            history = gr.State([])

        with gr.Row(elem_id="layout"):
            # ─── LEFT RAIL ─────────────────────────────────────────────
            with gr.Column(scale=0, min_width=260, elem_classes="left-rail"):
                gr.Markdown('<div class="brand-block"><span class="logo">🛩️</span> GraphRAG Aero</div>')

                with gr.Column(elem_classes="newq"):
                    new_btn = gr.Button("＋  New question", variant="primary")

                gr.Markdown('<div class="section-title">Recent</div>')
                with gr.Column(elem_classes="recent-list"):
                    recent = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        samples=[],
                        label=None,
                        type="values",
                        samples_per_page=10,
                    )

                gr.Markdown('<div class="section-title">Sample queries</div>')
                with gr.Column(elem_classes="recent-list"):
                    samples = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        samples=_sample_rows(),
                        label=None,
                        type="values",
                        samples_per_page=10,
                    )

                with gr.Column(elem_classes="footer-block"):
                    health_md = gr.Markdown("_checking backend…_")

            # ─── CENTER ─────────────────────────────────────────────────
            with gr.Column(scale=3, elem_classes="center-pane"):
                topbar_md = gr.Markdown("", elem_classes="topbar", visible=False)

                with gr.Column(elem_classes="messages"):
                    status = gr.Markdown("", elem_classes="status", visible=False)

                    user_msg = gr.Markdown("", elem_classes="msg-user", visible=False)

                    hitl_explainer = gr.HTML(
                        '<div class="hitl-explainer">'
                        '<b>Human-in-the-Loop gate.</b> The agent paused before committing to a final answer. '
                        'Review the draft below — edit anything wrong, then Finalize. '
                        'This gate is unique to this app; most LLMs commit silently.'
                        '</div>',
                        visible=False,
                    )

                    with gr.Column(elem_classes="draft-card", visible=False) as draft_card:
                        gr.HTML('<div class="agent-tag"><span class="dot"></span> Draft · HITL gate — edit before finalizing</div>')
                        draft = gr.Textbox(
                            label="",
                            show_label=False,
                            lines=8,
                            interactive=True,
                            placeholder="Draft will appear here once the agent pauses at the HITL gate.",
                        )
                        with gr.Row():
                            finalize_btn = gr.Button("✅ Finalize", variant="primary")
                            discard_btn = gr.Button("Discard", variant="secondary", elem_classes="discard")

                    with gr.Column(elem_classes="final-card", visible=False) as final_card:
                        gr.HTML('<div class="agent-tag"><span class="dot"></span> Final answer</div>')
                        final_md = gr.Markdown("")

                with gr.Column(elem_classes="composer"):
                    # Per-turn filters live with the input — they change what
                    # this question searches, not a global app preference.
                    with gr.Row(elem_classes="composer-meta"):
                        lang = gr.Radio(
                            ["all", "en", "fr"], value="all", label="Lang",
                            container=False, scale=1, elem_classes="filter-pills",
                        )
                        source = gr.Radio(
                            ["all", "tsb", "tc"], value="all", label="Corpus",
                            container=False, scale=1, elem_classes="filter-pills",
                        )
                        with gr.Accordion("Advanced", open=False):
                            max_hops = gr.Slider(1, 5, value=2, step=1, label="Max hops")
                    with gr.Row(elem_classes="composer-row"):
                        query = gr.Textbox(
                            placeholder="Ask in English or French…  (Enter to send)",
                            lines=1,
                            max_lines=4,
                            scale=8,
                            show_label=False,
                            autofocus=True,
                            container=False,
                        )
                        ask_btn = gr.Button("Send ↑", variant="primary", scale=0)
                        stop_btn = gr.Button("⏹ Stop", variant="secondary", scale=0, elem_classes="ghost-btn")

            # ─── RIGHT RAIL ─────────────────────────────────────────────
            with gr.Column(scale=0, min_width=340, elem_classes="right-rail"):
                with gr.Tabs():
                    with gr.Tab("Sources"):
                        gallery = gr.Gallery(
                            label=None,
                            show_label=False,
                            columns=1,
                            rows=2,
                            height=360,
                            object_fit="contain",
                            elem_classes="gallery",
                        )
                        chunks_md = gr.Markdown("_run a query to see citations._")
                    with gr.Tab("Trace"):
                        trace = gr.Dataframe(
                            headers=["node", "elapsed_ms", "extras"],
                            datatype=["str", "number", "str"],
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("Logs"):
                        logs_code = gr.Code(value="", language=None, label=None)

        # ─── wiring ────────────────────────────────────────────────────
        ask_outputs = [
            history, recent,
            topbar_md, status,
            user_msg,
            hitl_explainer,
            draft_card, draft,
            final_card,
            gallery, chunks_md, trace, logs_code,
            sess,
        ]
        ask_event_a = ask_btn.click(
            on_ask,
            inputs=[query, lang, source, max_hops, history],
            outputs=ask_outputs,
        )
        ask_event_b = query.submit(
            on_ask,
            inputs=[query, lang, source, max_hops, history],
            outputs=ask_outputs,
        )

        # Stop cancels any in-flight ask. cancels=[...] also implicitly
        # frees the queue slot so the next request goes through.
        stop_btn.click(
            fn=lambda: gr.update(value="_stopped by user._", visible=True),
            inputs=None,
            outputs=[status],
            cancels=[ask_event_a, ask_event_b],
        )

        finalize_btn.click(
            on_finalize,
            inputs=[draft, sess],
            outputs=[final_card, final_md, hitl_explainer, status, trace, logs_code],
        )

        new_outputs = [
            query, lang, source, max_hops,
            user_msg, hitl_explainer, draft_card, draft, final_card, final_md,
            topbar_md, status, sess,
        ]
        new_btn.click(on_new, inputs=[sess], outputs=new_outputs)
        discard_btn.click(on_new, inputs=[sess], outputs=new_outputs)

        recent.select(on_pick_recent, inputs=[history], outputs=[query])
        samples.select(on_pick_sample, inputs=None, outputs=[query, lang, source, max_hops])

        app.load(on_load, inputs=[history], outputs=[recent, health_md])

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
