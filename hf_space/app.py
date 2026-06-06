"""Gradio Blocks app — the HuggingFace Space surface (Gradio 5.x).

Two-zone layout:
    LEFT  Sidebar — brand + New chat + lang/source filters + recent + samples + health
    CENTER Column — gr.Chatbot(type="messages") + HITL row + composer

Each answered turn renders its source PDF pages in a collapsible "Source pages"
panel below the chat — a standalone ``gr.Gallery`` in preview mode (one
full-width page + a thumbnail reel). A gallery embedded inside a Chatbot
message can't enter preview mode, so the panel lives outside the chat.

The Space remains a thin shell over the FastAPI backend at $BACKEND_URL.
The only locally-computed work is PDF-page-with-bbox rendering.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from hf_space.api_client import ApiClient, ApiError, RetrievedChunk, RetrieveResponse, make_client
from hf_space.pdf_render import PdfRenderError, render_page_with_bbox
from hf_space import corpus_tab, graph_tab, eval_tab, embedding_tab, about_tab


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


def _source_param(choice: str) -> str | None:
    return None if choice in ("all",) else choice


def _sources_to_retrieve(sources: list[dict], query: str) -> RetrieveResponse:
    """Adapt the ``sources`` list from /query/stream into a RetrieveResponse.

    The backend returns candidates in *reading order* (doc_id, page) so the
    synthesiser sees coherent passages. For the human-facing source list and
    page gallery we re-sort by rerank descending so ``#1`` is the most
    relevant chunk and the displayed scores read monotonically (None last).
    Display rank is decoupled from the model's citation tags, which key on
    [doc_id p.page], not on this index — so re-ordering here is safe.
    """
    ordered = sorted(
        sources,
        key=lambda s: (
            s.get("rerank_score") is None,
            -(s.get("rerank_score") or 0.0),
        ),
    )
    chunks: list[RetrievedChunk] = []
    for i, s in enumerate(ordered, start=1):
        bbox = s.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        chunks.append(RetrievedChunk(
            rank=i,
            doc_id=s.get("doc_id", ""),
            source_url=s.get("source_url"),
            section_title=s.get("section_title", ""),
            page=int(s.get("page", 0)),
            bbox=tuple(bbox),  # type: ignore[arg-type]
            # WS-B: region-level grounding rects (page, x0, top, x1, bottom).
            page_bboxes=tuple(tuple(float(v) for v in pb) for pb in s.get("page_bboxes", ())),
            lang=s.get("lang", ""),
            text=s.get("text", ""),
            # kind="figure" marks a chunk whose text is a Qwen2.5-VL caption+OCR
            # of a figure read at ingestion — surfaced visually in the gallery.
            kind=s.get("kind", "text"),
            ann_score=float(s.get("ann_score", 0.0)),
            rerank_score=None if s.get("rerank_score") is None else float(s["rerank_score"]),
        ))
    return RetrieveResponse(query=query, results=chunks)


# A citation in the synthesised answer looks like:  [tsb/a03q0109 p.4] … "the quote"
# The model frequently trails the page with the chunk's section title it saw in
# the citation block, e.g. [tsb/a03q0109 p.2 §26 JULY 2003] — so allow any run
# of non-']' chars after the page before the bracket closes.
# Capture (doc_id, page, quote): the citation tag, then the next quoted string
# (allowing a short run of words like 'states that' in between, but not crossing
# into the next citation tag).
_CITATION_RE = re.compile(
    r"\[(?P<doc>[^\]\s]+)\s+p\.\s*(?P<page>\d+)[^\]]*\][^\"\[\]]{0,160}?[\"“](?P<quote>[^\"”]+)[\"”]"
)


def _parse_citations(answer: str) -> dict[tuple[str, int], str]:
    """Map (doc_id, page) → the quoted span the answer attributes to it.

    First citation for a given (doc, page) wins. Used to anchor PDF
    highlights to exactly what the model cited, not the whole chunk.
    """
    out: dict[tuple[str, int], str] = {}
    for m in _CITATION_RE.finditer(answer or ""):
        try:
            key = (m.group("doc"), int(m.group("page")))
        except (TypeError, ValueError):
            continue
        quote = m.group("quote").strip()
        if key not in out and quote:
            out[key] = quote
    return out


# Bare citation tag — [doc_id p.page] — with no required trailing quote. This is
# what the model actually emits, so it's how we decide which pages were cited.
# In practice the model appends the chunk's section title inside the bracket
# (e.g. [tsb/a03q0109 p.2 §26 JULY 2003]), so tolerate any non-']' run after the
# page. _parse_citations stays the more precise quote-capturing path used to
# tighten the highlight when a quote exists.
_CITED_TAG_RE = re.compile(r"\[(?P<doc>[^\]\s]+)\s+p\.\s*(?P<page>\d+)[^\]]*\]")


def _cited_keys(answer: str) -> set[tuple[str, int]]:
    """Set of (doc_id, page) the answer cites, regardless of trailing quote."""
    keys: set[tuple[str, int]] = set()
    for m in _CITED_TAG_RE.finditer(answer or ""):
        try:
            keys.add((m.group("doc"), int(m.group("page"))))
        except (TypeError, ValueError):
            continue
    return keys


# Function words (EN + FR) dropped before highlighting query terms.
# (No longer used for PDF highlighting; we now use exact structured quotes from the LLM.)
_TERM_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "after", "before", "that",
    "this", "was", "were", "are", "has", "had", "not", "but", "its",
    "les", "des", "une", "dans", "pour", "par", "sur", "avec", "que", "qui",
    "aux", "est", "ont", "une", "lors", "apres", "avant",
}


def _query_terms(query: str, *, max_terms: int = 8) -> tuple[str, ...]:
    """Significant terms from the user's query, for on-page highlighting.

    Lowercased word tokens (EN + FR letters), minus stopwords and sub-3-char
    tokens, deduped, capped. These light up the title + every mention — e.g.
    "fuel exhaustion forced landing" → ("fuel","exhaustion","forced","landing"),
    which hits the report title "Forced Landing Following Fuel Exhaustion".
    """
    out: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[A-Za-zÀ-ÿ]+", (query or "").lower()):
        if len(tok) < 3 or tok in _TERM_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_terms:
            break
    return tuple(out)


def _page_regions(c: RetrievedChunk) -> tuple[tuple[float, float, float, float], ...]:
    """The chunk's stored region rects that fall on its own page (WS-B).

    ``page_bboxes`` entries are (page, x0, top, x1, bottom); keep the ones for
    ``c.page`` and drop the page index so they're plain (x0, top, x1, bottom)
    rects ready for ``render_page_with_bbox``."""
    out: list[tuple[float, float, float, float]] = []
    for pb in c.page_bboxes:
        if len(pb) == 5 and int(pb[0]) == c.page:
            out.append((pb[1], pb[2], pb[3], pb[4]))
    return tuple(out)


def _gallery_items(
    retrieve: RetrieveResponse,
    *,
    draw_bbox: bool = True,
    cited_dict: dict[tuple[str, int], str] | None = None,
) -> list[tuple[Any, str]]:
    cited_dict = cited_dict or {}
    items: list[tuple[Any, str]] = []
    for c in retrieve.results:
        key = (c.doc_id, c.page)
        is_cited = key in cited_dict
        quote = cited_dict.get(key)
        is_figure = c.kind == "figure"
        tag = " · ✦ cited" if is_cited else ""
        if is_figure:
            # Proof of image-intelligence: this chunk's text is the caption +
            # OCR that Qwen2.5-VL produced from the figure at ingestion. Show
            # the figure region (always boxed) next to that AI reading.
            cap_text = " ".join((c.text or "").split())
            if len(cap_text) > 90:
                cap_text = cap_text[:90] + "…"
            caption = f"🖼 AI-read figure · #{c.rank} · {c.doc_id} · p.{c.page}{tag}"
            if cap_text:
                caption += f' · "{cap_text}"'
        else:
            caption = (
                f"#{c.rank} · {c.doc_id} · p.{c.page} · "
                f"rerank={'—' if c.rerank_score is None else f'{c.rerank_score:.3f}'}{tag}"
            )
        if not c.source_url:
            continue
        # Highlight a page when the answer cites it (WS-B: draw the chunk's own
        # stored region(s), no quote-anchoring). Figures are always boxed — the
        # region *is* the image the vision model read, so we showcase it.
        do_box = bool(draw_bbox and (is_cited or is_figure))
        regions = _page_regions(c) if do_box else ()
        try:
            img = render_page_with_bbox(
                c.source_url, c.page, c.bbox,
                draw_bbox=do_box,
                region_bboxes=regions,
                terms=(quote,) if (do_box and quote) else (),
                box_images=do_box,
            )
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
    ("安捷飛航訓練中心 DA-40NG 發動機失效迫降高雄外海", "zh", "ttsb", 2),
    ("民用航空器维修计划和控制 CCAR-121", "zh", "caac", 2),
]


def _sample_rows() -> list[list[str]]:
    return [[q] for q, _, _, _ in SAMPLE_QUERIES]


_SAMPLE_CACHE_PATH = Path(__file__).with_name("sample_cache.json")


def _load_sample_cache() -> dict[str, dict]:
    """Pre-computed sample answers (built by ``build_sample_cache.py``).

    A missing/invalid file yields an empty dict so the Space still runs — a
    sample click just falls back to filling the composer instead of serving an
    instant answer.
    """
    try:
        return json.loads(_SAMPLE_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


SAMPLE_CACHE: dict[str, dict] = _load_sample_cache()


def _thought_from_trace(trace: list[dict]) -> tuple[str, int]:
    """Render a cached answer's trace as the same bullet list a live run shows.

    Returns ``(markdown, n_steps)``.
    """
    lines: list[str] = []
    for t in trace or []:
        node = t.get("node")
        if node == "retrieve":
            n = t.get("n_new") or t.get("n_merged") or "?"
            best = t.get("best_rerank")
            best_s = f" · best score {best:.2f}" if isinstance(best, (int, float)) else ""
            lines.append(f"Retrieved {n} chunks{best_s}")
        elif node == "graph_expand":
            lines.append(f"Graph context · {t.get('n_rows', '?')} edges")
        elif node == "synthesize":
            lines.append("Synthesised the answer")
    if not lines:
        lines = ["Loaded cached answer"]
    return "\n".join(f"- {ln}" for ln in lines), len(lines)


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
/* Fill the viewport: the app is a flex column (fill_height=True), so let the
   chat pane grow to consume all space above the HITL row + composer instead of
   sitting at a fixed height with a dead gap below. min-height:0 lets it shrink
   inside the flex parent so its own scrollbar (not the page) handles overflow. */
.chat-pane {
  flex-grow: 1 !important;
  min-height: 0 !important;
}
/* Source-pages panel: a standalone gr.Gallery in preview mode (one full-width
   page + thumbnail reel) inside a collapsible accordion below the chat. Give
   the page real height — readable text matters more than compactness. (The old
   2-col grid crammed two half-width pages per row, too small to read.) */
.pdf-inline img {
  max-width: 100% !important;
  height: auto !important;
  object-fit: contain;
}
"""


# ─── app ─────────────────────────────────────────────────────────────────────

def make_app(api: ApiClient | None = None) -> gr.Blocks:
    """Build the Gradio app. Tests pass a stubbed ``api``."""
    client = api or make_client()

    # Fixed message positions within a single answered turn. The optional
    # inline page gallery is appended after these (position 4+).
    IDX_USER, IDX_THINK, IDX_SRC, IDX_ANS = 0, 1, 2, 3

    # ── handlers ──────────────────────────────────────────────────────────

    def on_ask(
        query_text: str,
        lang_v: str,
        source_v: str,
        max_hops_v: int,
        show_bbox_v: bool,
        history: list[dict] | None,
        sess: dict,
        artifacts: dict,
    ):
        """Streaming generator. Yields (chat, sess, artifacts, history, recent)."""
        q = (query_text or "").strip()
        if not q:
            return

        thread_id = str(uuid.uuid4())

        chat_list: list[dict] = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": "",
             "metadata": {"title": "🧠 Thinking…", "status": "pending"}},
            {"role": "assistant", "content": "_retrieving sources…_",
             "metadata": {"title": "📑 Sources (0)"}},
            {"role": "assistant", "content": ""},
        ]

        def _yield(chat=None, s=None, a=None, hist=None, rec=None):
            return (
                chat if chat is not None else gr.update(),
                s if s is not None else gr.update(),
                a if a is not None else gr.update(),
                hist if hist is not None else gr.update(),
                rec if rec is not None else gr.update(),
            )

        yield _yield(chat=chat_list)

        text_buf: list[str] = []
        final_thread_id = thread_id
        sources_done = False
        artifacts = dict(artifacts)

        try:
            for ev in client.query_stream(
                q, thread_id, max_hops=max_hops_v,
                lang=_lang_param(lang_v), source=_source_param(source_v),
            ):
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
                        "content": _chunks_md(retrieve),
                        "metadata": {"title": f"📑 Sources ({len(retrieve.results)})",
                                     "status": "done"},
                    }
                    # Stash raw sources + query rather than pre-rendered images:
                    # the inline gallery is rendered after streaming (so PDF
                    # download/rasterise never delays tokens) and re-rendered on
                    # bbox-toggle from this stash + the finished answer's citations.
                    artifacts[IDX_SRC] = {"sources": raw_sources, "query": q}
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
                            "content": _chunks_md(retrieve),
                            "metadata": {"title": f"📑 Sources ({len(retrieve.results)})",
                                         "status": "done"},
                        }
                        artifacts[IDX_SRC] = {"sources": data["sources"], "query": q}
                    # Stash the finished answer so the inline gallery renderer can
                    # anchor highlights to the citations it contains.
                    if IDX_SRC in artifacts:
                        artifacts[IDX_SRC] = {
                            **artifacts[IDX_SRC],
                            "draft": "".join(text_buf),
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
                        hist=new_history,
                        rec=gr.update(samples=_recent_samples(new_history)),
                    )
                    # Source pages render into the collapsible "Source pages"
                    # panel via the chained render_pages handler (.then) once
                    # streaming ends — keeps PDF rasterise off the token path.
                    return

        except ApiError as e:
            chat_list.append({"role": "assistant", "content": _fmt_error(e)})
            yield _yield(chat=list(chat_list))
            return
        except Exception as e:  # noqa: BLE001
            chat_list.append({"role": "assistant", "content": f"❌ stream failed ({e})"})
            yield _yield(chat=list(chat_list))
            return

    def on_new(_sess):
        return [], {}, {}

    def _render_gallery(art: dict, show_bbox: bool) -> list[tuple[Any, str]]:
        srcs = art.get("sources") or []
        if not srcs:
            return []
        retrieve = _sources_to_retrieve(srcs, art.get("query", ""))
        draft = art.get("draft", "") if show_bbox else ""
        cited = _parse_citations(draft)
        # Fallback to _cited_keys if the LLM forgot the quote but still emitted the tag
        for k in _cited_keys(draft):
            if k not in cited:
                cited[k] = ""
        return _gallery_items(retrieve, draw_bbox=bool(show_bbox), cited_dict=cited)

    def render_pages(artifacts: dict, show_bbox: bool):
        """Render source pages into the collapsible panel, and open it.

        Runs after a turn finishes (chained via ``.then``) or when the bbox
        toggle flips. Local PDF raster only — never blocks the answer. Returns
        (gallery value, accordion open-state); empty + closed when no sources.
        """
        art = (artifacts or {}).get(IDX_SRC) or {}
        items = _render_gallery(art, bool(show_bbox)) if art.get("sources") else []
        return gr.update(value=items), gr.update(open=bool(items))

    def on_pick_sample(evt: gr.SelectData):
        """Click a sample → instant cached answer (no backend/LLM call).

        Returns the answered chat (user + thought + sources + answer) from the
        pre-built cache. Source pages render into the collapsible panel via the
        chained render_pages handler. Falls back to filling the composer if the
        query isn't cached.
        """
        nop7 = (gr.update(),) * 7
        idx = evt.index
        if isinstance(idx, (list, tuple)):
            idx = idx[0]
        if not (0 <= idx < len(SAMPLE_QUERIES)):
            return nop7
        q, lang_v, source_v, hops_v = SAMPLE_QUERIES[idx]
        cached = SAMPLE_CACHE.get(q)
        if not cached:
            # No cache → original behaviour: populate the composer, don't submit.
            return (q, lang_v, source_v, hops_v,
                    gr.update(), gr.update(), gr.update())

        retrieve = _sources_to_retrieve(cached.get("sources", []), q)
        thought, n_steps = _thought_from_trace(cached.get("trace", []))
        draft = cached.get("draft", "")
        chat_list: list[dict] = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": thought,
             "metadata": {
                 "title": f"🧠 Thought ({n_steps} step{'s' if n_steps != 1 else ''})",
                 "status": "done"}},
            {"role": "assistant", "content": _chunks_md(retrieve),
             "metadata": {"title": f"📑 Sources ({len(retrieve.results)})",
                          "status": "done"}},
            {"role": "assistant", "content": draft},
        ]
        new_artifacts = {IDX_SRC: {"sources": cached.get("sources", []), "query": q, "draft": draft}}
        new_sess = {"thread_id": cached.get("thread_id", ""), "draft": draft, "query": q}
        return (q, lang_v, source_v, hops_v,
                list(chat_list), new_sess, new_artifacts)

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
        sess         = gr.State({})
        artifacts    = gr.State({})
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
            lang = gr.Radio(["all", "en", "fr", "zh"], value="all", label="Lang")
            source = gr.Radio(["all", "tsb", "tc", "ttsb", "caac"], value="all", label="Corpus")
            with gr.Accordion("Advanced", open=False):
                max_hops = gr.Slider(1, 5, value=2, step=1, label="Max hops")
                show_bbox = gr.Checkbox(
                    value=True,
                    label="Highlight bbox on pages",
                    info="Draw the chunk's bounding box on rendered PDF pages.",
                )
            gr.HTML("<hr>")
            health_md = gr.Markdown("_checking backend…_")
            gr.HTML("<div style='text-align: right; color: gray; font-size: 0.8em; margin-top: 10px;'>v0.0.1</div>")

        # ── CENTER ────────────────────────────────────────────────────────
        with gr.Column(scale=1):
          with gr.Tabs():
            with gr.Tab("Chat"):
                chat = gr.Chatbot(
                    type="messages",
                    show_copy_button=True,
                    label=None,
                    show_label=False,
                    elem_classes=["chat-pane"],
                )
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
                with gr.Accordion("📄 Source pages", open=False) as pages_acc:
                    pages_gallery = gr.Gallery(
                        value=[],
                        preview=True,
                        object_fit="contain",
                        allow_preview=True,
                        show_label=False,
                        height=1000,
                        elem_classes="pdf-inline",
                    )
            corpus_tab.build(client)
            graph_tab.build(client)
            embedding_tab.build(client)
            eval_tab.build(client)
            about_tab.build()

        # ── wiring ────────────────────────────────────────────────────────
        ask_outputs = [chat, sess, artifacts, history, recent]
        ask_inputs = [query, lang, source, max_hops, show_bbox, history, sess, artifacts]
        pages_out = [pages_gallery, pages_acc]
        clear_pages = (lambda: (gr.update(value=[]), gr.update(open=False)))

        ask_event_a = ask_btn.click(on_ask, inputs=ask_inputs, outputs=ask_outputs)
        ask_event_a.then(render_pages, [artifacts, show_bbox], pages_out)
        ask_event_b = query.submit(on_ask, inputs=ask_inputs, outputs=ask_outputs)
        ask_event_b.then(render_pages, [artifacts, show_bbox], pages_out)

        def on_stop(chat_list):
            if not chat_list:
                return gr.update()
            new_chat = []
            for msg in chat_list:
                m = dict(msg)
                if "metadata" in m and m.get("metadata", {}).get("status") == "pending":
                    m["metadata"] = dict(m["metadata"])
                    m["metadata"]["status"] = "done"
                    if "Thinking" in m["metadata"].get("title", ""):
                        m["metadata"]["title"] = "🧠 Stopped"
                new_chat.append(m)
            return new_chat

        stop_btn.click(on_stop, inputs=[chat], outputs=[chat], cancels=[ask_event_a, ask_event_b])

        new_btn.click(
            on_new, [sess], [chat, sess, artifacts]
        ).then(clear_pages, None, pages_out)

        show_bbox.change(render_pages, [artifacts, show_bbox], pages_out)
        recent.select(on_pick_recent, [history], [query])
        samples.select(
            on_pick_sample,
            None,
            [query, lang, source, max_hops, chat, sess, artifacts],
        ).then(render_pages, [artifacts, show_bbox], pages_out)
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
