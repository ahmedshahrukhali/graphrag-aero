"""GraphRAG Aero — aviation document chat with RAG x-ray.

Conversation flow
─────────────────
User types → POST /query → backend runs agent → returns draft + trace (HITL pause)
Developer reviews draft + x-ray → Accept or edit draft
POST /resume → final answer rendered with cited sources

Session state keys
──────────────────
messages      list[Message]   completed turns (user + assistant)
hitl          HitlState|None  pending HITL turn waiting for approval
thread_ctr    int             monotonic counter for thread IDs
"""
from __future__ import annotations

import uuid
import sys
import os

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import api as backend

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Aero",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Hide Streamlit's Deploy button + main menu, but KEEP the sidebar toggle
   reachable. Previously `header {visibility: hidden}` wiped out the only
   way to open the sidebar. */
#MainMenu {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
footer {visibility: hidden;}
header {background: transparent;}
[data-testid="stSidebarCollapsedControl"] {visibility: visible !important;}

/* Page background */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Source citation card */
.src-card {
    border-left: 3px solid #4c9be8;
    background: rgba(76,155,232,.08);
    border-radius: 0 6px 6px 0;
    padding: 7px 12px;
    margin: 5px 0;
    font-size: .83em;
    line-height: 1.45;
}
.src-card .src-meta {
    color: #888;
    font-size: .78em;
    margin-bottom: 3px;
}
.src-card .src-text {
    color: inherit;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.score-pill {
    float: right;
    background: #4c9be8;
    color: #fff;
    border-radius: 12px;
    padding: 1px 8px;
    font-size: .72em;
    font-weight: 600;
}

/* HITL draft panel */
.hitl-banner {
    background: #fff8e1;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: .85em;
    color: #555;
}

/* Trace node row */
.trace-row {
    display: flex;
    gap: 10px;
    align-items: baseline;
    font-size: .78em;
    font-family: monospace;
    padding: 2px 0;
    border-bottom: 1px solid rgba(128,128,128,.12);
}
.trace-node-name { font-weight: 700; min-width: 140px; }
.trace-elapsed { color: #888; min-width: 60px; }
.trace-extras { color: #666; word-break: break-all; }

/* Health badge row */
.health-row { font-size: .8em; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── session state init ────────────────────────────────────────────────────────

def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []        # list of dicts
    if "hitl" not in st.session_state:
        st.session_state.hitl = None          # dict | None
    if "thread_ctr" not in st.session_state:
        st.session_state.thread_ctr = 0

_init_state()


# ── helpers ───────────────────────────────────────────────────────────────────

def _new_thread_id() -> str:
    st.session_state.thread_ctr += 1
    return f"t{st.session_state.thread_ctr}-{uuid.uuid4().hex[:6]}"


def _source_card(chunk: dict) -> str:
    doc = chunk.get("doc_id", "")
    sec = chunk.get("section_title") or ""
    page = chunk.get("page", "")
    score = chunk.get("rerank_score") or chunk.get("ann_score") or 0.0
    text = (chunk.get("text") or "")[:300]
    meta = f"{doc} · p.{page}" + (f" · {sec}" if sec else "")
    return (
        f'<div class="src-card">'
        f'<div class="src-meta">{meta}'
        f'<span class="score-pill">{score:.3f}</span></div>'
        f'<div class="src-text">{text}</div>'
        f"</div>"
    )


def _trace_html(trace: list[dict]) -> str:
    rows = []
    for node in trace:
        name = node.get("node", node.get("name", "?"))
        elapsed = node.get("elapsed_ms")
        extras = {k: v for k, v in node.items()
                  if k not in ("node", "name", "elapsed_ms")}
        elapsed_str = f"{elapsed:.0f} ms" if elapsed is not None else ""
        extras_str = ", ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        rows.append(
            f'<div class="trace-row">'
            f'<span class="trace-node-name">{name}</span>'
            f'<span class="trace-elapsed">{elapsed_str}</span>'
            f'<span class="trace-extras">{extras_str}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def _render_message(msg: dict) -> None:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant":
            sources = msg.get("sources") or []
            trace = msg.get("trace") or []
            if sources or trace:
                with st.expander(f"X-ray · {len(sources)} source(s) · {len(trace)} trace node(s)"):
                    if sources:
                        st.markdown("**Retrieved chunks**")
                        st.markdown(
                            "".join(_source_card(s) for s in sources),
                            unsafe_allow_html=True,
                        )
                    if trace:
                        st.markdown("**Agent trace**")
                        st.markdown(_trace_html(trace), unsafe_allow_html=True)


def _render_hitl(hitl: dict) -> None:
    """Render the pending HITL draft card. Returns True when resolved."""
    with st.chat_message("assistant"):
        st.markdown(
            '<div class="hitl-banner">⏸ <b>Draft ready</b> — review before finalising. '
            "Edit below or accept as-is.</div>",
            unsafe_allow_html=True,
        )
        draft_key = f"draft_{hitl['thread_id']}"
        edited = st.text_area(
            "Draft answer",
            value=hitl.get("draft") or "",
            height=200,
            key=draft_key,
            label_visibility="collapsed",
        )

        sources = hitl.get("sources") or []
        trace = hitl.get("trace") or []
        if sources or trace:
            with st.expander(f"X-ray · {len(sources)} source(s) · {len(trace)} trace node(s)"):
                if sources:
                    st.markdown("**Retrieved chunks**")
                    st.markdown(
                        "".join(_source_card(s) for s in sources),
                        unsafe_allow_html=True,
                    )
                if trace:
                    st.markdown("**Agent trace**")
                    st.markdown(_trace_html(trace), unsafe_allow_html=True)

        col_a, col_b, _ = st.columns([1, 1, 6])
        accept = col_a.button("✓ Accept", type="primary", key=f"accept_{hitl['thread_id']}")
        discard = col_b.button("✕ Discard", key=f"discard_{hitl['thread_id']}")

    if accept:
        _resolve_hitl(hitl, edited)
    elif discard:
        st.session_state.hitl = None
        st.rerun()


def _resolve_hitl(hitl: dict, draft: str) -> None:
    with st.spinner("Finalising…"):
        try:
            result = backend.resume(hitl["thread_id"], draft=draft or None)
        except Exception as exc:
            st.error(f"Resume failed: {exc}")
            return
    final = result.get("final") or draft or "(no answer)"
    st.session_state.messages.append({
        "role": "assistant",
        "content": final,
        "sources": hitl.get("sources") or [],
        "trace": result.get("trace") or hitl.get("trace") or [],
    })
    st.session_state.hitl = None
    st.rerun()


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✈ GraphRAG Aero")
    st.caption("Aviation document assistant · TC ACs + TSB reports")

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.hitl = None
        st.rerun()

    st.divider()
    st.markdown("**Settings**")

    lang_opt = st.radio("Language", ["Any", "English", "French"],
                        horizontal=True, label_visibility="collapsed")
    lang = None if lang_opt == "Any" else ("en" if lang_opt == "English" else "fr")

    source_opt = st.radio("Corpus", ["All", "TSB", "TC"],
                          horizontal=True, label_visibility="collapsed")
    source = None if source_opt == "All" else source_opt.lower()

    max_hops = st.slider("Agent hops", 1, 4, 2)

    st.divider()
    st.markdown("**Backend**")
    if st.button("Check health", use_container_width=True):
        try:
            h = backend.health()
            ok = h.get("ok", False)
            qdrant = "✓" if h.get("qdrant", {}).get("ok") else "✗"
            neo4j = "✓" if h.get("neo4j", {}).get("ok") else "✗"
            ollama = "✓" if h.get("ollama", {}).get("ok") else "✗"
            colour = "green" if ok else "red"
            st.markdown(
                f'<div class="health-row" style="color:{colour}">'
                f"Qdrant {qdrant} · Neo4j {neo4j} · Ollama {ollama}"
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Backend unreachable: {exc}")

    st.divider()
    st.markdown("**Conversation history**")
    if st.session_state.messages:
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        for i, m in enumerate(user_msgs[-8:], 1):
            label = m["content"][:50] + ("…" if len(m["content"]) > 50 else "")
            st.caption(f"{i}. {label}")
    else:
        st.caption("No messages yet")


# ── main chat area ────────────────────────────────────────────────────────────

st.markdown("### Ask a question about TC Advisory Circulars or TSB reports")

# Render completed messages
for msg in st.session_state.messages:
    _render_message(msg)

# Render pending HITL card (persists until accepted/discarded)
if st.session_state.hitl:
    _render_hitl(st.session_state.hitl)

# Chat input — disabled while HITL is pending
prompt = st.chat_input(
    "Ask about fuel systems, RNAV procedures, accident findings…",
    disabled=st.session_state.hitl is not None,
)

if prompt and not st.session_state.hitl:
    # Show user bubble immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    thread_id = _new_thread_id()

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            try:
                result = backend.query(prompt, thread_id, max_hops=max_hops)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.stop()

    # Result is a paused QueryPausedResponse — sources are the actual chunks
    # the synthesizer was given, no second /retrieve round-trip needed.
    draft = result.get("draft") or ""
    trace = result.get("trace") or []
    n_cands = result.get("n_candidates", 0)
    sources: list[dict] = result.get("sources") or []

    st.session_state.hitl = {
        "thread_id": thread_id,
        "query": prompt,
        "draft": draft,
        "trace": trace,
        "sources": sources,
        "n_candidates": n_cands,
    }
    st.rerun()
