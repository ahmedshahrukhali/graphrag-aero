"""Corpus viewer — search and browse the chunk index via /retrieve."""
from __future__ import annotations

import sys
import os

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import api as backend

st.set_page_config(page_title="Corpus — GraphRAG Aero", page_icon="📄", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.chunk-card {
    border: 1px solid rgba(128,128,128,.2);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.chunk-meta {
    font-size: .78em;
    color: #888;
    margin-bottom: 6px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.chunk-meta span { white-space: nowrap; }
.tag {
    display: inline-block;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: .72em;
    font-weight: 600;
    margin-right: 4px;
}
.tag-tsb { background: #e3f2fd; color: #1565c0; }
.tag-tc  { background: #e8f5e9; color: #2e7d32; }
.tag-en  { background: #f3e5f5; color: #6a1b9a; }
.tag-fr  { background: #fff3e0; color: #e65100; }
.score-pill {
    float: right;
    background: #4c9be8;
    color: #fff;
    border-radius: 12px;
    padding: 1px 8px;
    font-size: .72em;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 📄 Corpus Viewer")
st.caption("Search the chunk index — results come from the live Qdrant collection via /retrieve")

# ── controls ──────────────────────────────────────────────────────────────────

col_q, col_lang, col_src, col_k = st.columns([4, 1, 1, 1])
with col_q:
    query = st.text_input("Search query", placeholder="fuel exhaustion, RNAV approach, …",
                          label_visibility="collapsed")
with col_lang:
    lang_opt = st.selectbox("Lang", ["Any", "EN", "FR"], label_visibility="collapsed")
    lang = None if lang_opt == "Any" else lang_opt.lower()
with col_src:
    src_opt = st.selectbox("Source", ["All", "TSB", "TC"], label_visibility="collapsed")
    source = None if src_opt == "All" else src_opt.lower()
with col_k:
    top_k = st.number_input("Top K", min_value=1, max_value=50, value=15,
                            label_visibility="collapsed")

search_btn = st.button("Search", type="primary")

# ── results ───────────────────────────────────────────────────────────────────

if search_btn and query:
    with st.spinner("Retrieving…"):
        try:
            data = backend.retrieve(query, lang=lang, source=source, top_k=int(top_k))
        except Exception as exc:
            st.error(f"Retrieve failed: {exc}")
            st.stop()

    results = data.get("results", [])
    st.markdown(f"**{len(results)} chunks** for *{query!r}*")
    st.divider()

    for chunk in results:
        doc_id = chunk.get("doc_id", "")
        src_tag = "tsb" if doc_id.startswith("tsb/") else "tc"
        lang_str = chunk.get("lang", "en")
        section = chunk.get("section_title") or ""
        page = chunk.get("page", "")
        ann  = chunk.get("ann_score", 0.0)
        rnk  = chunk.get("rerank_score") or ann
        url  = chunk.get("source_url") or ""
        text = chunk.get("text", "")

        tag_src  = f'<span class="tag tag-{src_tag}">{src_tag.upper()}</span>'
        tag_lang = f'<span class="tag tag-{lang_str}">{lang_str.upper()}</span>'
        score_pill = f'<span class="score-pill">{rnk:.3f}</span>'
        meta_parts = [
            f"<span>{doc_id}</span>",
            f"<span>p.{page}</span>",
        ]
        if section:
            meta_parts.append(f"<span><i>{section}</i></span>")

        meta_html = "".join(meta_parts)

        with st.container():
            st.markdown(
                f'<div class="chunk-card">'
                f'<div class="chunk-meta">{tag_src}{tag_lang}{meta_html}{score_pill}</div>',
                unsafe_allow_html=True,
            )
            preview = text[:400] + ("…" if len(text) > 400 else "")
            st.markdown(f"<small>{preview}</small>", unsafe_allow_html=True)

            with st.expander("Full text + metadata"):
                st.code(text, language=None)
                cols = st.columns(3)
                cols[0].metric("ANN score", f"{ann:.4f}")
                cols[1].metric("Rerank score", f"{rnk:.4f}")
                cols[2].metric("Page", page)
                if url:
                    st.markdown(f"[Source PDF]({url})")
                bbox = chunk.get("bbox")
                if bbox:
                    st.caption(f"bbox: {[round(v, 1) for v in bbox]}")

            st.markdown("</div>", unsafe_allow_html=True)

elif search_btn and not query:
    st.warning("Enter a search query.")
else:
    st.info("Enter a query above and click Search to browse the corpus chunk index.")
