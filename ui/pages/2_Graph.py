"""Graph viewer — explore the Neo4j knowledge graph for a document occurrence."""
from __future__ import annotations

import sys
import os

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import api as backend

st.set_page_config(page_title="Graph — GraphRAG Aero", page_icon="🕸", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.node-card {
    border-left: 4px solid #4c9be8;
    background: rgba(76,155,232,.06);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
}
.node-card.finding   { border-color: #e57373; background: rgba(229,115,115,.06); }
.node-card.rec       { border-color: #81c784; background: rgba(129,199,132,.06); }
.node-card.reg       { border-color: #ffb74d; background: rgba(255,183,77,.06); }
.node-card.ac        { border-color: #9575cd; background: rgba(149,117,205,.06); }
.node-label {
    font-size: .72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: 4px;
    opacity: .7;
}
.node-text { font-size: .88em; line-height: 1.5; }
.node-meta { font-size: .74em; color: #888; margin-top: 4px; }
.stat-box {
    text-align: center;
    border: 1px solid rgba(128,128,128,.2);
    border-radius: 8px;
    padding: 12px 8px;
}
.stat-num { font-size: 1.8em; font-weight: 700; }
.stat-lbl { font-size: .78em; color: #888; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🕸 Graph Viewer")
st.caption("Explore knowledge-graph context for a TSB occurrence or TC AC document")

# ── lookup form ───────────────────────────────────────────────────────────────

st.markdown("""
**How to find a doc ID:**
Use the Corpus viewer to search for a document, then copy its `doc_id`
(format: `tsb/a13q0098` or `tc/ac-302-001`).
""")

col_id, col_btn = st.columns([5, 1])
with col_id:
    doc_id = st.text_input("Document ID", placeholder="tsb/a13q0098",
                           label_visibility="collapsed")
with col_btn:
    lookup_btn = st.button("Look up", type="primary", use_container_width=True)

st.divider()

# ── result ────────────────────────────────────────────────────────────────────

if lookup_btn and doc_id:
    with st.spinner("Querying graph…"):
        try:
            data = backend.graph_query(doc_id.strip())
        except Exception as exc:
            if "404" in str(exc):
                st.warning(f"No graph data found for **{doc_id}**. "
                           "The document may not have been upserted into the graph yet.")
            else:
                st.error(f"Graph query failed: {exc}")
            st.stop()

    findings        = data.get("findings", [])
    recommendations = data.get("recommendations", [])
    direct_regs     = data.get("direct_regs", [])
    acs             = data.get("acs", [])
    occ_url         = data.get("occ_url")

    # ── summary stats ─────────────────────────────────────────────────────────
    st.markdown(f"### `{data.get('occ_id', doc_id)}`")
    if occ_url:
        st.caption(f"Source: {occ_url}")

    cols = st.columns(4)
    for col, num, label, colour in [
        (cols[0], len(findings),        "Findings",        "#e57373"),
        (cols[1], len(recommendations), "Recommendations", "#81c784"),
        (cols[2], len(direct_regs),     "Regulations",     "#ffb74d"),
        (cols[3], len(acs),             "Advisory Circulars", "#9575cd"),
    ]:
        col.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-num" style="color:{colour}">{num}</div>'
            f'<div class="stat-lbl">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── findings ──────────────────────────────────────────────────────────────
    if findings:
        st.markdown("#### Findings")
        cause_f  = [f for f in findings if f.get("category") == "cause"]
        risk_f   = [f for f in findings if f.get("category") == "risk"]
        safety_f = [f for f in findings if f.get("category") == "safety_action"]
        other_f  = [f for f in findings if f.get("category") not in
                    ("cause", "risk", "safety_action")]

        for group_label, group in [
            ("Causes & Contributing Factors", cause_f),
            ("Findings as to Risk", risk_f),
            ("Safety Actions", safety_f),
            ("Other", other_f),
        ]:
            if not group:
                continue
            st.markdown(f"**{group_label}** ({len(group)})")
            for f in group:
                meta_parts = []
                if f.get("source_doc_id"):
                    meta_parts.append(f.get("source_doc_id"))
                if f.get("page"):
                    meta_parts.append(f"p.{f['page']}")
                if f.get("cites_reg"):
                    meta_parts.append(f"→ CAR {f['cites_reg']}")
                lang_tag = f.get("lang", "")
                meta_str = " · ".join(meta_parts)
                st.markdown(
                    f'<div class="node-card finding">'
                    f'<div class="node-label">finding · {lang_tag}</div>'
                    f'<div class="node-text">{f.get("text","")}</div>'
                    f'<div class="node-meta">{meta_str}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── recommendations ───────────────────────────────────────────────────────
    if recommendations:
        st.divider()
        st.markdown(f"#### Recommendations ({len(recommendations)})")
        for r in recommendations:
            rec_id   = r.get("id") or ""
            lang_tag = r.get("lang", "")
            meta_parts = []
            if r.get("source_doc_id"):
                meta_parts.append(r["source_doc_id"])
            if r.get("page"):
                meta_parts.append(f"p.{r['page']}")
            meta_str = " · ".join(meta_parts)
            label = f"recommendation {rec_id} · {lang_tag}" if rec_id else f"recommendation · {lang_tag}"
            st.markdown(
                f'<div class="node-card rec">'
                f'<div class="node-label">{label}</div>'
                f'<div class="node-text">{r.get("text","")}</div>'
                f'<div class="node-meta">{meta_str}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── regulations & ACs ─────────────────────────────────────────────────────
    col_r, col_a = st.columns(2)

    with col_r:
        if direct_regs:
            st.divider()
            st.markdown(f"#### Regulations cited ({len(direct_regs)})")
            for reg in sorted(direct_regs):
                st.markdown(
                    f'<div class="node-card reg">'
                    f'<div class="node-label">regulation</div>'
                    f'<div class="node-text">CAR {reg}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with col_a:
        if acs:
            st.divider()
            st.markdown(f"#### Advisory Circulars referenced ({len(acs)})")
            for ac in sorted(acs):
                st.markdown(
                    f'<div class="node-card ac">'
                    f'<div class="node-label">advisory circular</div>'
                    f'<div class="node-text">AC {ac}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if not any([findings, recommendations, direct_regs, acs]):
        st.info("The graph record exists but has no extracted findings, "
                "recommendations, or regulation links yet. "
                "Run `python -m agent.run upsert-graph --in data/chunks --extract` "
                "to populate entity nodes.")

elif lookup_btn and not doc_id:
    st.warning("Enter a document ID.")
else:
    st.info("Enter a document ID above to explore its knowledge-graph context.")

    with st.expander("Sample document IDs to try"):
        st.code("""tsb/a13q0098   # Fuel exhaustion, Sioux Lookout 2013
tsb/a08c0124   # Fuel starvation, Timmins 2008
tsb/a03a0013   # Forced landing following fuel exhaustion
tsb/a00a0051   # (sample from eval dataset)""", language=None)
