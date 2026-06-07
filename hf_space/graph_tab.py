"""Graph Viewer tab — explore the Neo4j knowledge graph for any document."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hf_space.api_client import ApiClient


def _render_graph(data: dict) -> str:
    occ_id = data.get("occ_id", "?")
    occ_url = data.get("occ_url") or ""
    header = f"[{occ_id}]({occ_url})" if occ_url else f"`{occ_id}`"
    md = f"## {header}\n\n"

    findings = data.get("findings", [])
    if findings:
        md += f"### Findings ({len(findings)})\n\n"
        for f in findings:
            src = f.get("source_doc_id") or ""
            page = f.get("page", "?")
            cat = f" **[{f['category']}]**" if f.get("category") else ""
            reg = f" — cites **{f['cites_reg']}**" if f.get("cites_reg") else ""
            text = (f.get("text") or "")[:300]
            md += f"- `{src} p.{page}`{cat}{reg}\n  {text}\n\n"
    else:
        md += "_No findings extracted._\n\n"

    recs = data.get("recommendations", [])
    if recs:
        md += f"### Recommendations ({len(recs)})\n\n"
        for r in recs:
            rid = r.get("id") or ""
            src = r.get("source_doc_id") or ""
            page = r.get("page", "?")
            text = (r.get("text") or "")[:300]
            label = f"**{rid}**" if rid else ""
            md += f"- {label} `{src} p.{page}`\n  {text}\n\n"
    else:
        md += "_No recommendations extracted._\n\n"

    direct_regs = data.get("direct_regs", [])
    if direct_regs:
        md += f"### Regulations cited ({len(direct_regs)})\n\n"
        md += ", ".join(f"`{r}`" for r in direct_regs) + "\n\n"

    acs = data.get("acs", [])
    if acs:
        md += f"### Advisory Circulars ({len(acs)})\n\n"
        md += ", ".join(f"`{a}`" for a in acs) + "\n\n"

    if not findings and not recs and not direct_regs and not acs:
        md += "\n_This document has no graph edges. It exists as an isolated Occurrence node._\n"

    return md


def build(client: "ApiClient") -> None:
    """Create the Graph Viewer tab and wire its events."""
    import gradio as gr
    from hf_space.api_client import ApiError

    with gr.Column(visible=False) as page_col:
        gr.Markdown(
            "### Graph Viewer\n"
            "Look up the knowledge graph context for any document in EN, FR, or ZH: "
            "findings, recommendations, regulations, and advisory circulars.\n\n"
            "Enter a doc ID like `tsb/a13q0098` or `ttsb/3287_ttsb-aor-19-11-001`."
        )
        with gr.Row():
            doc_input = gr.Textbox(
                placeholder="e.g. tsb/a13q0098 (EN/FR) or ttsb/3287_ttsb-aor-19-11-001 (ZH)",
                label="Document ID", scale=6, show_label=False,
            )
            lookup_btn = gr.Button("Lookup", variant="primary", scale=1)

        result_md = gr.Markdown("_Enter a document ID and click Lookup._")

        def do_lookup(doc_id: str):
            doc_id = (doc_id or "").strip()
            if not doc_id:
                return "_Enter a document ID._"
            try:
                data = client.graph_lookup(doc_id)
            except ApiError as e:
                if e.status == 404:
                    return f"No graph data found for `{doc_id}`."
                return f"Error ({e.status}): {e}"
            except Exception as e:  # noqa: BLE001
                return f"Error: {e}"
            return _render_graph(data)

        lookup_btn.click(do_lookup, [doc_input], [result_md])
        doc_input.submit(do_lookup, [doc_input], [result_md])

    return page_col
