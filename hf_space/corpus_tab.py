"""Corpus Viewer tab — search the vector store and preview PDF pages."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hf_space.api_client import ApiClient

from hf_space.api_client import ApiError, RetrieveResponse
from hf_space.pdf_render import PdfRenderError, render_page_with_bbox

logger = logging.getLogger(__name__)


def _lang(choice: list[str]) -> list[str] | None:
    if not choice or len(choice) == 3:
        return None
    return [c.lower() for c in choice]


def _source(choice: list[str]) -> list[str] | None:
    if not choice or len(choice) == 4:
        return None
    return [c.lower() for c in choice]


def _gallery_items(resp: RetrieveResponse) -> list[tuple[Any, str]]:
    items: list[tuple[Any, str]] = []
    for c in resp.results:
        caption = (
            f"#{c.rank} · {c.doc_id} · p.{c.page} · "
            f"rerank={'—' if c.rerank_score is None else f'{c.rerank_score:.3f}'}"
        )
        if not c.source_url:
            continue
        try:
            img = render_page_with_bbox(
                c.source_url, c.page, c.bbox, doc_id=c.doc_id, draw_bbox=True,
            )
            items.append((img, caption))
        except PdfRenderError as e:
            logger.warning("pdf render: %s", e)
    return items


def build(client: "ApiClient") -> None:
    """Create the Corpus Viewer tab and wire its events."""
    import gradio as gr

    with gr.Column(visible=False) as page_col:
        gr.Markdown("### Corpus Viewer\nSearch the embedded corpus via dense retrieval + reranking.")
        with gr.Row():
            query = gr.Textbox(
                placeholder="e.g. fuel exhaustion forced landing",
                label="Query", scale=6, show_label=False,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)
        with gr.Row():
            c_lang = gr.CheckboxGroup(["en", "fr", "zh"], value=["en", "fr", "zh"], label="Language")
            c_source = gr.CheckboxGroup(["tsb", "tc", "ttsb", "caac"], value=["tsb", "tc", "ttsb", "caac"], label="Source")
            c_topk = gr.Slider(5, 50, value=10, step=5, label="Top-K")

        results_table = gr.Dataframe(
            headers=["#", "doc_id", "page", "section", "score", "text"],
            datatype=["number", "str", "number", "str", "str", "str"],
            label="Retrieved chunks",
            interactive=False,
            wrap=True,
        )

        with gr.Accordion("Page preview", open=False) as preview_acc:
            preview_gallery = gr.Gallery(
                value=[], preview=True, object_fit="contain",
                show_label=False, height=700,
            )

        def do_search(q: str, lang_v: list[str], src_v: list[str], topk: int):
            q = (q or "").strip()
            if not q:
                return gr.update(), gr.update(), gr.update()
            try:
                resp = client.retrieve(
                    q, lang=_lang(lang_v), source=_source(src_v), top_k=int(topk),
                )
            except ApiError as e:
                return [[0, f"Error: {e}", 0, "", "", ""]], gr.update(), gr.update()

            rows = []
            for c in resp.results:
                score = "—" if c.rerank_score is None else f"{c.rerank_score:.4f}"
                rows.append([c.rank, c.doc_id, c.page, c.section_title or "", score, c.text[:250]])

            items = _gallery_items(resp)
            return rows, gr.update(value=items), gr.update(open=bool(items))

        search_btn.click(do_search, [query, c_lang, c_source, c_topk],
                         [results_table, preview_gallery, preview_acc])
        query.submit(do_search, [query, c_lang, c_source, c_topk],
                     [results_table, preview_gallery, preview_acc])

    return page_col
