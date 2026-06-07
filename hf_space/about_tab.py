"""About tab — short blurb and tech stack."""
from __future__ import annotations

_ABOUT_MD = """\
## About GraphRAG Aero

GraphRAG Aero is a graph-augmented retrieval system for aviation safety documents. It indexes the full public corpus of Transport Canada Advisory Circulars and TSB investigation reports in English, French, and Chinese. It answers technical queries with cited, multi-hop synthesis—grounded in the actual text and regulatory web.

## Tech Stack
- **Ingestion:** Python, pdfplumber, PaddleOCR, Qwen2.5-VL-7B (for figures)
- **Embeddings & Retrieval:** FlagEmbedding (bge-m3, bge-reranker-v2-m3), Qdrant
- **Knowledge Graph:** Neo4j 5, LangGraph, PostgreSQL (checkpointer)
- **Generation:** Ollama (qwen3:4b Q4_K_M)
- **Backend & Frontend:** FastAPI, Pydantic v2, Gradio 5 (this Space)
"""

def build() -> None:
    """Create the About tab."""
    import gradio as gr

    with gr.Column(visible=False) as page_col:
        gr.Markdown(_ABOUT_MD)
    return page_col
