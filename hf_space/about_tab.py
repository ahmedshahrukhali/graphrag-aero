"""About tab — short blurb and tech stack."""
from __future__ import annotations

_WHAT_MD = """\
## What is GraphRAG Aero?

GraphRAG Aero is a graph-augmented retrieval system for aviation safety documents. It indexes the full public corpus of Transport Canada Advisory Circulars and TSB investigation reports (over 77,179 chunks) in English, French, and Chinese. It parses both text and figure chunks using Qwen2.5-VL.
"""

_HOW_MD = """\
## How it Works & Tech Stack

The pipeline steps:
1. **Qdrant**: Vector search.
2. **reranker**: Cross-encoder scoring.
3. **Neo4j**: Knowledge graph traversal.
4. **LangGraph**: Orchestrates the LLM reasoning loop.

- **Ingestion:** Python, pdfplumber, PaddleOCR, Qwen2.5-VL-7B (for figures)
- **Embeddings & Retrieval:** FlagEmbedding (bge-m3, bge-reranker-v2-m3), Qdrant
- **Knowledge Graph:** Neo4j 5, LangGraph, PostgreSQL (checkpointer)
- **Generation:** Ollama (qwen3:4b Q4_K_M)
- **Backend & Frontend:** FastAPI, Pydantic v2, Gradio 5 (this Space)
- **Deployment:** Dual-backend architecture (auto-fallback to serverless Hugging Face Inference APIs if local GPU/Ollama is unavailable)
"""

def build() -> None:
    """Create the About tab."""
    import gradio as gr

    with gr.Column(visible=False) as page_col:
        gr.Markdown(_WHAT_MD)
        gr.Markdown(_HOW_MD)
    return page_col
