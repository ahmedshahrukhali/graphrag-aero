"""About tab — short blurb and tech stack."""
from __future__ import annotations

_WHAT_MD = """\
## What is GraphRAG Aero?

GraphRAG Aero is a graph-augmented retrieval system for aviation safety documents. It indexes the full public corpus of Transport Canada Advisory Circulars and TSB investigation reports (over 77,179 chunks) in English, French, and Chinese. It parses both text and figure chunks using Qwen2.5-VL.
"""

_WHY_MD = """\
## Why?

Aviation regulations and incident reports form a dense, regulatory web. For instance, an incident might cite CAR 602.115, which relates to weather, which links to an advisory. A flat semantic search struggles here. We use Neo4j to store these relationships, allowing multi-hop synthesis across languages. Even if a document is in French, multilingual embeddings (BGE-M3) help bridge the gap.
"""

_HOW_MD = """\
## How it Works

The pipeline steps:
1. **Qdrant**: Vector search.
2. **reranker**: Cross-encoder scoring.
3. **Neo4j**: Knowledge graph traversal.
4. **LangGraph**: Orchestrates the LLM reasoning loop.
5. **HITL**: Human-in-the-loop overrides.

| Feature | Design Choice |
|---------|--------------|
| Bounding Boxes | Preserved as page_bboxes for frontend |
| Similarity | Cosine similarity |
"""

_STACK_MD = """\
## Tech Stack
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
        gr.Markdown(_WHY_MD)
        gr.Markdown(_HOW_MD)
        gr.Markdown(_STACK_MD)
    return page_col
