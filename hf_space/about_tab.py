"""About tab — What / Why / How for GraphRAG Aero."""
from __future__ import annotations


_WHAT_MD = """\
## What is GraphRAG Aero?

GraphRAG Aero is a **graph-augmented retrieval system** for aviation safety documents.

It indexes the full public corpus of **Transport Canada Advisory Circulars** and
**TSB (Transportation Safety Board) investigation reports** in English, French, and Chinese,
then answers technical queries with cited, multi-hop synthesis — grounded in the actual text
and regulatory web, not hallucinated.

**Corpus (as-built):**
| Corpus | Docs | Lang |
|--------|------|------|
| TSB aviation investigation reports | 1,199 | EN + FR |
| TC Advisory Circulars | 242 | EN + FR |
| TTSB / CAAC reports (Chinese OCR) | 51 | ZH |
| **Total chunks** | **77,179** | **EN · FR · ZH** |

Figure images are captioned by a Qwen2.5-VL-7B vision model and embedded as
`kind=figure` chunks alongside the text, so a query about a runway diagram retrieves
the diagram's region, not just surrounding text.
"""

_WHY_MD = """\
## Why GraphRAG instead of plain vector search?

Plain dense retrieval finds *relevant text chunks* — but it misses the **regulatory web**.

A TSB finding like *"fuel tanks were empty at the time of the forced landing"* cites
**CAR 602.115** (fuel requirements). CAR 602.115 is implemented by **AC 602-001**
(fuel management guidance). A pilot, AME, or regulator asking about fuel-exhaustion
prevention needs all three layers, not just the incident narrative.

GraphRAG Aero adds a **Neo4j knowledge graph** populated from the corpus:

```
Occurrence → HAS_FINDING → Finding → CITES → Regulation → GUIDED_BY → AC
```

A single query traverses from incident → findings → regulations → guidance documents.
The LangGraph agent performs this multi-hop walk and feeds the graph context to the
synthesis prompt alongside the vector-retrieved chunks, producing answers that are:

- **Cited** — every sentence is grounded to `[doc_id p.page]`
- **Cross-layer** — incident narrative + regulatory reference + operational guidance
- **Bilingual** — BGE-M3 and the reranker-v2-m3 are natively multilingual; the same
  query works in EN, FR, and ZH without translation
"""

_HOW_MD = """\
## How does it work?

```
PDF corpus
    │
    ▼
Ingestion (pdfplumber + PaddleOCR GPU fallback + Qwen2.5-VL figure captions)
    │  fixed-size chunks 512 tok / 64 overlap · page_bboxes stored
    ▼
BGE-M3 dense+sparse → Qdrant  (77,179 pts, dim 1024, Cosine)
    │
    ▼ ANN search
BGE-M3 reranker-v2-m3 cross-encoder → top-K reranked chunks
    │
    ▼
Neo4j multi-hop traversal  (Occurrence→Finding→Regulation→AC)
    │
    ▼
LangGraph synthesis agent  (qwen3:4b via Ollama)
    │  cites [doc_id p.page] inline
    ▼
HITL gate  (human review before final answer delivery)
    │
    ▼
FastAPI backend → Gradio Space
```

**Key design decisions:**

| Concern | Choice | Why |
|---------|--------|-----|
| Vector store | Qdrant dense+sparse | Named sparse (BGE-M3 lexical weights) for exact-match recall on reg codes like "CAR 602.115" |
| Embeddings | BGE-M3 1024-dim | Best multilingual dense+sparse in its class; same model embeds EN/FR/ZH |
| Reranker | bge-reranker-v2-m3 | Multilingual cross-encoder; closes the recall/precision gap without a larger generator |
| Figure extraction | Qwen2.5-VL-7B | Vision-language captions make runway diagrams and photos retrievable as text |
| Graph | Neo4j | Native Cypher traversal; incident→regulation→guidance in one round-trip |
| Checkpointer | Postgres (LangGraph) | Durable HITL state; resume any thread after review |
| LLM | Ollama (local, 6.2 GB VRAM) | Runs on a single RTX 3060 Ti; no cloud API key, no data leaves the machine |
| Grounding | Stored `page_bboxes` | Region-level PDF highlight from ingestion metadata — no re-search at render time |
"""

_STACK_MD = """\
## Tech stack

**Ingestion:** Python · pdfplumber · PaddleOCR (GPU, PP-OCRv5) · HuggingFace Transformers
(Qwen2.5-VL-7B) · Docker (isolated image, avoids dep conflicts with agent runtime)

**Embeddings & retrieval:** FlagEmbedding (BAAI/bge-m3, bge-reranker-v2-m3) · Qdrant

**Graph:** Neo4j 5 · LangGraph (multi-hop agent) · PostgreSQL (LangGraph checkpointer)

**LLM:** Ollama · qwen3:4b Q4_K_M

**Backend:** FastAPI · Pydantic v2 · OpenTelemetry (OTLP → otel-collector)

**Frontend:** Gradio 5 (this Space) · Next.js 14 + TypeScript (local UI) · react-pdf

**Eval:** Recall@k · MRR · nDCG@k · bilingual test set (EN/FR/ZH, n=11)
"""


def build() -> None:
    """Create the About tab (static content, no API calls)."""
    import gradio as gr

    with gr.Tab("About"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(_WHAT_MD)
                gr.Markdown(_WHY_MD)
            with gr.Column(scale=1):
                gr.Markdown(_HOW_MD)
                gr.Markdown(_STACK_MD)
