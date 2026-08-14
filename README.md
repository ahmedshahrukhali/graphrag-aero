# GraphRAG Aero 🛩️

A Graph RAG system for Aerospace Documents (Transport Canada & TSB reports), supporting English, French, and Chinese.

This project uses a combination of **Dense Retrieval** and a **Neo4j Knowledge Graph** to power a LangGraph agent, exposed through a FastAPI backend and a Gradio UI.

## 🏗️ Architecture

```mermaid
flowchart TD
    %% Ingestion Phase
    A[Raw PDFs] -->|pdfplumber & OCR| B[Text Chunks & Bboxes]
    
    %% Storage Phase
    B -->|BGE-M3 Embed| C[(Qdrant Vector DB)]
    B -->|Cypher Queries| D[(Neo4j Graph DB)]
    
    %% Retrieval & Agent Phase
    C -->|Dense Search| E[Retrieve Module]
    E -->|Cross-Encoder Rerank| F{LangGraph Agent}
    D <-->|Graph Traversal| F
    
    %% Backend & UI
    F -->|REST API| G[FastAPI Backend]
    G <-->|Proxy| H[Gradio Space UI]
    
    classDef database fill:#1d4ed8,stroke:#1e3a8a,stroke-width:2px,color:#fff
    classDef agent fill:#047857,stroke:#065f46,stroke-width:2px,color:#fff
    classDef ui fill:#be185d,stroke:#9d174d,stroke-width:2px,color:#fff
    
    class C,D database
    class F agent
    class H ui
```

## 🚀 Quickstart

1. **Setup Environment**: Copy `.env.example` to `.env` and fill in secrets.
2. **Start Services**: `docker compose up -d qdrant neo4j postgres ollama otel-collector`
3. **Run App**: `docker compose up --build backend hf-space`
4. **Access UI**: Open [http://localhost:7860](http://localhost:7860)

For the complete documentation, detailed layout, testing, and deployment instructions, see [README-detailed.md](README-detailed.md).
