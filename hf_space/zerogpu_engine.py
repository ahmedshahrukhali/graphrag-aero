import json
import logging
import os
import time
from pathlib import Path

# Try to import spaces, provide a mock for local dev without ZeroGPU dependencies
try:
    import spaces  # noqa: F401
except ImportError:
    class _MockSpaces:
        def GPU(self, duration=None):
            def decorator(func):
                return func
            return decorator
    spaces = _MockSpaces()

logger = logging.getLogger(__name__)

# Engine State
_qdrant_client = None
_embedder = None
_reranker = None
_graph_artifacts = None
_llm_pipeline = None

SPACE_INDEX_DIR = Path(os.environ.get("SPACE_INDEX_DIR", "data/space_index/v1"))
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "Qwen/Qwen3-14B")


def available() -> bool:
    """Return True if the engine can be used (artifacts exist)."""
    if not (SPACE_INDEX_DIR / "qdrant_local").exists():
        return False
    if not (SPACE_INDEX_DIR / "graph_context.json").exists():
        return False
    return True


def _load_models():
    """Load Qdrant, GraphArtifacts, embedders, rerankers, and generation model."""
    global _qdrant_client, _embedder, _reranker, _graph_artifacts, _llm_pipeline
    if _qdrant_client is not None:
        return

    from qdrant_client import QdrantClient
    from embed.bge_m3 import BGE_M3Embedder
    from retrieve.reranker import CrossEncoderReranker
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    from hf_space.graph_local import GraphArtifacts

    logger.info("Loading offline engine artifacts from %s", SPACE_INDEX_DIR)
    _qdrant_client = QdrantClient(path=str(SPACE_INDEX_DIR / "qdrant_local"))
    _graph_artifacts = GraphArtifacts.load(SPACE_INDEX_DIR)

    logger.info("Loading embedder and reranker")
    _embedder = BGE_M3Embedder()
    _reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3", batch_size=32)

    logger.info("Loading generator %s", GENERATION_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    _llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


def is_quota_error(exc: Exception) -> bool:
    """Return True if the exception is a ZeroGPU quota error."""
    msg = str(exc).lower()
    if "quota" in msg and "gpu" in msg:
        return True
    name = type(exc).__name__.lower()
    if "quota" in name:
        return True
    return False


@spaces.GPU(duration=120)
def answer_stream(
    query: str, 
    lang: str | list[str] | None = None, 
    source: str | list[str] | None = None, 
    history: list[dict] | None = None
):
    """
    Generator that mirrors backend/app.py /query/stream behavior locally inside the Space.
    Yields dictionary events matching the parsed SSE events: {"event": "...", "data": {...}}
    """
    _load_models()
    
    from retrieve.pipeline import anchored_retrieve
    from agent.prompts import build_user_prompt, SYSTEM_PROMPT, format_sources_block
    from transformers import TextIteratorStreamer
    from threading import Thread

    # 1. Retrieve
    yield {"event": "status", "data": {"node": "retrieve", "msg": "Retrieving relevant chunks…"}}
    collection = os.environ.get("QDRANT_COLLECTION", "aerospace_dense")
    chunks = anchored_retrieve(
        query, embedder=_embedder, reranker=_reranker, client=_qdrant_client, collection=collection,
        lang=lang, source=source
    )
    
    best_score = max((float(c.score) for c in chunks), default=0.0)
    yield {"event": "status", "data": {
        "node": "retrieve", 
        "msg": f"Retrieved {len(chunks)} chunks · best score {best_score:.2f}"
    }}
    
    sources = []
    for i, c in enumerate(chunks, 1):
        sources.append({
            "doc_id": c.record.doc_id,
            "source_url": c.record.source_url,
            "section_title": c.record.section_title or "",
            "page": c.record.page or 0,
            "bbox": c.record.bbox or [0.0, 0.0, 0.0, 0.0],
            "page_bboxes": getattr(c.record, "page_bboxes", ()),
            "lang": c.record.lang,
            "text": c.record.text,
            "kind": getattr(c.record, "kind", "text"),
            "ann_score": float(c.score),
            "rerank_score": float(c.score),
        })

    yield {"event": "sources", "data": {"sources": sources}}

    # 2. Graph events
    yield {"event": "status", "data": {"node": "graph_expand", "msg": "Looking up graph context…"}}
    occurrence_ids = [c["doc_id"] for c in sources]
    graph_context = _graph_artifacts.graph_context(occurrence_ids)
    recurring_context = _graph_artifacts.recurring_context(occurrence_ids)
    
    yield {"event": "status", "data": {
        "node": "graph_expand", 
        "msg": f"Graph context · {len(graph_context)} edges"
    }}

    # 3. Synthesize
    yield {"event": "status", "data": {"node": "synthesize", "msg": "Generating draft…"}}
    
    prompt = build_user_prompt(
        query=query,
        candidates=sources,
        graph_context=graph_context,
        recurring_context=recurring_context,
    )
    
    # Format history exactly as expected by the transformers chat template
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    streamer = TextIteratorStreamer(_llm_pipeline.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    def generate_fn():
        _llm_pipeline(
            messages,
            streamer=streamer,
            max_new_tokens=1024,
            return_full_text=False
        )

    thread = Thread(target=generate_fn)
    thread.start()

    pieces = []
    for chunk in streamer:
        pieces.append(chunk)
        yield {"event": "token", "data": {"text": chunk}}
        
    draft = "".join(pieces)
    
    # Append the deterministic Sources block
    sources_text = format_sources_block(sources)
    if sources_text:
        tail = f"\n\n---\n{sources_text}"
        draft = f"{draft.rstrip()}{tail}"
        yield {"event": "token", "data": {"text": tail}}

    trace = [{
        "node": "synthesize", "elapsed_ms": 0, "streamed": True,
        "prompt_chars": len(prompt), "draft_chars": len(draft),
    }]
    
    thread_id = "offline-thread" # Mocked thread_id since space chat doesn't rely heavily on thread_id for stream
    yield {"event": "done", "data": {
        "thread_id": thread_id,
        "draft": draft,
        "trace": trace,
    }}
