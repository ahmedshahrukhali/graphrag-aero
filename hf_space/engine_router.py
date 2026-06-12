"""Engine selection with two-way fallback (S50).

Routes one Ask turn to the in-Space ZeroGPU engine or the FastAPI backend,
yielding the shared SSE event shapes either way. Kept out of ``app.py`` so the
routing logic imports (and tests) without gradio.
"""
from __future__ import annotations

import logging

from hf_space.api_client import ApiClient, ApiError
import hf_space.zerogpu_engine as zgpu

logger = logging.getLogger(__name__)


def stream_with_fallback(
    client: ApiClient, *, q: str, thread_id: str, max_hops: int,
    lang: str | list[str] | None, source: str | list[str] | None,
    history: list[dict] | None, use_zgpu: bool,
):
    """Yield SSE events with two-way engine fallback.

    Preferred engine: in-Space ZeroGPU (toggle ON + artifacts present), falling
    back to the backend on quota exhaustion (S47). The reverse direction (S50):
    if the backend/tunnel is down (ApiError 5xx) and the in-Space engine hasn't
    already burned its quota this turn, answer in-Space instead — but only when
    the backend stream died before its first event; a mid-stream fallback would
    replay status/sources/token events into the transcript.
    """
    quota_burned = False
    if use_zgpu and zgpu.available():
        try:
            yield from zgpu.answer_stream(q, lang=lang, source=source, history=history)
            return
        except Exception as e:
            if not zgpu.is_quota_error(e):
                raise
            quota_burned = True
            logger.warning("ZeroGPU quota error, falling back to local backend: %s", e)
            yield {"event": "status", "data": {"node": "fallback", "msg": "GPU Quota exceeded, falling back to backend…"}}

    streamed_any = False
    try:
        for e in client.query_stream(
            q, thread_id, max_hops=max_hops, lang=lang, source=source, history=history,
        ):
            streamed_any = True
            yield e
    except ApiError as e:
        if e.status < 500 or streamed_any:
            raise
        if quota_burned:
            raise ApiError(e.status, str(e), {
                "detail": "GPU quota exhausted and the local backend is unreachable — try again in a few minutes."
            }) from e
        if not zgpu.available():
            raise
        logger.warning("backend unreachable (%s), falling back to in-Space engine", e)
        yield {"event": "status", "data": {"node": "fallback", "msg": "Backend unreachable, falling back to in-Space engine…"}}
        yield from zgpu.answer_stream(q, lang=lang, source=source, history=history)
