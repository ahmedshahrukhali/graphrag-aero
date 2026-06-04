# REDESIGN_PLAN — retrieval + graph + agent loop + eval

> Origin: Session 24 (2026-06-04, opus-4.8), design-only walkthrough of the live
> code with the user. No source changed this session. Each item is anchored to a
> real file/symbol read during the review — execute against those anchors, do not
> re-derive. Out of scope: `backend/`, `frontend/`, HF Space direct deploy.

## Audit anchors (what's broken, where)
- `agent/nodes.py:218` `make_decide_continue` loops back to `retrieve` with the
  **same query**; `make_retrieve_node` (`:122`) re-embeds the identical string →
  identical ANN hits. The "multi-hop" is a no-op. `CONFIDENCE_THRESHOLD = 0.5`
  (`:42`) is hardcoded, never calibrated.
- `agent/graph.py:59` `interrupt_before=["finalize"]` — HITL gate is manual
  self-approval, no learning. Dead weight for a single-user demo.
- `retrieve/search.py` / `retrieve/pipeline.py` — **dense only**. No sparse path;
  jargon/identifiers ("CAR 602.115", "AC 702-001") rely on embedding recall.
- `graph/schema.py` + `graph/extract.py` — Occurrence-rooted, TSB/TC regex-bound.
  `Aircraft` declared but never populated (`extract.py:24` "reserved"). `upsert.py`
  node ids are `{occ_id}:...` → can't hold non-incident docs (manuals).
- `embed/run.py` / `embed/bge_m3.py` — BGE-M3 loaded dense-only; its native sparse
  output is discarded.
- LLM gemma2:9b → Qwen3-8B already decided (VRAM 6.2 GB, fits; S19,
  `docs/ws0_vram_measurement.md`).

---

## 1. Hybrid retrieval (dense + sparse)
Dense misses exact jargon/identifiers; aviation text is identifier-dense. Hybrid +
transparent fused scoring.
- `embed/bge_m3.py`: also return BGE-M3 sparse weights (FlagEmbedding
  `return_sparse=True`) alongside dense; keep dense API intact.
- `embed/qdrant.py` + `embed/run.py`: add a **named sparse vector** to the collection
  config; upsert sparse alongside dense. `--recreate` required (schema change). Point
  ID stays `chunk_hash`-derived.
- `retrieve/search.py`: add `sparse_search`; fuse with `dense_search` via RRF →
  one candidate list into the reranker (`retrieve/reranker.py` unchanged).
- Flag fused vs dense-only so eval can A/B.

## 2. Agent loop — real reformulation, kill fake hops
- New `agent/reformulate.py`: from hop-1 candidates, extract **high-importance,
  low-match** terms on the top failed page (salience ≠ similarity), build an
  expanded/re-weighted query ("Flange" → "Flange + Phoebe"). One mechanism =
  "different retrieval" + "different framing".
- `agent/nodes.py` `make_retrieve_node`: hop N>1 uses the reformulated query, not
  `state["query"]`. `make_decide_continue` keeps the hop budget; the loop now yields
  new candidates.
- Calibrate `CONFIDENCE_THRESHOLD`: histogram real rerank scores from the eval set;
  set the boundary at the observed gap. Record number + date in a code comment.

## 3. Remove HITL, add negative-feedback loop
- `agent/graph.py`: drop `interrupt_before=["finalize"]`; `synthesize → finalize`
  direct. Collapse `draft`/`final` → one `answer` in `agent/state.py`.
- New `agent/feedback.py` + store (reuse the LangGraph Postgres DB; new table
  `unaccepted_qa`): on **explicit** reject, write `{query, query_embedding, answer,
  terms, ts}`.
- New query cosine-similar (>0.80) to an `unaccepted_qa` row →
  (a) prompt states the prior attempt was wrong, (b) force reformulation (§2)
  excluding the chunks that produced the rejected answer, (c) **raise LLM temperature**
  for the retry (`agent/llm.py` — add a `temperature` arg; currently fixed). Surface
  high-salience terms as a "Related Terms" payload for the UI.
- A resolved retry updates/clears the matching row.

## 4. Graph schema restructuring (admit manuals)
- `graph/schema.py`: generic **Document** root + typed entities; Occurrence becomes
  one Document subtype. Relationships typed by role (`IMPLEMENTS`, `AFFECTED_BY`,
  `ILLUSTRATES`, `CITES`), not a fixed chain.
- `graph/upsert.py`: node identity → `{doc_id}:{type}:{hash}` (drop hard `occ_id`
  dependency). Populate `Aircraft` (currently dead).
- `graph/extract.py`: dispatch extractor by `corpus`/doc-type instead of one
  TSB-regex path. Land the **dispatch seam** now; manual/drawing extractors are a
  follow-up (no manual corpus in tree yet).

## 5. LLM swap → Qwen3-8B
- `agent/llm.py` + compose/env (`OLLAMA_MODEL`): gemma2:9b → qwen3:8b. Add the
  `temperature` pass-through (needed by §3). VRAM cleared (S19).

## 6. Eval — IN SCOPE (measure every change)
`eval/` already has `recall_at_k` / `reciprocal_rank` / `ndcg_at_k`
(`eval/metrics.py`) + runner (`eval/run.py`). Extend, don't rebuild.
- Expand `eval/dataset.jsonl` (currently ~4–7 q) with EN/FR/ZH + identifier-jargon
  queries dense-only is expected to miss (so hybrid's win is visible).
- `eval/run.py`: a runner variant per mode (dense / hybrid / hybrid+reformulation),
  A/B in one report; keep the stub-runner seam.
- New metric: **reformulation lift** = hop-2 (reformulated) vs hop-1 Recall@10 / MRR
  delta. Add to `eval/metrics.py`.
- New `eval/feedback_eval.py`: replay the `unaccepted_qa` flow (fail → reject → retry
  → assert resolve/improve). Audit trail that §3 works.

---

## Sequencing (commit boundaries, one model per commit)
1. §5 LLM swap + `temperature` pass-through (small; unblocks §3).
2. §1 hybrid + §6 dataset expansion → run eval, record dense-vs-hybrid numbers.
3. §2 reformulation + threshold calibration → eval reformulation-lift.
4. §3 remove HITL + feedback store/loop → `eval/feedback_eval.py`.
5. §4 graph restructure + dispatch seam (independent; can interleave).

Each step: mockable offline tests, live on sample corpus, then MANIFEST ☑ +
SESSIONS entry + resume-pointer move.

## Verification
- Offline: `python -m pytest` (+ new tests) — green, no weight downloads.
- Live retrieval: `python -m eval.run --json` before/after §1 and §2; numbers in the
  SESSIONS entry. Hybrid must beat dense on the jargon queries or it doesn't ship.
- Live loop: drive a known-failing query, reject, re-ask → confirm `unaccepted_qa`
  row, retry reformulates + raises temp + cites the prior miss. `eval/feedback_eval.py`
  asserts headless.
- Graph: `agent.run upsert-graph` on sample chunks → Document-rooted nodes +
  populated `Aircraft` + typed edges via `graph/query.py`.

## Deferred / accepted (not this program)
- Citations stay 1-chunk-1-cite (accepted looseness).
- OCR-first + multimodal chunking for manuals (needs Qwen3-VL figure-tier wiring).
  §4 lands only the schema seam.
