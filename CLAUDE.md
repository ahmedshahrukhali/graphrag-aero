# CLAUDE.md — Operating Instructions for Claude Code

Read MANIFEST.md first. Follow the resume pointer. Work one phase at a time.
After each phase: run tests, mark ☑ in MANIFEST.md, move resume pointer, pause for human review.

## Locked architecture
- **Vector store:** Qdrant — dense vectors only (no sparse)
- **Embeddings:** BGE-M3 dense (BAAI/bge-m3) via FlagEmbedding
- **Reranker:** BAAI/bge-reranker-v2-m3 (multilingual cross-encoder)
- **Chunking:** fixed-size 512 tokens, 64 overlap; capture metadata: doc_id, section_title, page, bbox
- **Ingestion:** pdfplumber primary; PaddleOCR fallback for image-only pages only; no unstructured.io
- **Checkpointer:** Postgres (LangGraph PostgresSaver) — not Neo4j
- **Graph:** Neo4j — Occurrence→Aircraft→Finding→Recommendation→Regulation→AC
- **Agents:** LangGraph; HITL interrupt before final answer delivery; surface full checkpoint trace
- **LLM:** gemma2:9b via Ollama (Q4_K_M, sequential VRAM loading)
- **Languages:** EN + FR (BGE-M3 + reranker-v2-m3 both multilingual)
- **Tracing:** OpenTelemetry

## Data acquisition (P1b — before ingestion)
Scrape and download to data/corpus/:
- TSB report index: https://www.tsb.gc.ca/eng/rapports-reports/aviation/index.html
  PDF pattern: https://www.bst-tsb.gc.ca/sites/default/files/rapports-reports/aviation/{ID}/eng/{id}.pdf
  Also download FR versions replacing /eng/ with /fra/
- TC Advisory Circulars: https://tc.canada.ca/en/aviation/reference-centre/advisory-circulars
  PDF pattern: https://tc.canada.ca/sites/default/files/
- No auth required. Be polite (rate limit requests, respect robots.txt).

## Phase plan
- P1b Data acquisition: scrape + download TC ACs + TSB investigation reports (EN+FR)
- P1  Ingestion: pdfplumber text+tables+bbox; PaddleOCR fallback if page is image-only; fixed-size chunk; SHA-256 dedup
- P2  Embed: BGE-M3 dense → Qdrant
- P3  Retrieve + rerank: Qdrant ANN → bge-reranker-v2-m3
- P4  Graph+agents: Neo4j schema; LangGraph multi-hop; PostgresSaver; HITL final-answer gate + trace
- P5  Eval: Recall@k / nDCG / MRR
- P6  FastAPI backend + OTel + Ollama
- P7  Next.js + TS + PDF highlight
- P8  HF Space Gradio multimodal EN+FR
- P9  Docs

## VRAM plan (3060Ti, 8GB)
Load sequentially — not simultaneously:
  BGE-M3 dense:       ~0.5 GB (embed query, then unload or keep small)
  reranker-v2-m3:     ~0.5 GB (rerank top-k, then unload)
  gemma2:9b Q4_K_M:   ~5.5 GB (generate)
  Total peak:         ~6.5 GB — fits.

## Conventions
- Ingestion isolated in its own image — PaddleOCR + torch dep conflicts with agent runtime.
- Mock all model loads in tests. CI must pass offline with no weight downloads.
- Every chunk carries: {doc_id, source_url, section_title, page, bbox, chunk_hash, lang}
- Secrets via .env only.

## Definition of done per phase
Code + mockable unit tests + dir README updated + MANIFEST.md ☑ + end-to-end on sample corpus.

## Model roles & discipline  *(append to CLAUDE.md, after "Definition of done per phase")*

You always know which model you are. **Open every response with your model tag** —
`[opus-4.7]`, `[sonnet-4.6]`, `[haiku-4.5]` — so authorship is traceable.

### Who may do what
| Model | Allowed work |
|---|---|
| **Opus 4.6+ / Sonnet 4.6+** | Phase code-of-record, architecture, real diffs, anything in `backend/`, `ingestion/`, `eval/`, `frontend/`, `hf_space/`. |
| **Haiku 4.5** | Grunt only: run commands, run tests, download corpus (P1b execution), apply a diff already written by a senior model, format/lint, generate dir trees, fill MANIFEST status. **Never authors phase logic.** |

**What "maintenance" means for Haiku:** retries, dependency installs, `.env`/config edits,
formatting, MANIFEST updates, and generated artifacts. **Any change to logic in a `.py` / `.ts`
source file is authoring → queue it** (see degraded-mode banner). Running existing code is fine;
editing its logic is not.

### Degraded mode = Haiku-only (usage limits hit)
When only Haiku is available, it does mechanical work **only**. Any task that would write
phase logic stops and queues — it does not improvise the code:

```
⛔ NEEDS SONNET 4.6+ — Haiku can't author phase logic. QUEUED, not written.
   Phase/task: <P#>.  Why blocked: <...>.  Approve override or wait for reset.
```

### Wrong-model / consequential = stop and wait (extends the existing per-phase HITL pause)
```
⚠️ HITL — about to <run / download / mutate data / commit>. Approve, or say what to change.
```
In chat this pause is honored by convention, not enforced. Honor it anyway. This is in
addition to the mandatory pause-for-review after every phase.

### Authorship / blame
Every commit carries a trailer so `git blame` shows who wrote each line:
```
Model: <model-name>
```
**Commit at every model handoff — one model per commit.** Opus commits its code
(`Model: opus-4.x`) *before* handing to Haiku; Haiku commits its run/maintenance work
separately. Without this the trailer is meaningless and blame blurs across models.
When marking a phase ☑ in MANIFEST.md, append `(by <model>)` next to the checkmark.

### Token discipline
- Work from **diffs**; **git + MANIFEST.md are the state** — do not re-read prior versions
  of files to "catch up." The **resume pointer** in MANIFEST.md tells you where you are.
- Only open a file when you're about to edit or run it.
- In chat: the user uploads the zip **once per session** to orient, then we work from diffs.
  Do not request a re-upload mid-session.
- No filler, no restating the task back, no summaries unless asked.

### Where things run (supersedes the short version)
- **Chat (Opus/Sonnet):** writes phase code + diffs. Sandbox only — tests logic, cannot
  reach the machine, real services, or the corpus.
- **Claude Code / Cowork on the machine:** reads/edits real files, runs docker/make/tests,
  downloads the corpus. Cowork can also use Skills, connectors, and computer use directly —
  no zip courier needed; it touches the repo itself.
- **Haiku-only degraded mode:** mechanical work on the machine; queue all phase logic.
- The repo (MANIFEST.md + git) is the message bus between surfaces. No model talks to
  another directly — everything passes through these files.
- **Opus dev step uses plan mode:** plan first, get approval, *then* implement — matches the
  per-phase HITL pause and stops code from being written before the plan is right.

### Handback (reuses your existing loop — no new file)
A phase is handed back when: tests pass, the dir README is updated, MANIFEST.md is ☑ `(by <model>)`,
the resume pointer is moved to the next phase, and the run is verified on the sample corpus.
That MANIFEST diff + `git diff` is everything the next model (or a fresh session) needs.
**If a phase was worked but MANIFEST wasn't updated, that's a bug — flag it before continuing.**
