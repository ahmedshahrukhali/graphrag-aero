# Plan — Add a Chinese PDF corpus + enable Chinese OCR (keep the PDF→answer demo)

**Status:** APPROVED (user, S19, 2026-06-01). NOT STARTED. In-repo copy of the approved plan so a
fresh session (possibly on another machine) can execute. Read this + MANIFEST resume pointer.

## Context
The demo's whole point is **PDF → grounded answer**: pdfplumber/OCR → chunk → BGE-M3 → rerank →
cited answer with **pixel bbox highlight** (WS-B) and **figure understanding** (Qwen3-VL, WS-C). The
"Chinese corpus" exists to prove that pipeline works on **Chinese documents, including OCR of
scanned/image-only Chinese pages**. CAAC was only *a source* of Chinese PDFs; its on-site index is
JS/JSONP and unscrapable — but **the PDFs themselves download fine** (caac.gov.cn reachable,
robots-green). So we do **not** abandon PDFs (an earlier Wikipedia/HTML idea was wrong); we get
Chinese PDFs from scrapable places and switch OCR to Chinese.

**Decisions (user):**
- Keep the **entire PDF pipeline + the English TC/TSB corpus** as-is (WS-B bbox, Qwen3-VL figures,
  PaddleOCR). The demo's shape does not change.
- Chinese sources = **BOTH**: **Taiwan TTSB** (Traditional, direct static PDF URLs) **and** **CAAC**
  (Simplified, enumerate PDF URLs via search engine/sitemap, bypassing the JS index).
- **Must showcase true OCR** → deliberately source **scanned / image-only Chinese PDFs** (older
  reports, scanned annexes) so `image_only` pages trigger PaddleOCR, not just born-digital text.

Downstream is already multilingual: BGE-M3 + bge-reranker-v2-m3 handle Chinese; Qwen3-VL captions
Chinese figures (verified S19); the WS-0 `corpus`/`page_bboxes` schema is language-agnostic. The two
real changes are **acquisition of Chinese PDFs** and **Chinese OCR config**.

## Approach (reuse first)

### 1. Chinese OCR — `ingestion/processing/ocr.py` (core change)
- Today `_get_ocr()` builds ONE PaddleOCR with `lang="latin"`. Make it a **per-language cache**
  `_ocr_by_lang: dict[str, PaddleOCR]` → `latin` (en/fr), `ch` (Simplified/CAAC), `chinese_cht`
  (Traditional/TTSB). `ocr_page(page, page_no, lang)` selects the model.
- Thread `lang` through: `processing/run.py::extract_pages_with_ocr(path, lang)` → `ocr_page(..., lang)`.
  `process_doc` already has the `DocRef` (lang + corpus).
- `use_angle_cls=True` for the Chinese models (scanned skew); keep off for latin. PaddleOCR
  `ch`/`chinese_cht` weights download on first use (ingest isn't offline-gated; optionally pre-cache
  in the ingestion Dockerfile).
- `image_only` pages already route to OCR (`pdf.extract_page`); OCR bbox is stored in PDF-point space
  → WS-B highlight + cited-page render work unchanged on Chinese pages.

### 2. Acquisition — Chinese PDFs → `data/corpus/zh/{ttsb,caac}/`
Reuse `acquisition/http_client.py` (rate-limited, idempotent SHA-256 `download`) verbatim.
- **`ingestion/acquisition/ttsb.py`** (new; distinct from Canada `tsb.py`): direct
  `https://www.ttsb.gov.tw/media/{id}/{file}.pdf`. Enumerate the report listing (verify static; else
  use the search-seed approach). Prefer older ASC-era reports (more scanned). Traditional Chinese.
- **`ingestion/acquisition/caac.py`** (new): bypass the JS index — enumerate CAAC PDF URLs via a
  **search-engine/sitemap seed** (`site:caac.gov.cn filetype:pdf` + markers `咨询通告` / `运行安全通告`),
  harvested once into a committed **seed-URL manifest** (e.g. `data/corpus/zh/caac_seed.txt`), then
  `download()` each directly. Reproducible, no live-search dependency. Simplified; target scanned docs.
- Wire both into `acquisition/run.py` (`--source ttsb|caac`) → `data/corpus/zh/...`.
- Curation (REINGEST §3): recon-flag born-digital vs scanned; curate a balanced set that **includes
  scanned docs**; emit an acquisition manifest.

### 3. Language + id plumbing (small)
- `processing/lang.py`: `Lang = "en"|"fr"|"zh"`; recognise `zh`.
- `processing/doc_id.py`: add `ttsb`, `caac` to `_KNOWN_SOURCES`; `corpus` = source; `zh` lang.
- `embed/jsonl.py` `LANGS`/`SOURCES` + backend `Literal` lang/source: admit `zh`, `ttsb`, `caac`
  (WS-0 record/payload fields already tolerate it; no schema redesign).

### 4. Everything else: unchanged
chunk.py, embed, retrieve, rerank, agent, backend, hf_space, WS-B render, Qwen3-VL WS-C. The 3D viz
(`build_embedding_space.py`) gains `zh` points after re-ingest; overlap shows EN↔ZH.

## Critical files
- Core: `ingestion/processing/ocr.py`, `ingestion/processing/run.py`, `…/lang.py`, `…/doc_id.py`.
- New: `ingestion/acquisition/{ttsb.py,caac.py}` (+ tests), edit `ingestion/acquisition/run.py`.
- Filters: `embed/jsonl.py`, `backend/schemas.py`.
- Ingestion Dockerfile: optionally pre-cache PaddleOCR `ch` + `chinese_cht`.
- Docs: REINGEST_PLAN §2, MANIFEST resume pointer + Corpus row, SESSIONS.

## Verification (offline-first, then a live scanned sample)
- Unit (mocked, offline): per-lang OCR routing (stub PaddleOCR → assert `zh`→`ch`/`cht`, `en`→`latin`);
  acquisition URL/seed parsing; `lang_for_path('zh')`; doc_id for ttsb/caac. Full suite green, no
  weights/network.
- **Live OCR showcase (acceptance test):** a **scanned** Chinese PDF → `image_only` pages OCR via the
  Chinese model, real 中文 in chunks → embed → Chinese `/query` → **cited answer with bbox highlight
  on the OCR'd Chinese region**, plus a Qwen3-VL caption on a Chinese figure. The PDF→answer demo, in
  Chinese, end-to-end.
- Curated re-ingest (English TC/TSB kept + zh/ttsb + zh/caac) last, after code + offline tests green.

## Sequencing
OCR per-lang routing (+ offline tests) → `ttsb.py` (direct PDFs) → `caac.py` (search-seed) →
lang/doc_id/filter plumbing → live scanned-OCR sample → curated re-ingest.

## Parked
ASN / Wikipedia / HTML: dropped. HK CAD reports are English-only (not a Chinese source).
Simplified vs Traditional handled per-source (caac→`ch`, ttsb→`chinese_cht`).
