# Sonnet task queue — assigned by opus-4.8 (S20, 2026-06-02)

Three self-contained briefs queued for **Sonnet 4.6+** (senior tier — these author
phase code-of-record, not grunt work). The repo is the message bus: read the
MANIFEST resume pointer first, work one task at a time, commit at each boundary
with the model trailers, and update MANIFEST + SESSIONS per CLAUDE.md.

**Context you need:** S20 landed the Chinese-OCR plan's *code* in three commits —
per-language OCR routing (`ca96406`), `ttsb.py` (`c1c6e40`), `caac.py` + seed
(`90208c8`). Full suite **398 passed**, all offline. None of the three scrapers
has been run live yet. Plan of record: [CHINESE_OCR_PLAN.md](CHINESE_OCR_PLAN.md)
and [REINGEST_PLAN.md](REINGEST_PLAN.md).

**Shared rules (all tasks):**
- Offline, mocked tests; CI must pass with no weight downloads / no network.
- Open every response with your model tag (`[sonnet-4.6]`).
- Commit trailers: `Model: sonnet-4.6` + `Co-Authored-By: sonnet-4.6 <chat-session-uuid>`
  (resolve the uuid per CLAUDE.md right before committing).
- One model per commit; update the MANIFEST resume pointer + SESSIONS entry.

**Suggested order & dependencies:** **D → B → A.** D and B are independent quick
wins (each one commit); do them first so A's live acceptance test benefits from
clickable Chinese citations (D) and runs its curated tail against frozen criteria
(B). A is the live capstone and is **HITL-gated** (downloads + mutates data).

---

## Task D — `source_url` for ttsb/caac doc_ids (small, code, no deps)

**Goal.** Chinese citations currently carry `source_url = None` (TC-style stopgap
from S20). Wire real URLs so the frontend/hf_space citation links are clickable
for ZH docs.

**Why.** `[caac/P020…]` / `[ttsb/9234_…]` citations render without a source link;
the demo's whole point is grounded, source-linked answers.

**The asymmetry (already analysed — don't re-derive):**
- **TTSB is reconstructable from the stem.** Stem is `{media_id}_{name}` (see
  `acquisition/ttsb.filename_for`). Split on the **first** `_` → rebuild
  `https://www.ttsb.gov.tw/media/{media_id}/{name}.pdf`. Add
  `ttsb.build_pdf_url(media_id, name)` and call it from `doc_id._source_url`.
- **CAAC is NOT reconstructable from the stem** — the on-disk name is just the
  `P{digits}.pdf` basename; the URL's directory path (`/XXGK/.../201511/`) is lost.
  So look it up: build a `{basename → url}` map once from
  `caac.load_seed_file()` (key = `caac.filename_for(url)`), memoise it, and have
  `doc_id._source_url('caac', stem)` return `map.get(f"{stem}.pdf")`. If the
  basename isn't in the seed, return `None` (graceful, same as today).

**Files.** `ingestion/processing/doc_id.py` (`_source_url`), `acquisition/ttsb.py`
(+`build_pdf_url`), maybe a tiny `acquisition/caac.py` helper for the lookup map.
`ingestion/processing/tests/test_doc_id.py` (+cases).

**Acceptance.** Offline tests: `doc_ref_for_path("data/corpus/zh/ttsb/9234_x.pdf").source_url`
== the rebuilt media URL; a caac stem present in the seed resolves to its seed URL;
a caac stem absent from the seed → `None`. Full suite green.

**Scope guards.** Don't change the stored schema (source_url already optional in
the record/payload). Percent-encoding caveat: TTSB names were decoded on download;
`requests` will re-encode the UTF-8 path — fine, note it in a comment, don't
over-engineer.

---

## Task B — Freeze curation admission criteria (REINGEST_PLAN §3)

**Goal.** Turn the prose bullets in [REINGEST_PLAN.md](REINGEST_PLAN.md) §3 into
**frozen, testable** admission rules + a reusable predicate + a curation manifest,
so the curated re-ingest (WS-F) is reproducible and honest.

**Why.** §3 is the last unfrozen gate before WS-F (noted in §6). "Quality over
volume" is the North Star (§0); right now nothing enforces it.

**Author into §3 (concrete numbers, marked FROZEN):**
- **Min content per doc/chunk** — reject docs whose total extracted text < N chars
  and chunks that are cover-only/boilerplate (reuse the page-marker heuristics
  already seen in bbox eval: `"- 2 -"`, date-only headers).
- **Dedup** — SHA-256 already exists (`processing/dedup.py`); state the policy
  (first-occurrence wins, cross-doc) as frozen.
- **Language sanity** — reject lang-misdetected docs (e.g. a `zh/` doc whose chunks
  are >X% ASCII letters → not actually Chinese).
- **Balance target** — a stated EN/TC vs ZH ratio band for the overlap demo to be
  honest (display is tag-stratified, but the underlying set should be balanced).
- **Reject reasons** — enumerate the closed set (empty/failed scan, cover-only,
  boilerplate-only, lang-misdetect, sub-threshold length).

**Implement.** `ingestion/processing/curation.py` — a pure
`admit(doc_record) -> Admission` (admit/reject + reason) + a manifest accumulator
(counts per corpus/lang, born-digital vs OCR, rejects-by-reason). Wire an
**opt-in** `--curate` flag into `processing/run.py` that filters + writes a
`curation_manifest.json`. Keep it off by default so existing runs are unchanged.

**Files.** `docs/REINGEST_PLAN.md` §3, new `ingestion/processing/curation.py` +
tests, `ingestion/processing/run.py` (flag + manifest write).

**Acceptance.** Offline tests for each admit/reject rule (cover-only rejected,
sub-threshold rejected, good doc admitted, lang-misdetect rejected); manifest
counts add up. Full suite green. §3 reads as FROZEN with numbers.

**Scope guards.** Don't run the live re-ingest here (that's Task A / WS-F). Don't
retune chunking (§4.5 owns that). Default behaviour unchanged unless `--curate`.

---

## Task A — Live scanned-Chinese-OCR acceptance test + fix pass  ⚠️ HITL-GATED

**Goal.** Prove the PDF→answer demo works end-to-end **in Chinese, through OCR**:
a scanned Chinese PDF → Chinese PaddleOCR → 中文 chunks → BGE-M3 → a Chinese
`/query` returns a cited answer with a bbox highlight on the OCR'd region.

**Why.** This is the acceptance criterion of the whole Chinese-OCR plan
([CHINESE_OCR_PLAN.md](CHINESE_OCR_PLAN.md) "Verification"). The code is in and
unit-green; only a live run proves it. Bugs *will* surface on real Chinese PDFs —
that's why this is senior, not Haiku.

**⚠️ Before running:** this DOWNLOADS PDFs, triggers a first-time PaddleOCR
`ch`/`chinese_cht` weights download (ingest is not offline-gated), and MUTATES the
index. Honor the CLAUDE.md HITL pause — confirm before the download/embed steps.

**Steps (small sample first — curate, don't bulk-fetch):**
1. `python -m ingestion.acquisition.run --source ttsb --limit 5`
   `python -m ingestion.acquisition.run --source caac --limit 5`
   (CAAC reachability already live-validated S20: 4 seed URLs → 200 `application/pdf`.)
2. Recon the pulled PDFs: which pages are `image_only` (OCR fires) vs born-digital?
   Deliberately keep at least one **scanned** doc per script (older CAAC ACs
   2003–2005; ASC-era TTSB) so OCR actually runs. This is where Task B's criteria
   help — run with `--curate` if B has landed.
3. `python -m ingestion.processing.run --source ttsb` (+ `caac`). Watch the
   `ocr fallback: … (ch|chinese_cht)` log lines fire. Confirm chunks contain real
   中文 (not mojibake / empty) and bboxes are in PDF-point space.
4. `python -m embed.run --source ttsb` (+ `caac`) into Qdrant (GPU per the existing
   compose reservations).
5. Chinese `/query` (backend up): e.g. a maintenance/AC topic present in the corpus.
   Verify a **cited** answer + that the hf_space/frontend bbox highlight lands on
   the OCR'd Chinese region (WS-B region render).

**Likely failure points to watch (fix as they surface):**
- PaddleOCR `ch`/`chinese_cht` not present in the ingestion image → may need to
  pre-cache in `ingestion/Dockerfile` (CHINESE_OCR_PLAN §1 / "Critical files").
- BGE-M3 tokenizer on CJK — chunk token counts; the 512-window may behave
  differently on Chinese (REINGEST §4.5). Confirm chunks aren't degenerate.
- Reranker dtype / "Already borrowed" (pre-existing flags in MANIFEST S5) under
  the new lang.

**Acceptance / handback.** A committed note (MANIFEST + SESSIONS) showing: N
Chinese docs ingested incl. ≥1 scanned, OCR fired (log evidence), a Chinese
`/query` transcript with a citation, and a screenshot/confirmation of the bbox
highlight on an OCR'd region. Then the curated re-ingest (WS-F) can follow.

**Scope guards.** Small sample first — do NOT kick off the full multi-hour WS-F
re-ingest in this task; that's the separate overnight run (REINGEST §7/§10, Haiku
monitors). Stop and report if OCR quality is poor rather than bulk-ingesting noise.
