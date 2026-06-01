# WS-C figure-tier model decision — Qwen3-VL-8B (adopted)

**Author:** sonnet-4.6 (S19, 2026-05-31). **Decision:** adopt **Qwen3-VL-8B** as the figure-tier
model, replacing the Florence-2 + Moondream2 pair. **User-approved** (figure-tier scope only — NOT a
query-time generator). The InternVL3-8B bake-off is now **optional** (only revisit if Qwen3-VL
quality disappoints on ZH figures during WS-C).

## How it was tested (quick Ollama spike, not the WS-C implementation)
- `qwen3-vl:8b` pulled via Ollama 0.24.0 (6.1 GB on disk). RTX 3060 Ti, 8 GB, driver 596.49.
- Sample: `data/corpus/en/tsb/a13q0098.pdf` (the fuel-exhaustion forced-landing report), p.60 —
  the page carries 2 embedded photos. Rendered at 150 DPI; the largest figure was cropped from the
  pdfplumber `page.images` bbox and fed to the model (the real figure-tier shape: a crop, not a page).

## Quality — strong ✅
Prompt: *"This is a photograph from an aviation accident investigation report. Caption it in 1-2
sentences, then transcribe any visible text/labels."*

Response (60.7 s; prompt_eval 365 tok, eval 1046 tok):
> **Caption:** Aircraft fuel gauge panel showing pre-departure readings from CYHU, with labels for
> wing tanks (130 GAL), nacelle tank (57 GAL), and engine designations.
> **Transcribed:** "Readings prior to departure from CYHU", "WING TANKS 130 GAL", "NAC TANK 57 GAL",
> "FUEL", "E", "F", "PRIMARY ON", "SECONDARY ON", "CROSS FEED OPEN", "CROSS FEED CLOSED", "LH ENG",
> "RH ENG".

Accurate scene understanding **and** accurate region OCR (incl. domain labels), and topically
on-point (this report's cause is fuel exhaustion; the figure is the pre-departure fuel state). This
is the figure understanding the demo wants, and very likely beats the Florence-2/Moondream baseline.

## VRAM — borderline ⚠️ (acceptable for the offline figure tier)
- `ollama ps`: **SIZE 7.7 GB, PROCESSOR 28% CPU / 72% GPU** — spills ~28% to CPU at default.
- `nvidia-smi` while resident: **6927 MiB used / 1098 free** of 8192.
- Cause: the desktop GUI holds ~1.5 GB, leaving ~6.5 GB; the 7.7 GB model overflows by ~1.2 GB.
- **Mitigations / why it's fine:** (1) the figure tier runs in the **isolated ingestion image,
  offline batch** — a CPU-spill slowdown (~60 s/cropped figure here) is tolerable; it is NOT
  query-time and never contends with the generator's VRAM. (2) A **headless** ingestion run (no
  desktop GUI) would likely fit at/near 100% GPU. (3) WS-C runs it via **HF transformers** (per
  REINGEST_PLAN §4.2), where quantization/offload are controllable.
- Contrast: the text generator `qwen3:8b` fit cleanly at 6.2 GB / 100% GPU (`ws0_vram_measurement.md`).
  Do NOT use Qwen3-VL as the query-time generator on this card — it would spill and starve
  reranker/embed.

## Gotchas for the WS-C implementation
- **Feed crops, not full pages.** A full 150-DPI page tiles into more vision tokens than ctx 4096
  holds → empty/degenerate output. Crop the detected figure (pdfplumber `page.images`) first; ctx
  8192 was ample for a crop.
- **Thinking field:** Ollama ignored `think:false`; the model still emits a `thinking` field, but
  `response` is clean. Strip/ignore `thinking` when minting the Figure caption.

## Head-to-head vs InternVL3-8B (the bake-off §4.6 asked for — run for real, not asserted)
Same p.60 crop, same prompt. InternVL3-8B run in the embed container (host torch is CPU-only).
bitsandbytes 4-bit was a dead end in this image (bnb 0.45+ wants CUDA 13 `libnvJitLink.so.13`;
bnb 0.43 wants the old `triton.ops` removed in triton 3.x), so it ran **bf16 + CPU offload**.

| | Qwen3-VL-8B (Ollama) | InternVL3-8B (HF, bf16+offload) |
|---|---|---|
| Caption | fuel gauge panel, pre-departure readings from CYHU, wing/nacelle + engines | fuel gauge panel, accident-related, CYHU readings |
| OCR | WING TANKS **130** GAL · NAC TANK **57** GAL · CROSS FEED OPEN/CLOSED · LH/RH ENG · E/F | WING TANKS **150** GAL · NAC TANK **55** GAL · CROSS FEED OPEN/CLOSED · LH/RH ENG |
| Speed | ~60 s | **361 s** (CPU offload; 4-bit never loaded) |
| Setup | 2-min `ollama pull`, first try | transformers pin + bnb/CUDA/triton hell → slow CPU-offload fallback |

**Ground truth (report a13q0098 p.16):** "wing tanks with a total capacity of **130** U[S gal]" and
"nacelle tank with a usable capacity of **57** U[S gal]". → **Qwen3-VL read both correctly;
InternVL3 misread both (150/55).** Caveat retained: figure OCR of fine print is error-prone in
general — treat figure numbers as context, verify against report text when they matter.

**Decision confirmed with data:** Qwen3-VL-8B wins on this sample — accurate OCR (verified) *and*
far simpler to operate. InternVL3 dropped (not just "optional").

## Net
Qwen3-VL-8B collapses Florence-2 + Moondream2 into one model with better, verified output, at the
cost of a borderline VRAM fit that's acceptable because the figure tier is offline. Adopt for WS-C.
