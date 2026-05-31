# WS-0 VRAM Measurement Report: qwen3:8b vs gemma2:9b

## Measurement Context

Machine: RTX 3060 Ti (8 GB VRAM)
Test Date: 2026-05-31
GPU Driver: 596.49
Measurement Method: nvidia-smi + ollama ps + ollama show

## GPU State: Pre-Load

- **Total VRAM:** 8192 MiB (8 GB)
- **Free VRAM:** 7337 MiB (initial idle state)

## qwen3:8b Measurement (PRIMARY MODEL)

### Load Details
- **Model Pulled:** qwen3:8b (5.2 GB on disk)
- **Loading Method:** `ollama run qwen3:8b "hi"` → direct VRAM load
- **Time to Load:** ~15-20 seconds

### VRAM Footprint While Resident
- **ollama ps SIZE column:** 6.0 GB
- **ollama ps PROCESSOR column:** 100% GPU (confirms fully on GPU, no CPU spillover)
- **nvidia-smi used:** 6212 MiB (6.05 GB used, 1813 MiB free)

### Model Parameters & Quantization
- **Parameters:** 8.2B
- **Quantization:** Q4_K_M (4-bit key-value quantization)

## Comparison: gemma2:9b (Current Generation Model)

### Disk & Parameters
- **Model List SIZE:** 5.4 GB (on disk)
- **Parameters:** 9.2B
- **Quantization:** Q4_0 (4-bit, simpler than K-M variant)

### VRAM Footprint (from prior session)
- Expected VRAM load: ~5.5 GB (per project CLAUDE.md budget)
- Not re-measured in this session (idle at test start)

---

## VERDICT

### Does qwen3:8b fit fully on GPU?

**YES** ✓

- **PROCESSOR = 100% GPU:** Model is fully loaded on GPU with zero CPU spillover.
- **VRAM Used:** 6.2 GB / 8.0 GB total = 77.5% occupancy.
- **Headroom:** 1.8 GB free remains on-GPU.
- **Conclusion:** qwen3:8b (8.2B params, Q4_K_M) fits comfortably within the RTX 3060 Ti's 8 GB VRAM budget.

### Against Project VRAM Budget

- **Project Peak Budget:** ~6.5 GB (gemma2:9b + reranker + embeddings, sequential loading)
- **qwen3:8b Actual:** 6.2 GB (generation alone)
- **Margin:** +0.3 GB *over* gemma2:9b (~6.5 GB vs 5.5 GB estimated)
- **Risk Level:** **LOW** — qwen3:8b is marginally heavier but still fits with headroom; sequential unload of embedding/reranker before generation is respected, leaving ~1.8 GB free during generation.

### Secondary Findings

1. **Quantization Quality:** qwen3:8b uses Q4_K_M (more sophisticated grouping) vs gemma2:9b's Q4_0 (simpler). Both are 4-bit; K-M variant typically yields better perplexity at the cost of slightly larger footprint.
2. **Parameter Density:** qwen3:8b (8.2B) is 0.9B smaller than gemma2:9b (9.2B) but loads to ~0.7 GB *more* VRAM, likely due to Q4_K_M overhead or Ollama runtime overhead.
3. **GPU Saturation:** 100% GPU usage with 1.8 GB free allows for small headroom if KV-cache or batch size grows during inference.

---

## Recommendation for Senior Model

**qwen3:8b is viable as a gemma2:9b replacement on RTX 3060 Ti**, provided:
- Sequential model loading discipline is maintained (unload embeddings/reranker before generation).
- Inference batch size remains small (single or dual samples).
- No concurrent models run during generation.

If inference performance (quality/speed) benchmarks favor Qwen3-8B, proceed with the swap.
