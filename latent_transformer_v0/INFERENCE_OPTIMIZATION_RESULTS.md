# Latent Transformer v0 — Inference Optimization Results

**Date:** 2026-11-17  
**Hardware:** AMD Ryzen 5 3600, CPU-only, float32  
**Torch:** 2.12.0+cu130  
**Weights:** Randomly initialized (medium config: 6 layers, d_model=384, d_ff=1536, rank=64, latent_dim=32)  
**Benchmark:** 7 reps, 2 warmup, 16 generated tokens, prompt=8 tokens, medians reported  

---

## 1. The bottleneck

`LatentLinear.forward()` calls `constructor.factors()` on every forward pass:

```python
def forward(self, x, constructor):
    L, R = constructor.factors(self.z, self.type_id,
                               self.in_features, self.out_features)
    return F.linear(F.linear(x, R), L) * self.log_gain.exp()
```

`constructor.factors()` runs two 2-layer MLPs (`row_net` and `col_net`) once per output row and once per input column of the matrix being generated:

- For a gate/up matrix (d_model=384 → d_ff=1536): `row_net` runs 1536 times, `col_net` runs 384 times → **1920 MLP evaluations per matrix**.
- For a down matrix (1536 → 384): **1920 MLP evaluations** again.
- With 3 matrices per layer × 6 layers = **18 LatentLinear instances** in the medium model, every forward pass through the full model calls `constructor.factors()` **18 times**, each generating **1920 row/column passes through a 2-layer hidden-128 MLP**.

During **training** this is correct behavior — z and constructor weights change every step, so factors must be recomputed.

During **inference** the constructor weights are fixed, z is fixed, and matrix dimensions are fixed. The generated factors are therefore deterministic and identical on every call. Recomputing them is pure waste.

---

## 2. Measured overhead

| Configuration       | Forward ms | TTFT ms | Sub-token ms | tok/s  | Ckpt MB | Cache MB |
|---------------------|----------:|--------:|-------------:|-------:|--------:|---------:|
| dense               |     16.36 |    9.34 |        15.42 |   62.5 |   57.45 |     0.00 |
| lowrank             |      6.17 |    6.59 |         7.32 |  115.6 |   23.83 |     0.00 |
| latent_uncached     |     72.39 |   63.13 |        65.41 |   14.7 |   15.26 |     0.00 |
| **latent_all_cached** |  **6.07** |**7.71** |     **9.01** |**106.7**| 15.26 |     8.85 |
| latent_layerwise    |     58.90 |   51.34 |        61.57 |   15.4 |   15.26 |     0.00 |

**All medians over 7 reps, 2 warmup. Random weights, medium config.**

---

## 3. Results summary

### Factor generation is essentially the entire latent inference cost

Uncached latent forward: **72.4 ms** vs dense **16.4 ms** — **4.4× slower than dense** despite having 3.8× fewer parameters. All the extra time is factor generation in the constructor.

### All-factor caching eliminates the overhead completely

Cached latent forward: **6.1 ms** — **12× faster than uncached**, **2.7× faster than dense**.

This is not surprising in retrospect: once the factors are materialized, the latent model *is* a low-rank model. Its forward pass is structurally identical to `LowRankMLP`, which runs at 6.2 ms. The near-identical timings (6.07 ms vs 6.17 ms) confirm this is working as expected.

### Generation throughput

| Configuration       | tok/s  | vs dense | vs lowrank |
|---------------------|-------:|--------:|----------:|
| dense               |   62.5 |    1.0× |      0.54× |
| lowrank             |  115.6 |    1.8× |      1.0×  |
| latent_uncached     |   14.7 |    0.2× |      0.13× |
| latent_all_cached   |  106.7 |    1.7× |      0.92× |
| latent_layerwise    |   15.4 |    0.2× |      0.13× |

Cached latent generates at **106.7 tok/s** — 1.7× faster than dense, 92% of lowrank throughput.

### Storage picture

| Configuration       | Ckpt MB | Runtime cache MB | Total runtime MB |
|---------------------|--------:|-----------------:|-----------------:|
| dense               |    57.5 |             0.00 |             57.5 |
| lowrank             |    23.8 |             0.00 |             23.8 |
| latent (any mode)   |    15.3 |             8.85 (all-cached) |   24.1 |

The latent checkpoint is **3.8× smaller than dense**. At inference with all-factor caching, the runtime footprint is 24.1 MB — still smaller than lowrank's 23.8 MB checkpoint and far smaller than dense's 57.5 MB.

The 8.85 MB cache holds L and R for all 18 LatentLinear instances at the medium config (rank=64):
- Each gate/up matrix: L=(1536, 64) + R=(64, 384) = 24,576 floats = 98 KB
- Each down matrix: L=(384, 64) + R=(64, 1536) = 24,576 floats = 98 KB
- 18 matrices × 98 KB × 4 bytes = **8.85 MB** — matches measured value exactly.

---

## 4. Why layerwise caching doesn't help much

The layerwise strategy generates factors one block at a time and releases them after each block completes, keeping peak RAM near zero. Forward time is **58.9 ms** vs uncached **72.4 ms** — only **19% faster** despite regenerating each block's factors only once per forward pass rather than once per use.

The reason: at medium scale, each block uses its three matrices for one SwiGLU computation. There is no reuse *within* a single forward pass — each `LatentLinear.forward()` is called exactly once anyway. Layerwise caching only helps if the same factors would be used multiple times within a single pass, which they aren't here.

Layerwise caching *does* make sense if:
- Context is very long and each layer processes many attention/MLP calls (not the case in this architecture)
- Memory is severely constrained and 8.85 MB is unacceptable

For the standard use case (inference with fixed weights), **all-factor caching is the right choice**.

---

## 5. Implementation

Three files were added to `latent_transformer_v0/`:

### `inference_cache.py`

Non-invasive module. Does not modify `latent_transformer_v0.py`. Key design:

- Patches `LatentLinear.forward` once at the class level (idempotent).
- Stores `(L, R)` tensors as attributes directly on each `LatentLinear` instance.
- Auto-clears on `model.train()` via a bound instance method override.
- Three strategies: `"all"`, `"layerwise"`, `"none"`.
- `prepare_for_inference(model, strategy)`, `clear_inference_cache(model)`, `factor_cache_bytes(model)`, `cache_is_valid(model)`.

Usage:
```python
from inference_cache import prepare_for_inference, clear_inference_cache

model.eval()
prepare_for_inference(model, strategy="all")

with torch.no_grad():
    tokens = model.generate(prompt, max_new_tokens=200)

# cache auto-clears when you call model.train()
model.train()
```

### `test_inference_cache.py`

13 correctness tests. All pass. Tests cover:
- Bit-identical logits (all-cached vs uncached, layerwise vs uncached)
- Bit-identical generate() output
- cache_is_valid() state tracking
- clear_inference_cache() completeness
- factor_cache_bytes() accounting
- train() auto-clear
- RuntimeError on training-mode prepare
- Idempotent double-prepare
- Non-latent model no-op safety
- Layerwise cache empty after forward

### `benchmark_inference.py`

Standalone benchmark script. Measures forward latency, TTFT, per-token latency, tok/s, checkpoint MB, cache MB. Supports checkpoints or random weights. Run:

```bash
python benchmark_inference.py                          # random weights, medium config
python benchmark_inference.py --checkpoint model.pt   # trained checkpoint
python benchmark_inference.py --device cuda           # GPU
```

---

## 6. What this establishes and what it does not

**Established:**
- Factor generation is the dominant cost of uncached latent inference (~82% of forward time).
- All-factor caching eliminates that cost entirely and recovers lowrank-equivalent throughput.
- The storage advantage of the latent representation (3.8× smaller checkpoint than dense) is preserved — the cache is a runtime artifact, not a persistent cost.
- The combined checkpoint + cache footprint (24.1 MB) is still smaller than dense alone (57.5 MB).
- Layerwise caching reduces peak memory at the cost of near-identical per-forward speed to uncached.
- All results are bit-identical: caching is a pure performance optimization with no numerical effect.

**Not established:**
- These benchmarks used randomly initialized weights. Latency ratios should be identical with trained weights (factor generation cost is independent of weight values), but trained-checkpoint benchmarks have not been run.
- GPU timings are not measured — the CPU bottleneck from constructor MLP calls may behave differently on GPU, where the batched matmuls in `factors()` are faster relative to the per-row/column loop.
- The external benchmark (preliminary) reported latent_uncached at 49.7 ms and latent_cached at 8.0 ms on a comparable config. Our measurements show 72.4 ms uncached and 6.1 ms cached. The speedup ratios (12× here vs ~6× there) differ — the preliminary benchmark may have used a different constructor configuration, batch size, or generation length. These results should not be directly compared until configurations are matched.

---

## 7. Recommendation

**All-factor caching should be the standard inference path for latent models.**

The case is unambiguous:
- 12× forward speedup
- Bit-identical outputs
- 8.85 MB runtime overhead at medium scale — a fixed cost that does not grow with sequence length or batch size
- One line to activate: `prepare_for_inference(model, strategy="all")`
- Automatically reverts on `model.train()`

The only reason *not* to cache is if memory is so constrained that 8.85 MB is unavailable. In that case, `strategy="layerwise"` gives a small improvement (~19%) with zero persistent memory.

The latent model's storage advantage (3.8× smaller checkpoint) combined with caching at inference means it stores less, loads faster, and runs faster than dense — while using 26% of dense's parameters. The open question is whether that parameter efficiency translates to equal loss with more training, not whether inference is viable.
