# Neural Compression Archive — NCA-01 / NCA-02 Results

**Date:** 2026-08-31  
**Branch:** `main`

## Purpose

Record the paired archive experiments performed against the same `largecalgarycorpus`. NCA-01 establishes the baseline archive configuration; NCA-02 tests the stronger neural configuration carried forward from Series 4.

## Common test corpus

- Input: `/home/xetal/Desktop/largecalgarycorpus`
- Original size: **3,179,847 bytes**
- Strategy: `adaptive`
- Block behavior: adaptive local selection among store/classical/neural

## NCA-01 — Baseline

### Configuration

NCA-01 uses the earlier neural compressor configuration:

- embedding: 32
- hidden: 64
- context: 16 bytes
- neural states: 4
- optimizer: SGD
- learning rate: 0.04
- seed: 42

### Result

- payload bytes: **1,146,715**
- archive bytes: **1,172,168**
- reduction: **63.1376%**
- neural blocks: **48**
- classical blocks: **8**
- store blocks: **0**
- elapsed: **2,532.5846 s** (~42m 12.6s)
- SHA-256 manifest: `4c83e773af71310efd20af445df17f96d73b0829930c05fb58c63badc63d348f`

The NCA-01 archive was subsequently verified through the planned exact round-trip test; the reconstruction matched the original byte-for-byte.

### Interpretation

NCA-01 demonstrated that the adaptive archive architecture can produce a substantial lossless reduction on the Calgary corpus. The archive selected neural coding for most blocks while retaining classical coding for the remainder. This is an ensemble/adaptive compressor rather than a claim that the neural method is universally best.

## NCA-02 — Strong Neural Configuration

### Hypothesis

A stronger neural configuration identified during Series 4 would improve the compression achieved by the same adaptive archive architecture.

### Configuration

- embedding: **48**
- hidden: **300**
- context: **10 bytes**
- neural states: **3**
- optimizer: SGD
- learning rate: 0.04
- seed: 42
- state selection: deterministic minimum predictive entropy

All other archive machinery was intended to remain unchanged from NCA-01.

### Result

- payload bytes: **1,297,872**
- archive bytes: **1,323,325**
- reduction: **58.3840%**
- neural blocks: **48**
- classical blocks: **8**
- store blocks: **0**
- elapsed: **2,351.8005 s** (~39m 11.8s)
- SHA-256 manifest: `1f3c2c498091020804248d6136cde48057b1d356f343c1452df8d252592fd038`

The compression run completed successfully. Exact NCA-02 round-trip verification remains the required next validation step before final experimental closeout.

### Interpretation

NCA-02 **did not support the hypothesis**. Its archive was **151,157 bytes larger** than NCA-01, despite using the stronger Series-4-derived neural configuration.

The adaptive block allocation was identical to NCA-01 (48 neural, 8 classical, 0 store). Therefore the observed difference is not explained by a change in which blocks were assigned to neural versus classical coding. It points instead toward differences in the neural predictor/state dynamics and their resulting coded payloads.

NCA-02 was also approximately **180.8 seconds faster** than NCA-01. Thus increased model capacity did not produce a runtime penalty in this run and instead coincided with faster completion, while compression quality decreased.

## Paired comparison

| Metric | NCA-01 | NCA-02 |
|---|---:|---:|
| Original | 3,179,847 B | 3,179,847 B |
| Archive | **1,172,168 B** | **1,323,325 B** |
| Reduction | **63.1376%** | **58.3840%** |
| Neural blocks | 48 | 48 |
| Classical blocks | 8 | 8 |
| Store blocks | 0 | 0 |
| Runtime | 42m 12.6s | **39m 11.8s** |

NCA-02 archive size is **12.90% larger than NCA-01** (equivalently, NCA-01 is about 11.43% smaller than NCA-02).

## Scientific conclusion

The paired experiments establish a useful negative result:

> **A neural configuration that performed better in the Series 4 experimental regime does not automatically produce a better sequential lossless archive predictor.**

This separates at least two optimization questions that should not be conflated:

1. performance in the original Series 4 task;
2. usefulness as a probability model for byte-level lossless coding.

The identical adaptive block allocation in NCA-01 and NCA-02 further suggests that the next investigation should examine neural prediction/state behavior inside the selected blocks rather than immediately changing the archive selector.

## Status

- NCA-01 compression: complete
- NCA-01 exact round trip: verified
- NCA-02 compression: complete
- NCA-02 exact round trip: pending
- NCA-03: **not yet designed**

Do not treat the NCA-02 compression result as a final closeout until its exact reconstruction test passes.
