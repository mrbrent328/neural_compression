# Historical Review — Test 9

## Source
- `artifacts/test9.py`

## Purpose
Test 9 is primarily an engineering/performance iteration of Test 8 rather than a new compression hypothesis. It preserves the real-corpus, independent train/held-out, exact-reconstruction design while making neural training and inference practical on AMD ROCm hardware.

## Experimental rules retained from Test 8
The script explicitly preserves:

- first 75% training / final 25% held out;
- no held-out neural training;
- a separate neural model for each scaling point;
- exact reconstruction and SHA-256 verification;
- previous-byte, 4-byte context, neural, and hybrid methods;
- residual zlib compression;
- model + residual + metadata accounting;
- real source-code/text corpus construction with file boundaries.

## New contribution: ROCm/GPU execution
The major change is operational scaling.

The script is designed for AMD ROCm PyTorch using the normal PyTorch `cuda` interface. It documents the expected `HSA_OVERRIDE_GFX_VERSION=10.3.0` launch environment and deliberately keeps fp32 for stability/reproducibility.

Training samples are created on CPU, moved to GPU in bounded batches, and the neural model remains resident on GPU. Neural held-out prediction and hybrid validation/prediction are batched, with explicit GPU synchronization around timed operations.

The script also measures GPU memory usage and reports neural inference throughput.

## Causal decoding remains sequential
A particularly important architectural point is stated directly in the code: exact predictive decoding is kept sequential because each next prediction depends on bytes already reconstructed.

This distinguishes parallelizable work:

`training / teacher-forced prediction-side evaluation`

from inherently causal work:

`decoder-known history → next prediction → residual correction → next history`.

That constraint later matters for practical neural-codec throughput even when training and encoding can exploit GPU batching.

## Hybrid selector
Test 9 retains Test 8's corrected hybrid design. The selector is learned from the final 10% of the training region, recording context-specific success of the context and neural predictors. It does not inspect the held-out target byte to choose between predictors.

Test 9 mainly accelerates this validation and prediction process with GPU batching.

## Remaining accounting issue
Test 9 does not resolve the central accounting problem inherited from Test 8.

Predictive totals still describe:

`zlib(held-out 25% residual) + model cost + metadata`

while the comparison baseline is full-corpus zlib.

Therefore any whole-file `beats zlib` interpretation remains invalid unless the first 75% prefix/training information is also charged or an equivalent shared-state comparison is used.

The same model-storage caveats also remain: neural model size is based on raw tensor storage, context-model storage is estimated, and hybrid selector-state cost must be represented for a true standalone codec.

## Preserved evidence
The preserved source establishes the intended ROCm execution strategy and experimental design. No authoritative numerical Test 9 output is asserted here unless separately recovered from preserved result files.

## Retrospective interpretation
Test 9 represents the transition from experimental validity toward operational scalability.

The key question is no longer primarily "does the idea work at all?" but:

**can the corrected experiment be run at useful corpus sizes and GPU throughput without violating the causal decoding requirement?**

This is historically useful because it begins separating algorithmic compression quality from systems constraints such as training time, inference throughput, GPU memory, and sequential decoder latency.

## Historical transition

`real-corpus, leakage-free predictive experiment`

→

`GPU-accelerated training and encoding-side evaluation`

while preserving:

`sequential causal exact decoding`.

## Closing label
**Operational scaling iteration: preserves Test 8's corrected methodology while making neural training/inference practical on AMD ROCm hardware; causal decoding remains deliberately sequential.**