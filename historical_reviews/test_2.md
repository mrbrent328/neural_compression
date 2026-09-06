# Historical Review — Test 2

## Source
- Original script: `artifacts/test2.py`
- Historical position: follows the early lettered Tests C and D.

## Purpose
Test 2 shifts the central question from next-byte prediction toward learned representation of whole regions of a file.

The experiment asks whether a 1,024-byte block can be described through a much smaller learned latent state, while an exact modulo-256 residual preserves every detail the learned representation fails to reproduce.

Conceptually:

`raw block = learned structural approximation + exact correction information`

The compression question is whether the total reversible description — learned model/structure plus correction information and protocol overhead — can become cheaper than conventional compression.

## Design
The input defaults to `pic.png` and is split into fixed 1,024-byte blocks, with zero padding on the final block.

A neural encoder/decoder is trained directly on the blocks:

`1024 → 64 → 16-dimensional latent → 64 → 1024`

The decoder output is converted back into predicted byte values. An exact residual is then calculated for every byte:

`residual = (actual - prediction) mod 256`

This makes exact reconstruction mathematically possible through:

`original = (prediction + residual) mod 256`

The residual is compressed with zlib. The model state is serialized and zlib-compressed. Metadata records the original size/hash, block dimensions, model dimensions, parameter count, and formats. The intended package is:

`header + metadata + compressed model + compressed residual`

The experiment reports prediction MSE/MAE, maximum error, percentage of zero residual bytes, compressed residual size, residual Shannon entropy, total protocol size, and comparison against both the original and zlib(original).

## What progressed from Test D
Tests C and D primarily explore temporal/contextual prediction. Test 2 changes the decomposition:

- C: context → guessed byte → exact residual → zlib
- D: context → categorical probability distribution → arithmetic coder
- Test 2: whole block → compact latent representation → reconstructed block + exact correction

The important research move is from **learned prediction** toward **learned representation**.

Rather than asking only what comes next, Test 2 asks how much of a region's structure can be represented through a small latent description and how much irreducible correction information remains afterward.

The residual entropy calculation makes this especially explicit: after the neural representation explains what it can, how much information remains in the correction channel?

## Improved accounting
Unlike Test D's held-out continuation setup, Test 2 attempts to create and measure a complete packaged representation rather than assuming that most of the source file is already shared with the decoder.

The model is trained on the same object being represented, but this is meaningful in the intended formulation because the model itself is charged as part of the representation. This makes the experiment closer to **model-as-description** than conventional train/test machine learning.

## Important protocol limitations
### 1. Reconstruction does not exercise the stored package
The lossless reconstruction test uses `predicted_bytes` that remain in memory from the encoder run. It does not deserialize the `.neuralres` package, restore the model, regenerate all required representation state, decompress the residual, and reconstruct independently.

The test therefore proves the modulo-residual identity and exact reconstruction from in-memory predictions + residual, but not yet standalone decodability of the written package.

### 2. Block-specific latent states are not stored
Each input block produces its own 16-value latent vector. Those latent vectors are not included in the package.

A decoder possessing only the model cannot obtain the correct latent vector for an unknown block by running the encoder, because the encoder requires that original block as input. Therefore a standalone version of this architecture would need to transmit the block latents or introduce a mechanism that can regenerate them without the source data.

This is the most important conceptual gap in Test 2 and naturally raises a later Neural Compression question: **how can useful structural state be communicated or regenerated cheaply enough to function as compression?**

### 3. Package framing is incomplete
The package header stores the metadata length and then concatenates metadata, compressed model, and compressed residual. The compressed-model length is not recorded, so an independent decoder does not have an explicit boundary between model and residual payloads.

## Preserved-result status
The script is preserved. No authoritative historical run output has yet been associated with this review, so the original numerical outcome is not claimed here.

## Retrospective conclusion
**Historically important conceptual transition; incomplete standalone codec.**

Test 2 moves Neural Compression from learned next-symbol prediction toward the idea that an object's information may admit a cheaper learned structural representation than its literal byte arrangement, with an exact correction channel preserving losslessness.

The experiment also exposes the central representation problem that follows: the latent state that explains each block is itself information. A practical codec must either store that state cheaply or make it reconstructible from information already available to the decoder.