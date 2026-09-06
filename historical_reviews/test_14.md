# Test 14 — Model-as-Latent-State / Whole-Model Latent Container Branch

## Historical position
Test 14 follows the Test 13 latent-native computation branch. Test 13 asked whether a learned latent could become a computational substrate rather than only an intermediate compressed representation. Test 14 applies that idea to neural-network model weights themselves.

This review covers the preserved early Test 14 branch:

- `artifacts/test14A.py` / equivalent single-tensor latent-weight prototype;
- `ROCM-test/test14b.py` / whole-GGUF latent container prototype;
- `artifacts/test14_inference.py`;
- `artifacts/test14_inference_hashcheck.py`.

The repository also contains later root-level files named `experiment14_shared_constructor_latent_sweep*.py`. Those belong to a separate later numbering lineage and are not part of this historical Test 14.

## Core question
Can learned model weights themselves be treated as structured data, mapped into a compact latent representation, stored in an indexed latent container, reconstructed back into tensors, and then used directly by a neural-network computation?

The branch develops through four linked stages:

`single real model tensor -> tiled latent representation`

`whole model -> indexed latent container`

`latent container -> reconstructed tensor -> forward pass`

`reconstruction -> intended integrity/fidelity verification`

## 14A — real model tensor into latent space
The first stage moves beyond dummy image tensors and applies the convolutional latent representation to a real GGUF model-weight matrix.

A large two-dimensional weight tensor is converted to float32, divided into 256x256 tiles, and passed through a convolutional encoder. Each tile is reduced to a spatial latent with shape approximately `[16, 32, 32]`.

This establishes an architectural mapping:

`model weight matrix -> tiled spatial field -> latent tensor`

The raw float32 latent contains one quarter as many scalar elements as the 256x256 float32 tile because spatial area is reduced by 64x while channel count rises to 16.

### Important limitation
This is not yet evidence of compression of the original GGUF file. The comparison is against the expanded float32 tensor, while a GGUF may already store the source weight in a much more compact quantized form.

More importantly, the encoder is randomly initialized and is not trained to preserve the original tensor. A decoder exists in the architecture, but reconstruction fidelity is not measured. Thus 14A demonstrates representational plumbing, not preservation of model information.

## 14B — whole-model latent container
14B promotes the single-tensor experiment into a whole-model storage format.

It walks the GGUF tensors and divides them into two classes:

- large tensors are reshaped/tiled and passed through the latent encoder;
- small or one-dimensional tensors are copied through without latent transformation.

The resulting representations are concatenated into a binary `.latents` store. A JSON index records the information needed to locate and interpret each tensor, including original shape, padded shape, tile counts, latent shape, byte offset, and byte length.

Conceptually this is much more than a tensor-compression test. It sketches an alternate model container organized around latent state:

`GGUF -> tensor inventory -> latent/raw representation -> binary store + index`

### Important limitation
The latent encoder is still random/untrained. The converter does not establish that latent tensors retain the source information, nor does it provide a trained matching decoder or exact correction stream. Therefore the resulting latent container is not yet a faithful replacement for the GGUF model.

## `test14_inference.py` — reconstruction and executability
The inference follow-up reads the binary latent store and JSON index, seeks to a requested tensor by byte offset, loads its latent grid, passes it through a transpose-convolution decoder, reconstructs and stitches the tiles, crops padding, restores the original tensor shape, and injects the resulting weight/bias into a normal PyTorch `Linear` layer.

It then performs a forward pass.

This successfully extends the systems path to:

`latent container -> tensor-shaped reconstruction -> executable neural operation`

However, the decoder is freshly initialized and is not the matching decoder of a trained encoder/decoder pair. Thus a successful forward pass proves only that the reconstructed arrays are shape-compatible and numerically usable by PyTorch. It does not prove that the original tensor or model behavior has been preserved.

## `test14_inference_hashcheck.py` — intended fidelity validation
The hash-check follow-up is historically important because it recognizes that executability is not enough.

The script contains a `compare_tensors()` routine capable of reporting exact equality, element counts, maximum and mean reconstruction error, and SHA-256 equality between original and reconstructed tensors. This is the correct validation question for the branch:

`Does the latent reconstruction preserve the actual information, not merely the tensor shape?`

However, the preserved implementation never invokes that comparison routine. Its alternate hash path expects reference hashes in the 14B JSON index, but 14B did not store those hashes.

As a result, the experiment as preserved never obtains the intended original-versus-reconstruction fidelity result.

## Historical termination note
The missing validation was an implementation mistake that was not noticed at the time. The hash-check script appeared to contain the requested validation machinery, but the meaningful comparison function was dead code and the reference hashes required by the alternate path were absent.

The branch was therefore discontinued without answering its central fidelity question.

This matters for retrospective interpretation: Test 14 should not be classified as experimental evidence that latent model-weight storage failed. The branch was interrupted by an unnoticed validation implementation error before the decisive test was actually performed.

## What Test 14 genuinely establishes
The preserved branch demonstrates several architectural capabilities:

1. real neural-network weight tensors can be treated as structured spatial data and mapped into latent tensors;
2. a whole model can be reorganized into an indexed binary container containing latent and pass-through tensor representations;
3. latent entries can be addressed by offset, decoded into tensor-shaped arrays, stitched back into the expected shape, and injected into a PyTorch layer;
4. the experiment correctly evolved toward the realization that reconstruction fidelity must be tested independently from simple executability.

It does **not** establish that the latent representation preserves weight values, model function, or useful compression relative to the original quantized model.

## Retrospective interpretation
The most important conceptual transition is from treating the latent as a compressed representation of ordinary data to treating the model itself as latent-representable state.

Test 13:

`source representation -> latent -> downstream computation`

Test 14:

`model parameters -> latent-state container -> reconstructed computational parameters`

This is an early model-as-state / latent-native model-storage idea. Later Neural Compression work repeatedly returns to the broader premise that useful structure may be better represented as compact state plus relationships than as a literal copy of all source values.

## Closing assessment
**Architecturally significant, informationally unresolved.**

Test 14 builds the first preserved whole-model latent container and the plumbing required to reconstruct and use tensors from it. But the encoder/decoder pair is not trained for fidelity, and the intended hash/tensor comparison is never actually executed. The branch therefore ends with its central scientific question still open rather than with a negative result.

**Transition:** latent-native computation -> model weights as latent-representable state -> whole-model indexed latent container -> recognition that exact/fidelity validation is required before the representation can be considered a valid model codec.