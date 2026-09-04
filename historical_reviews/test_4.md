# Historical Review — Test 4

## Source
- Original script: `artifacts/test4.py`
- Historical position: follows Test 3.
- Script title: `LEARNED CONTEXT-WIDTH STORAGE EXPERIMENT`

## Purpose
Test 4 changes the optimization target from simply improving prediction toward finding the most economical amount of learned context.

It asks how much contextual information is worth giving a neural predictor once the storage cost of the larger model is charged against any reduction in residual uncertainty.

The central tradeoff is:

`more context/capacity → potentially cheaper residual`

versus

`more context/capacity → larger model payload`

The desired model is therefore the one with the smallest complete measured learned representation, not necessarily the one with the lowest prediction loss.

## Design
The experiment accepts a real input file and tests neural next-byte predictors with context widths of 4, 8, 16, and 32 bytes by default.

Each byte is represented by a learned 16-dimensional embedding. The embedded context is flattened and passed through two 64-unit hidden layers to a 256-symbol categorical output. Training uses cross-entropy.

The first 75% of the file supplies training positions. The final 25% is used as the held-out prediction region.

For each context width the experiment records:

- held-out cross-entropy loss;
- prediction MSE/MAE and maximum error;
- exact-hit/zero-residual percentage;
- residual Shannon entropy;
- zlib-compressed exact modulo-256 residual size;
- raw and zlib-compressed model size;
- a 1,024-byte metadata allowance;
- total learned representation size.

Exact reconstruction of the held-out region is checked by adding prediction and residual modulo 256.

## Capacity-allocation significance
Test 4 is an early explicit capacity-allocation experiment.

Increasing context width increases the amount of structural evidence available to the predictor, but also increases parameter count and model storage. The final selection uses minimum total bytes rather than minimum loss.

This establishes an important compression principle that recurs later in Neural Compression: **representation capacity should be allocated according to total information cost, not model quality in isolation.**

## Progression from Test 3
Test 3 established causal, decoder-reproducible context as an alternative to source-dependent local latent vectors and broadened testing across structural classes.

Test 4 focuses that architecture on a new question:

`How much decoder-reconstructible context is economically useful?`

It therefore moves from proving the general learned-context mechanism toward optimizing the amount of representational capacity devoted to that mechanism.

## Remaining accounting limitation
Test 4 does not yet resolve the coding-domain mismatch identified in Test 3.

The learned total consists of:

`compressed model + compressed residual for final 25% + metadata`

but this is compared against both the full original file and zlib of the full original file.

The first 75% used to train the source-specific model is not represented in the learned total. Therefore a printed claim that the learned protocol beats the original or zlib would not, by itself, demonstrate a superior standalone whole-file codec.

A fair held-out/shared-model comparison would compare the learned representation against compression of the same held-out region. A standalone whole-file comparison would need to account for how the training portion/source-specific model becomes available to the decoder.

## Decoder/protocol completeness
Predictions for the held-out region are generated using contexts sliced directly from the original file. Because the architecture is causal, those contexts should in principle be reproducible sequentially from previously decoded bytes, unlike Test 2's inaccessible block latents. However, Test 4 does not implement a separate sequential decoder that restores the model and reconstructs the stream from a packaged representation.

The 1,024-byte metadata value is also an allowance rather than a concrete serialized file format, and raw model tensors are concatenated without complete tensor framing needed by an independent decoder.

Thus exact residual arithmetic is demonstrated, while a complete independently decoded codec artifact remains unfinished.

## Preserved-result status
The script is preserved. Repository search has not located an authoritative saved Test 4 run output, so the historically winning context width and numerical performance are not claimed here.

## Retrospective conclusion
**Accounting-aware capacity optimization; whole-file accounting and independent codec packaging still unresolved.**

Test 4's major historical contribution is the explicit recognition that more structural modeling is not automatically better compression. The correct amount of model/context capacity is the amount that minimizes the combined cost of model plus unexplained information.

Historical transition:

`decoder-reconstructible learned context works`

→

`optimize how much learned context is worth paying for`

This is an early precursor to later Neural Compression work on adaptive representation capacity, local structural complexity, and strategy selection.