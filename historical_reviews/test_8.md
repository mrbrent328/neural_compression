# Historical Review — Test 8

## Source
- `artifacts/test8.py`

## Purpose
Test 8 is a methodological repair and real-data extension of Test 7's scaling/amortization question. It asks whether predictive compression can become advantageous with increasing corpus size when tested on real source-code/text data rather than generated corpora.

## What changed from Test 7
Test 7's scaling curve had three major problems: smaller held-out regions leaked into a model trained on the largest dataset, the hybrid selector used the target byte as an oracle, and learned totals covered only the held-out region while the zlib comparison covered the full file.

Test 8 explicitly fixes the first two problems:

1. Every scaling point trains its own neural model using only the first 75% of that point's corpus.
2. The final 25% is genuinely excluded from neural/context training.
3. The hybrid selector is learned from training-region behavior rather than choosing a predictor after observing the held-out target byte.
4. Data comes from real source-code/text files under `/home/xetal/Downloads`.
5. File boundaries are tracked so training and prediction do not blindly treat concatenated files as one uninterrupted source.
6. Reconstruction is performed sequentially from reconstructed history and checked exactly.

## Data and scaling
The script discovers textual source files across many programming/configuration/document formats while excluding common build/cache/version-control directories and files over 5 MB.

Target corpus sizes range from 100 KB through 50 MB when enough source material is available. Each scaling point constructs its own corpus prefix and trains its own predictor on the first 75%.

## Predictors
The experiment compares:

- zlib baseline;
- previous-byte predictor;
- 4-byte context predictor;
- neural predictor;
- hybrid context + neural predictor.

The neural model uses a 4-byte causal context, 32-dimensional byte embeddings, a 128-unit hidden layer, and a 256-class output. Training uses up to 300,000 samples for 25 epochs.

Each predictive method generates an exact modulo-256 residual. The residual is compressed with zlib, and model + residual + fixed metadata cost are reported.

## Hybrid repair
Unlike Test 7's oracle hybrid, Test 8 builds a selector using the final 10% of the training region. For each observed context it records whether the context or neural predictor was historically correct. Held-out prediction then chooses the neural predictor only where its stored context-specific score exceeds the context predictor's score.

The same rule is reproduced during sequential reconstruction, so held-out target bytes are not consulted to choose the predictor.

This is a substantial methodological correction.

## Decoder realism and exactness
Test 8 includes sequential reconstruction for previous-byte, context, neural, and hybrid methods. Prediction contexts are taken from already reconstructed bytes rather than requiring unreconstructed held-out source bytes.

The reconstructed held-out region is checked both by direct byte equality and SHA-256. A mismatch raises an error.

This makes the predictor/residual mechanism much closer to an actual causal decoder than several earlier residual benchmarks.

## Remaining accounting problem
The major whole-file comparison problem remains unresolved.

For predictive methods, the reported total is:

`zlib(held-out 25% residual) + model cost + 1024-byte metadata`

but the baseline is:

`zlib(full 100% corpus)`

The first 75% of the corpus is not included in the learned representation total even though it is supplied as the reconstruction prefix. Therefore `ratio_vs_zlib`, `improvement_percent`, and `*** BEATS ZLIB ***` are not valid standalone whole-file compression comparisons.

A fair standalone representation would need to charge the prefix/training information or otherwise make the model/state independently available. A fair continuation-only comparison would instead compare against zlib of the same held-out continuation under equivalent shared-prefix assumptions.

## Other accounting caveats
- Neural model size is raw tensor storage rather than an actual serialized/compressed model package.
- Context-model size is an estimate rather than serialized storage.
- The hybrid total charges neural + context model sizes but does not separately account for storage of the `context_correct` and `neural_correct` selector tables required by the decoder.

## Preserved evidence
The current repository preserves the Test 8 implementation. No authoritative `test8_results.json` or `test8_results.csv` run output was identified during this retrospective, so no historical numerical performance claim is made here.

## Retrospective interpretation
Test 8 is best understood as a strong methodological recovery after Test 7 rather than a new conceptual branch.

The underlying hypothesis remains:

**fixed structural/model cost may become economical when amortized across enough information.**

Test 8 improves the experiment by moving to real corpora, eliminating cross-scale training leakage, replacing the oracle hybrid with a training-derived selector, respecting file boundaries, and exercising causal exact reconstruction.

It still does not establish a valid break-even point against whole-file zlib because the compared storage domains remain unequal.

## Historical transition

`model amortization hypothesis on synthetic data`

→

`real source-code corpus + genuinely held-out training + decoder-valid predictor selection`

while retaining the unresolved requirement:

`compare complete representations over equivalent coding domains`

## Closing label
**Real-corpus repair of the scaling experiment: removes training leakage and oracle hybrid selection, adds file-aware causal reconstruction, but still compares a 25%-domain learned representation against full-file zlib.**