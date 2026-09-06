# Historical Review — Test 10

## Source
- `artifacts/test10.py`

## Purpose
Test 10 is a major architectural step after the Test 8/9 real-corpus scaling line. Instead of reducing the neural model to one argmax byte and then storing a residual, it uses the model's full 256-way probability distribution directly with a deterministic range/arithmetic coder.

Conceptually:

`context -> best-byte prediction -> residual -> zlib`

becomes

`context -> probability distribution -> deterministic integer frequencies -> range-coded bitstream`

## What changed from Test 9
- The primary method is now direct neural probability coding rather than residual coding.
- The encoder and decoder maintain the same strictly causal reconstruction state.
- Probabilities are quantized deterministically to integer frequencies; floating-point probability tables are not stored.
- The actual coded payload is compared with theoretical cross-entropy.
- A probability-level neural/context mixture is tested instead of a hard predictor selector.
- The Test 9 neural residual method is retained as a reference control.

## Causality
The encoder is explicitly forbidden from using future held-out bytes as context. It starts from the training prefix and maintains a simulated reconstruction buffer. After encoding each known actual symbol, it appends that symbol to the buffer.

The decoder starts from the same training prefix, derives the same context from bytes already decoded, regenerates the same neural probabilities and deterministic integer frequency table, decodes one symbol, appends it, and continues.

This is a genuine causal continuation codec mechanism over the held-out region.

## Probability quantization and coder
The script converts the 256 floating-point probabilities to integer frequencies summing exactly to a fixed total frequency. Every symbol receives a positive minimum frequency. The encoder and decoder invoke the same conversion, so no probability table needs to be transmitted separately.

The implementation includes matching 32-bit range encoder/decoder logic and exact sequential decode verification.

## Neural probability method
For every held-out byte:
1. derive context only from the reconstructed prefix/history;
2. generate the neural 256-way distribution;
3. accumulate `-log2(p(actual))` for theoretical cross-entropy;
4. deterministically quantize probabilities;
5. range-code the actual symbol;
6. append the actual symbol to encoder-side reconstructed state.

The decoder reproduces the same distribution and frequency table from already decoded state.

The experiment therefore measures both:
- theoretical model cross-entropy in bits/byte;
- actual range-coded payload size.

That is a useful entropy-coder sanity check.

## Probability mixture
Test 10 also introduces a neural/context mixture in probability space:

`P = 0.75 * P_neural + 0.25 * P_context`

This is materially different from choosing one predictor's hard byte. The models contribute uncertainty distributions directly, allowing the entropy coder to exploit the combined probability mass.

Historically, this is an early step toward combining complementary predictive states rather than selecting a single winner.

## Exactness
The neural range-coded and mixed range-coded methods are decoded sequentially and compared against the held-out region. SHA-256 verification is required.

The mechanism is therefore substantially stronger than a teacher-forced prediction benchmark.

## Remaining accounting problem
The major whole-file accounting mismatch survives.

The learned methods report:

`held-out 25% payload + model + metadata (+ context-model estimate for mixture)`

while the baseline is:

`zlib(full 100% corpus)`

The training prefix is still supplied to the decoder but not charged to the learned representation. Therefore the script's whole-file `ratio_vs_zlib`, `improvement_percent`, and `*** BEATS ZLIB ***` output do not establish a standalone whole-file compression win.

A fair continuation comparison would compare both methods on the same held-out domain under identical shared-prefix assumptions. A fair standalone codec would need to include or otherwise account for the prefix/training information.

## Other accounting caveats
- Neural model size is raw parameter/tensor storage rather than a fully serialized compressed model package.
- Context-model size is estimated rather than measured from a standalone serialized representation.

## Preserved evidence
The current repository preserves the implementation. No authoritative `test10_results.json` / `.csv` run output was identified during this retrospective, so no historical numerical performance claim is made here.

## Retrospective interpretation
Test 10 reconnects the later real-corpus scaling work with the earlier Test D/Test 5 insight: **do not throw away the model's uncertainty by collapsing it to one guessed byte.**

The matured form is:

`real corpus + independent train/held-out split + exact causal state + full learned distribution + entropy coder`

This makes Test 10 one of the more important early architectural steps even though whole-file accounting is still unresolved.

## Historical transition

`operational neural prediction at scale`

->

`directly encode learned uncertainty`

and

`hard predictor combination`

->

`probability-space mixture`

## Closing label
**Mature causal probability coding: replaces hard prediction + residual with direct neural uncertainty + deterministic range coding, while preserving exact sequential decoder state; whole-file accounting still unresolved.**