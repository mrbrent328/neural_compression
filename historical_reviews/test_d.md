# Historical Review — Test D

## Source
- Original script: `artifacts/testd.py`
- Historical position: follows the early lettered Test C and precedes numbered Test 2.

## Purpose
Test D turns the learned-prediction idea from Test C into an actual probabilistic lossless coding experiment.

Instead of predicting one numerical byte and compressing the error, the neural model predicts a probability distribution across all 256 possible next-byte symbols. Those probabilities directly drive an arithmetic coder.

Conceptual progression:

`context → categorical probability distribution → arithmetic coder → exact bitstream`

## Design
Configuration includes an 8-byte context, 16-dimensional byte embeddings, two 64-unit hidden layers, and a 256-logit output. Training uses cross-entropy rather than scalar MSE.

The experiment implements its own 32-bit arithmetic encoder and decoder and converts floating neural probabilities to deterministic integer cumulative frequencies.

The neural model trains on the first 75% of the file and evaluates coding on the held-out final 25%. During neural decompression, newly decoded bytes are fed back into the context, creating a genuinely causal sequential decode loop after initialization.

The script also measures raw and zlib-compressed model size, assumes 1,024 bytes of metadata, reports cross-entropy and actual arithmetic payload size, and verifies reconstruction using both byte equality and SHA-256.

## What progressed from Test C
Test C used:

`context → guessed byte → exact residual → zlib`

Test D instead uses:

`context → P(0..255) → arithmetic coding`

This removes the residual/zlib indirection and lets learned uncertainty itself determine the number of bits assigned to each symbol.

D also adds several methodological features that recur throughout later Neural Compression work:

- categorical probability modeling;
- explicit entropy coding;
- held-out evaluation;
- causal reconstruction;
- exact lossless verification;
- separate accounting for model payload and coded payload.

## Important limitations / accounting issues
The neural experiment assumes the first 75% of the original file is already available to both encoder and decoder. The decoder begins by copying that original prefix and only reconstructs the held-out continuation from the arithmetic stream.

Therefore `neural payload + compressed model + metadata` is **not a complete standalone representation of the entire original file**, even though final ratios are displayed relative to the whole original file size. The training/shared prefix is not charged.

The order-3 control also builds its frequency model from the complete data before encoding it, and the cost of communicating that static table is not represented. Its coding domain is also different from the neural held-out-only domain. The final order-3 and neural byte totals therefore are not clean apples-to-apples compressor benchmarks.

These issues limit performance claims, but they do not invalidate D's central mechanism test.

## Preserved-result status
The script is preserved, but an authoritative historical console output/result has not yet been located. Numerical win/loss claims from the original run should therefore remain unresolved unless the output is recovered.

## Retrospective conclusion
**Architecture/mechanism: strong validation. Compression competitiveness: unresolved from preserved evidence. Historical importance: high.**

Test D demonstrates that a learned contextual state can produce categorical uncertainty that is converted directly into a lossless arithmetic-coded stream and decoded causally and exactly.

It also exposes the next major problem: obtaining learned structural advantage without relying on shared source data or paying excessive model/state transmission cost.