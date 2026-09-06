# Historical Review — Test C

## Source
- Original script: `artifacts/testc.py`
- Historical position: early lettered exploratory test, before numbered Test 2. The original work began with lettered tests because the project was not initially expected to require a long experiment series.

## Purpose
Test C asked whether a small learned predictor could capture enough reusable structure in text that its prediction residual would compress well enough to justify the cost of transmitting the learned model.

The important shift was from prediction accuracy by itself to **total representation cost**: predictor/model cost + residual cost + metadata.

## Design
Test C compares several approaches on raw text bytes:

1. Raw zlib baseline.
2. Previous-byte prediction, with `(actual - prediction) mod 256` stored as an exact residual and compressed with zlib.
3. An adaptive order-3 context predictor rebuilt causally from previously seen bytes, followed by exact residual + zlib.
4. A neural predictor using the previous 8 bytes, two 64-unit hidden layers, scalar normalized-byte regression, exact residual + zlib, plus compressed model weights.

The neural path is explicitly charged for its model. The adaptive context model has an important structural advantage because its state can be rebuilt from decoded history and therefore does not require a separately transmitted model payload.

## What the experiment established
The strongest conceptual result is that a learned predictor is useful for compression only when the structural savings it creates exceed the cost of communicating the predictor itself.

This is an early form of a principle that remains central later in Neural Compression:

> Better prediction is not sufficient. The useful quantity is the cost of the complete reversible representation.

Test C therefore moved the work from a generic machine-learning prediction question toward a codec question: can learned structure participate economically in a complete lossless representation?

## Limitations
The neural model treats byte values as a scalar regression problem and uses MSE, even though bytes are categorical symbols. Numeric closeness between byte values is not itself probabilistic closeness.

The order-3 comparison is also deliberately crude: it effectively remembers a recent continuation rather than maintaining a full probability/frequency distribution like a mature statistical compressor.

Accordingly, Test C should not be interpreted as establishing the performance ceiling of neural versus classical compression.

## Preserved-result status
The script is preserved, but no authoritative historical console output/result for Test C has yet been located. We therefore do not claim which method numerically won the original run.

## Retrospective conclusion
**Promising concept; incomplete codec formulation; directly motivates Test D.**

Test C established the accounting problem clearly: learned structure can reduce residual uncertainty, but the learner itself has a representation cost. The natural next step was to stop predicting a single byte value and instead make the network model the full probability distribution needed by an entropy coder.