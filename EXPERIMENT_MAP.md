# Neural Compression — Experiment Map

Purpose: a chronological, evidence-grounded index of experiments in this repository. Each entry separates what was attempted from what the surviving evidence establishes. Missing outputs are marked explicitly; no numerical result is inferred from code alone.

## Review protocol

For each experiment:
- **Question:** what it was trying to find out
- **Method:** what was actually tested
- **Evidence:** preserved measurements or artifacts
- **Conclusion:** what the evidence supports
- **Limit:** what it does not establish
- **Status:** SHOWN / SUPPORTED / OPEN / FAILED / RETIRED / INCOMPLETE

This map is a working index, not a replacement for the original scripts, result files, or detailed historical reviews.

---

## Test C — Learned prediction with complete representation cost

- **Source:** `artifacts/testc.py`
- **Detailed review:** [historical_reviews/test_c.md](historical_reviews/test_c.md)
- **Question:** Can a learned predictor reduce residual size enough to pay for the cost of storing the predictor itself?
- **Method:** Compared raw zlib, previous-byte prediction, a causal order-3 context predictor, and a neural predictor using the previous eight bytes. Prediction errors were retained as exact residuals; the neural model's compressed weights were included in its cost.
- **Evidence:** The script and retrospective review survive. No authoritative original console output or numerical result was located.
- **Conclusion:** The experiment framed the central accounting rule: prediction quality alone is not compression. The model, residual, and metadata must be counted together. Causal context state has a special advantage when the decoder can rebuild it from already-decoded history.
- **Limit:** It does not establish which method numerically won. The neural predictor treated byte values as scalar regression targets rather than categorical probabilities, and the order-3 baseline was crude.
- **Status:** SUPPORTED as a research framing / OPEN for historical numerical outcome.
- **Next transition:** Test D replaces single-value prediction with a probability distribution suitable for arithmetic coding.
