# Context Handoff — GSA-01 Generation Sampling Audit

**Date:** 2026-11-17  
**Project:** Neural Compression  
**Canonical branch:** main  
**Status:** Implementation committed; awaiting local test and experiment run.

## Current research position

The current language-model question is whether the repetitive/collapsed outputs observed in greedy decoding reflect a decoding artifact or a genuinely limited learned next-byte distribution.

This experiment deliberately does not retrain any model. It reuses the existing dense, low-rank, and latent/shared-constructor checkpoints and the prompt set from the earlier generation audit.

## Implemented artifact

- Script: generation_sampling_audit.py
- Tests: test_generation_sampling_audit.py
- Instructions: GENERATION_SAMPLING_AUDIT_README.md
- Research log: GSA-01 section in RESEARCH_LOG.md
- Output directory: runs/generation_audit/

The audit compares:
- Greedy decoding, one deterministic baseline.
- Temperature 0.7, 0.9, and 1.0.
- Top-p 0.90 and 0.95.
- Three deterministic seeds for each sampling strategy, prompt, and model.

It records diversity, repeated trigrams, whitespace/printability, UTF-8 validity, raw and filtered entropy, top-1 probabilities, and byte-level agreement across sampling seeds.

The custom greedy decoder is checked against the model's existing generation method. The latent model's all-factor inference cache is enabled only after a bit-identical logits check. No autoregressive KV cache is included.

## Run locally

First run the focused tests:

~~~bash
python test_generation_sampling_audit.py
~~~

Then run the complete audit on the AMD ROCm setup:

~~~bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python generation_sampling_audit.py --device auto
~~~

The default configuration produces 432 generation cases when all nine prompts are available. It generates 128 bytes per case. Existing checkpoints and previous reports are not overwritten.

For a quick smoke run:

~~~bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python generation_sampling_audit.py --device auto --seeds 1 --new-bytes 32 --heldout-prompts 0
~~~

## Validation status

The GitHub files and commit history were re-fetched after publishing to verify that the implementation, tests, README, and research-log entry are present on main.

The focused Python tests and full model audit have **not** been executed in the user's local ROCm environment yet. Do not describe the experiment as run or its results as measured until the user supplies the resulting output.

## Interpretation guardrails

- More diverse output is not automatically better output.
- Compare raw model entropy separately from temperature/top-p-filtered sampling entropy.
- Use UTF-8 validity, repeated n-grams, diversity, and human inspection together.
- The corpus is source-code-heavy; do not generalize this small prompt audit into a broad language benchmark.
- Keep the three checkpoints and the original greedy audit as immutable baselines.

## Next decision

After the user runs the tests and audit, inspect the timestamped JSON and TXT reports, compare each model across decoding strategies, classify the evidence, and decide whether the next bottleneck is decoding, corpus/data quality, or representation capacity.
