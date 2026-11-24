# E42 — Scaled Modular Concept Network

## Question

At larger concept counts, does a modular architecture with persistent local concept states, shared local processing, and query-conditioned relational routing improve held-out relational prediction over shared-embedding controls?

This is the first scaled test of the network-of-networks direction. It builds on the earlier persistent-relational work rather than treating communication compression as an unsolved prerequisite.

## Models

1. **Dense baseline** — shared node and relation embeddings with a common query projection.
2. **Modular** — persistent concept state vectors, a shared relation-aware local message processor, and a query-conditioned coordinator that selectively aggregates observed neighbor messages.
3. **Modular without messages** — same modular parameterization, but neighbor messages are disabled. This isolates the contribution of inter-concept communication.

The modular prototype uses shared processing weights over independently learned concept states. It is not yet a separate independently parameterized neural network per concept; that more expensive variant should only follow if this shared-constructor version shows a useful signal.

## Scales and defaults

- 8,192 / 32,768 / 65,536 concepts
- 16 relation types
- 8 observed edges per concept on average
- 64-dimensional concept states
- 12 retained context neighbors per concept
- 3 seeds per scale
- 3,000 training steps per model/run
- 16 training negatives and 64 evaluation negatives

The defaults are intended as a substantial overnight workload, not a promise of a particular runtime. Reduce --steps, --scales, or --seeds for a smoke test.

## Run

Requires Python 3.10+, NumPy, and PyTorch.

```bash
python -m pip install numpy torch
python E42_scaled_modular_concept_network.py
```

For an explicit ROCm/CUDA device:

```bash
python E42_scaled_modular_concept_network.py --device cuda
```

For a shorter smoke test:

```bash
python E42_scaled_modular_concept_network.py --scales 8192 --seeds 11 --steps 100 --max-eval-queries 1000
```

Outputs are written under results/e42/:
- e42_results.json — incrementally updated manifest and metrics
- model checkpoints for each scale, seed, and condition

The result file is updated after each completed run, so completed runs survive a later interruption. A checkpoint is written after each model finishes.

## Measurements

- held-out link-prediction MRR
- Hits@1 and Hits@10
- mean rank
- sampled training loss
- parameter count
- training wall-clock time
- checkpoint size (available from saved files)

## Controls and caveats

- The graph generator uses hidden family IDs to create regularities, but those IDs are never passed to the models.
- The context graph is built from training edges only; held-out edges are not available to the router.
- All three models use the same generated train/test edges and candidate evaluation protocol.
- Uniform negative sampling makes the task tractable but does not represent all forms of semantic reasoning.
- The current modular model uses a shared local processor over persistent concept states. A positive result would motivate testing dynamically instantiated or independently parameterized concept subnetworks.
- This experiment does not test natural-language modeling, consciousness, or general intelligence.
- Compare actual wall-clock time and memory alongside accuracy. Fewer messages or parameters alone do not establish a speed advantage.

## Interpretation rule

A modular win is meaningful only if it consistently improves held-out metrics across seeds and scales without an unacceptable parameter, memory, or runtime penalty. If modular and no-message results are similar, the evidence does not support the claim that inter-concept messaging helped. If the dense baseline wins, record that as a useful negative result rather than tuning only the modular model.

## Status

**Proposed / ready for execution.** No results are claimed until the experiment has been run and audited.
