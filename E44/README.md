# E44 — Randomized-World Latent Communication Robustness

## Research question

Does the E43 Phase 1 communication effect survive independently randomized, balanced assignments between fact identities and value classes?

E43's ten-seed replication varied model initialization and training order but reused one fixed mapping (`value = fact_id % 64`). E44 removes that structural regularity. E43 remains unchanged.

## Design

- 2,048 fact identities and 64 value classes.
- Each dataset seed creates a new randomized balanced mapping.
- Training identities (0–1,535) and held-out identities (1,536–2,047) are disjoint.
- Exactly 24 training and 8 held-out examples per class.
- Dataset seed and model seed are independent and crossed.
- Default: 5 dataset seeds × 5 model seeds = 25 fresh-process runs.
- Frozen synthetic sender/receiver bases and trainable sender/receiver adapters.
- Same 16-value int8 packet plus one float32 scale (20 bytes/message before framing).
- Same STE training path and quantized/dequantized evaluation as E43.

## Controls and accounting

Learned, shuffled, random, zero, and no-message channels; held-out majority baseline; adapter gradient/update diagnostics; packet bytes; parameter counts; FP32 checkpoint-size estimates; training and evaluation timing.

## Run from repository root

```bash
python E44/E44_factorial_replication.py
```

Smaller smoke run:

```bash
python E44/E44_factorial_replication.py --dataset-seeds 101 102 --model-seeds 43 44 --epochs 10
```

One factorial cell:

```bash
python E44/E44_randomized_world.py --dataset-seed 101 --model-seed 43 --epochs 60
```

Outputs go under `runs/E44/` and are not committed automatically.

## Interpretation boundaries

E44 remains a synthetic 64-class task with synthetic frozen networks. It does not test pretrained language models, unseen value classes, general semantic communication, or compression advantage. The factorial design tests robustness to controlled randomized balanced assignments and model initialization; it is not a sample of 25 independent real-world datasets.

A positive result weakens the explanation that E43 depends on its modulo-based fact/class pattern. A collapse is equally informative and should stop progression to pretrained models until understood.

## Decision gate

- Consistent learned-channel superiority across dataset and model seeds: proceed to a harder task with receiver-side queries and richer private evidence.
- Strong dataset-seed sensitivity: inspect data generation and sender evidence pathways before scaling.
- Collapse toward controls: revisit the E43 task structure; do not proceed to pretrained models yet.

No compression claim until matched text/neural baselines and complete amortized system costs are measured.
