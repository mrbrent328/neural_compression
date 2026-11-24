# Latent Transformer v1 — Structural Atom Constructor

## Purpose

This is an isolated experimental variant of latent_transformer_v0. It tests whether a shared learned dictionary of structural atoms can generate useful layer-specific low-rank MLP structure from tiny per-matrix latents.

The dense, direct-low-rank, and v0 coordinate-constructor implementations remain unchanged. The v1 runner reuses the v0 language model, tokenizer, data split, optimizer, evaluation loop, attention, residual stream, and normalization. Only the latent MLP constructor is replaced.

## Representation

For each matrix-specific latent z:

1. A shared selector maps z to scores over a global atom dictionary.
2. Top-k selection chooses a sparse subset of atoms.
3. A shared coordinate-conditioned generator turns each selected atom, matrix-type embedding, and coordinate into factor components.
4. The selected components are weighted and composed into L and R.
5. The MLP executes as (x @ R.T) @ L.T; no dense matrix is stored.

The dictionary and generator are shared across all layers and all three MLP matrix types. The per-matrix persistent state remains a tiny latent plus scalar gain.

## Hypothesis

A shared structural vocabulary may reduce the burden on the constructor: each latent specifies a sparse composition of reusable components instead of asking one coordinate constructor to synthesize every factor independently.

This is a hypothesis, not an established advantage. The v1 generator still uses coordinate features; the experimental change is the learned atom dictionary and sparse composition.

## Run

From latent_transformer_v0/:

```bash
python latent_transformer_v1_structural_atoms.py \
  --data /path/to/corpus.txt \
  --mode structural \
  --device cuda \
  --steps 5000 \
  --atom-count 32 \
  --atom-dim 16 \
  --atom-topk 4
```

Quick plumbing test:

```bash
python latent_transformer_v1_structural_atoms.py \
  --smoke-test --mode structural --device cpu \
  --layers 1 --d-model 48 --heads 4 --d-ff 96 \
  --rank 8 --context 32 --batch-size 2 --steps 2 \
  --eval-every 1 --eval-batches 1 \
  --atom-count 8 --atom-dim 8 --atom-topk 2
```

The runner accepts the common v0 arguments plus:

- --atom-count: number of shared dictionary entries (default 32).
- --atom-dim: learned atom embedding width (default 16).
- --atom-topk: number of atoms selected per matrix (default 4).
- --atom-temperature: softmax temperature over selected atom scores (default 1.0).

## Measurements

The report preserves v0 metrics and adds:

- structural atom configuration;
- atom selection counts across all matrix latents;
- number of distinct atoms used;
- selection entropy.

Parameter counts include the shared dictionary and selector in constructor cost, and matrix-specific latent/gain state in latent cost. Checkpoint size includes all model state.

## First comparison

Use identical corpus, model dimensions, rank, latent dimension, seed, optimizer, schedule, batch size, and training steps for v0 latent and v1 structural. Then compare:

- best and final validation loss;
- training and evaluation throughput;
- constructor parameters and bytes;
- per-matrix latent bytes;
- complete checkpoint bytes;
- optimizer state and peak device memory;
- atom usage and reuse.

Do not compare only the latent payload. The dictionary, selector, coordinate generator, and their optimizer state are part of the representation cost.

## Limitations

- This first version is a controlled architectural probe, not a matched multi-seed scaling study.
- Top-k selection is sparse but discrete in its selected indices.
- Coordinate features remain part of the constructor; this does not test a fully coordinate-free constructor.
- Atom usage counts show selection frequency, not whether atoms have learned interpretable or functionally distinct roles.
- The inherited v0 corpus split is a contiguous 90/10 split. Known validation deterioration must be investigated separately; this experiment does not fix or explain it.
- The model is trained from random initialization on byte-level text, not a pretrained model.

## Classification

OPEN — implementation branch. No quality, efficiency, or generalization advantage is claimed until smoke tests and matched runs are completed.
