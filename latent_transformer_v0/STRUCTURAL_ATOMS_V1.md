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

## First matched comparison — 2026-11-24

The first matched pair completed for seed 1337. Both runs used the same 204,955-byte corpus, model dimensions, seed, CPU float32 execution, 5,000-step schedule, and contiguous 90/10 train/validation split.

| Measurement | v0 latent | v1 structural atoms |
|---|---:|---:|
| Total parameters | 3,807,618 | 3,805,090 |
| Constructor parameters | 66,480 | 63,952 |
| Checkpoint bytes | 15,255,899 | 15,246,732 |
| Best validation loss (step 2,250) | 1.655740 | 1.653539 |
| Final validation loss | 2.201734 | 2.201038 |
| Overall throughput | 3,719.93 tok/s | 3,222.39 tok/s |
| Elapsed time | 5,505.48 s | 6,355.53 s |

The atom variant used 2,528 fewer total parameters (0.066%) and its checkpoint was 9,167 bytes smaller (0.060%). Its best validation loss was lower by 0.002202 (about 0.13%), but the final validation difference was only 0.000696. These differences do not establish a meaningful quality improvement from one seed.

The atom variant was 13.37% slower in overall CPU throughput and took 850 seconds longer. The sparse composition did not produce a training-speed advantage in this implementation.

Atom usage was concentrated: 7 of 32 atoms were selected across 18 matrix latents, with three atoms accounting for 54 of 72 selections. Selection entropy was 1.666 nats against a 3.466-nat maximum. This is a selection-frequency observation, not evidence of semantic specialization.

**Interpretation:** The first matched run shows effectively tied language-model quality, marginally fewer constructor parameters/checkpoint bytes, and slower training for the atom variant. It does not demonstrate an overall quality, storage, or compute advantage. Both models reached their best validation result at step 2,250 and deteriorated afterward while training loss continued to fall. Because the inherited split is a contiguous 90/10 byte split, this is validation deterioration; it should not be attributed solely to overfitting without a split/generalization control.

See [the full matched comparison and raw reports](RESULTS_STRUCTURAL_ATOMS_V1.md).

## Limitations

- One matched seed only; this is not a multi-seed scaling study.
- Top-k selection is sparse but discrete in its selected indices.
- Coordinate features remain part of the constructor; this does not test a fully coordinate-free constructor.
- Atom usage counts show selection frequency, not whether atoms have learned interpretable or functionally distinct roles.
- The inherited v0 corpus split is a contiguous 90/10 split; validation deterioration may include corpus-order distribution effects.
- The model is trained from random initialization on byte-level text, not a pretrained model.
- CPU float32 only; no GPU or inference-latency benchmark. Peak device memory was not reported.

## Classification

OPEN — implementation and exploratory result. The first matched run does not establish an advantage. Multi-seed replication, a controlled validation split, and further profiling are required before making a quality or efficiency claim.
