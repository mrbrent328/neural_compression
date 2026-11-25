# E48 — Latent capacity sweep

## Research question

Can increasing the per-matrix latent code dimension recover predictive quality
while retaining a substantial parameter-count advantage over the dense model?

E48 follows E47's tiny configuration and changes only `latent_dim` across
runs. The default sweep is 8, 16, 32, and 64.

## Controlled variables

- Scale: tiny (2 layers, d_model 128, 4 heads, d_ff 512)
- Mode: latent
- Rank: 32
- Shared constructor hidden width: 128
- Constructor type embedding: 16
- Constructor Fourier bands: 8
- Seed: 1337
- Context: 256
- Batch size: 8; gradient accumulation: 2
- Training steps: 5,000
- Device/dtype defaults: auto / float32

Only the latent dimension changes by default. CLI flags allow explicit changes
for follow-up studies; record those settings when comparing results.

## Run

From the repository root:

```bash
python experiments/e48_latent_capacity.py --help

python experiments/e48_latent_capacity.py \
  --data ~/E47_corpus \
  --latent-dims 8,16,32,64 \
  --steps 5000
```

Prepare and inspect the corpus split and run plan without training:

```bash
python experiments/e48_latent_capacity.py \
  --data ~/E47_corpus \
  --dry-run
```

Use `--resume` to skip dimensions that already have an `e48_result.json`.
The split is deterministic and its SHA-256 hashes are recorded so it can be
compared with E47. Use the same source corpus and context to obtain matching
split hashes.

## Outputs

Results are written under `runs/e48_latent_capacity/`:

- `corpus_split/`: train/validation/test byte files and split manifest
- Per-dimension training log, checkpoint, sample, and report
- Per-dimension `e48_result.json`
- Consolidated `e48_summary.json`

Each run is evaluated on the held-out test split after training. Use validation
loss to select promising configurations; do not tune hyperparameters against
the test split.

## Interpretation

Compare validation/test loss, parameter count, serialized state size, and
elapsed time across latent dimensions. E47's dense result is an external
reference point; E48 itself isolates capacity within the latent architecture.
A single seed is exploratory. Repeat the best configuration across multiple
seeds before drawing confirmatory conclusions.
