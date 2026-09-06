# Latent Transformer v0 — Configuration Sweeps

Use `sweep_latent_configs.py` to run many configurations sequentially and keep every result under one folder.

## Recommended first broad sweep

From `latent_transformer_v0/`:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
  --data Gorgias.txt \
  --profile broad \
  --name gorgias_broad
```

The `broad` profile runs:

- attention heads: `3, 4, 6, 8, 12`
- low-rank/latent ranks: `16, 32, 48, 64`
- latent dimensions: `1, 2, 4, 8, 16, 32`
- constructor hidden size: `128`
- Fourier bands: `8`
- seed: `1337`
- dense controls once per head count
- low-rank controls once per head/rank combination
- latent models for every head/rank/latent combination

That is 145 runs total with the default profile. At roughly four minutes per run on the current machine, expect on the order of 9–10 hours, although individual configurations will vary.

Use `--dry-run` to list the grid without training:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
  --data Gorgias.txt --profile broad --name gorgias_broad --dry-run
```

## Smaller core sweep

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
  --data Gorgias.txt \
  --profile core \
  --name gorgias_core
```

`core` uses heads `4,6,8`, ranks `32,48,64`, and latent dims `1,2,4,8,16,32` (66 total runs including controls).

## Constructor sweep

After identifying a good head/rank neighborhood, this profile varies the shared constructor instead:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
  --data Gorgias.txt \
  --profile constructor \
  --name gorgias_constructor
```

It uses heads `6`, rank `64`, latent dims `1,2,4,8,16,32`, constructor hidden sizes `64,128,256`, and Fourier-band counts `4,8,12`.

## Result layout

For a sweep named `gorgias_broad`:

```text
sweeps/gorgias_broad/
├── manifest.json
├── summary.csv
├── summary.json
├── leaderboard.md
└── runs/
    ├── dense_H3_S1337/
    │   ├── console.log
    │   ├── model.pt
    │   ├── report.json
    │   └── sample.txt
    ├── lowrank_H3_R16_S1337/
    └── latent_H6_R64_Z2_CH128_FB8_S1337/
```

`leaderboard.md` is sorted by best observed validation loss and also shows final validation loss, making overfitting visible. It contains a separate latent-only table.

`summary.csv` is intended for quick sorting/filtering in a spreadsheet. It includes heads, rank, latent dimension, constructor settings, parameter counts, best/final validation loss, speed, memory, checkpoint size, and a short generated sample.

## Resume behavior

The sweep is resumable. Re-running the same command skips any run that already has a valid `report.json` and continues with unfinished configurations.

Use `--force` only when you intentionally want to rerun completed configurations.

## Save disk space

Add `--reports-only` to delete `model.pt` after each successful run while retaining reports, samples, logs and the sweep summaries:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
  --data Gorgias.txt --profile broad --name gorgias_broad --reports-only
```

This is useful for large grids. If a promising configuration appears, rerun that one separately later to keep its checkpoint.

## Custom grids

Any profile dimension can be overridden:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
  --data Gorgias.txt \
  --profile custom \
  --name custom_test \
  --heads 4,6,8 \
  --ranks 32,48,64,96 \
  --latent-dims 1,2,4,8,16,32 \
  --constructor-hiddens 128 \
  --fourier-bands 8 \
  --seeds 1337,1338
```

For reliable scientific comparisons, keep the corpus, training steps, batch size, learning-rate schedule and seed fixed while sweeping one structural family at a time. Multi-seed sweeps are most useful after the broad search identifies a few promising configurations.
