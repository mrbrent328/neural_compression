# E47 — Multi-scale parameterization study

E47 compares the repository's dense, low-rank, and latent MLP parameterizations
at matched model scales. It uses the existing
`latent_transformer_v0/latent_transformer_v0.py` training implementation.

## Corpus

Pass a text file or directory containing varied source documents. Supported
extensions include TXT, Markdown, Python, JSON, CSV, HTML, XML, RST, and LOG.
Exact duplicate files are removed. With at least three source files, complete
files are assigned to train/validation/test buckets; otherwise, a contiguous
80/10/10 split uses context-sized guard gaps. Each split must contain at least
4 KiB. The split manifest records source assignments, byte counts, and hashes.

## Quick checks

```bash
python experiments/e47_scale.py --help

python experiments/e47_scale.py \
  --data ~/E47_corpus \
  --scales tiny \
  --modes dense,lowrank,latent \
  --steps 100 \
  --dry-run
```

Remove `--dry-run` to execute the pilot. Training output streams live to the
terminal and is also saved in each run's `training.log`.

## Full study

The defaults run tiny, small, and medium scales across dense, low-rank, and
latent modes for 5,000 steps, with seed 1337. For stronger evidence, use
multiple seeds (for example, `--seeds 1337,2027,31415`). The large scale is
available explicitly with `--scales large`.

```bash
python experiments/e47_scale.py \
  --data ~/E47_corpus \
  --scales tiny,small,medium \
  --modes dense,lowrank,latent \
  --seeds 1337,2027,31415
```

## Outputs

Results are written under `runs/e47_scale/`:

- `corpus_split/`: frozen byte splits and split manifest
- per-run `training.log`, model checkpoint, and model report
- per-run `e47_result.json`: validation and held-out test metrics
- `e47_summary.json`: consolidated settings and results

Use `--resume` to skip runs that already have an `e47_result.json`. The
held-out test split is evaluated after each training run; do not tune against
test results. A single seed is exploratory, not confirmatory.
