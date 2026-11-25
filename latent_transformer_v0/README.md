# Latent Transformer v0

This folder turns the neural-compression latent/shared-constructor work into a language model that trains from random initialization.

The experiment compares three MLP parameterizations inside the same tiny decoder-only transformer:

- `dense`: ordinary dense MLP matrices.
- `lowrank`: each MLP matrix directly owns trainable `L` and `R` factors, with `W = L @ R`.
- `latent`: each MLP matrix owns only a small latent vector `z` (plus one scalar gain). One shared coordinate constructor generates `L` and `R` on demand. The full dense MLP matrix is never an independently learned or serialized parameter.

Attention, embeddings, RMSNorm, residual structure, loss, tokenizer, and training loop are intentionally conventional. v0 changes only the MLP parameter representation so failure/success is interpretable.

## Why the constructor is coordinate-based

A naive hypernetwork whose last layer outputs every factor value can hide the storage cost inside a giant decoder. v0 avoids that. The shared constructor receives:

`latent + matrix-type embedding + fixed Fourier coordinate features`

and produces one rank-dimensional factor row/column. Its parameter count therefore does not scale directly with `d_model`, `d_ff`, or number of transformer layers.

For a latent matrix:

`z -> shared constructor -> (L, R)`

and execution is direct through the factors:

`y = (x @ R.T) @ L.T`

No dense `W` has to be materialized.

## First test

Requires PyTorch 2.x. The tokenizer is byte-level (`vocab_size=256`) so no tokenizer package or pretrained model is needed.

Quick plumbing test:

```bash
python latent_transformer_v0.py --smoke-test --mode latent --steps 50
```

Train on any UTF-8 text corpus:

```bash
python latent_transformer_v0.py \
  --data /path/to/corpus.txt \
  --mode latent \
  --device auto \
  --steps 5000
```

Run the controls with the same settings:

```bash
python latent_transformer_v0.py --data /path/to/corpus.txt --mode dense   --steps 5000
python latent_transformer_v0.py --data /path/to/corpus.txt --mode lowrank --steps 5000
python latent_transformer_v0.py --data /path/to/corpus.txt --mode latent  --steps 5000
```

Default model geometry is 6 layers, `d_model=384`, 6 heads, `d_ff=1536`, context 256, factor rank 64, latent dimension 32.

On AMD ROCm, PyTorch normally exposes the GPU through the CUDA API, so `--device auto`/`cuda` is still the expected PyTorch device name when ROCm is installed correctly.

## Outputs

Each run writes a separate directory under `runs/latent_transformer_v0/` containing:

- `model.pt` — checkpoint
- `report.json` — configuration, parameter accounting, loss/perplexity history, throughput, and peak CUDA/ROCm allocation when available
- `sample.txt` — greedy byte-level sample

The report separates total parameters, constructor parameters, per-matrix latent state, MLP parameters, attention parameters, and other parameters.

## What counts as success in v0

The first question is deliberately narrow: **can the latent-generated model learn language with a normal-looking loss curve?**

If it can, the next experiments should sweep latent dimension and factor rank under identical data/training budgets, then compare validation loss, checkpoint bytes, optimizer memory, peak device memory, and tokens/sec against `dense` and `lowrank`.

The `lowrank` control matters: if `latent` only matches the gains of direct low-rank factors, the improvement is factorization rather than the shared latent representation. If `latent` can retain quality while using much less matrix-specific persistent state, the shared constructor is contributing something additional.

## Structural atom comparison

The first matched v0-latent versus v1-structural-atoms pair (seed 1337, 5,000 steps each) is complete. The atom model achieved effectively tied validation quality, used 2,528 fewer total parameters, and trained 13.37% more slowly on CPU. This single-seed result does not establish an advantage.

See [the matched results, interpretation, and raw JSON artifacts](RESULTS_STRUCTURAL_ATOMS_V1.md).
