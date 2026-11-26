# E49 — LLM-scale latent/shared-constructor test

## Research question

At approximately 7B dense-model scale, does the shared-constructor latent MLP
retain language-model quality while reducing persistent parameter state,
checkpoint size, or total training/inference cost?

This is falsifiable. Parameter reduction alone is not success if constructor
computation makes training/inference slower or quality collapses.

## Matched families

All modes share decoder-only transformer structure, RoPE, RMSNorm, SwiGLU,
causal scaled-dot-product attention, embeddings, tokenizer, data splits,
context, optimizer, global token batch, evaluation schedule and hardware.

Only MLP parameterization changes:
- dense: three conventional SwiGLU matrices per layer;
- lowrank: independently learned low-rank factors per MLP matrix;
- latent: tiny per-matrix latent plus gain, interpreted by one shared
  coordinate constructor that generates low-rank factors.

Default geometry: 32 layers, width 4096, FFN width 11008, 32 query heads,
32 KV heads, context 4096, vocabulary 32000, rank 512, latent dimension 16.
This is Llama-7B-class geometry, not an exact reproduction of a commercial model.


## Configuration-derived parameter estimate

For the default 32-layer, 4096-wide, 11008-FFN, 32K-vocabulary geometry,
the architecture formulas give these expected counts before a runtime
preflight:

| Family | Trainable parameters | BF16 weight file (approx.) |
|---|---:|---:|
| Dense | 6.738B | 12.55 GiB |
| Direct low-rank, rank 512 | 3.152B | 5.87 GiB |
| Latent/shared constructor, rank 512 and z=16 | 2.410B | 4.49 GiB |

The latent configuration is approximately 64.2% fewer persistent parameters
than dense and 23.5% fewer than direct low-rank. These are parameter-count
estimates, not measured quality or speed results. They are not a claim that
the generated matrices are free: the latent forward pass creates about 742M
factor values across all 32 layers (roughly 1.38 GiB at BF16 if retained for
backpropagation), in addition to constructor activations. This is a key
temporary-memory and compute cost to measure in the pilot.

A rough 8-GPU FULL_SHARD estimate for BF16 parameters, BF16 gradients and two
FP32 Adam moments is approximately 9.41 / 4.40 / 3.37 GiB per GPU for dense /
low-rank / latent respectively. It excludes activations, block all-gathers,
temporary factor tensors, CUDA runtime and fragmentation, so it is not a
hardware sizing guarantee.

## Why a new implementation

The v0 script is a CPU-scale prototype. E49 adds RoPE, SwiGLU, grouped-query
compatible attention, BF16 mixed precision, activation checkpointing, FSDP
FULL_SHARD, distributed checkpoint state, and tokenized memory-mapped data.

The latent constructor still materializes generated low-rank factors as
temporary tensors during forward. That preserves the existing hypothesis,
but its compute and peak temporary memory must be measured. Smaller persistent
state does not imply faster execution.

## Data preparation

Use a varied corpus whose use is permitted. Complete source files are assigned
to train/validation/test; exact duplicate files are removed and hashes recorded.
At least three distinct source files are required. A one-million-token warning
is emitted for small splits. Repeatedly cycling a small project archive measures
memorization, not general language learning.

Install dependencies (PyTorch 2.4+ recommended for the distributed checkpoint APIs):
    pip install torch numpy transformers tokenizers

Run the CPU forward/backward check before any accelerator allocation:
    python experiments/e49_smoke_test.py

Prepare:
    python experiments/e49_prepare_data.py --data /path/to/corpus \
      --tokenizer mistralai/Mistral-7B-v0.1 --output-dir data/e49_tokens

The tokenizer is only a tokenizer; no pretrained model weights are loaded.
Pass the exact tokenizer length as --vocab-size. train.bin and validation.bin
are used during training; test.bin is reserved for final held-out evaluation.

## Preflight and short systems pilot

    python experiments/e49_preflight.py --gpus 8

    torchrun --standalone --nproc_per_node=8 experiments/e49_train.py \
      --mode latent --data-dir data/e49_tokens --output-dir runs/e49_latent \
      --vocab-size 32000 --tokens 50000000 --micro-batch 1 \
      --grad-accum 8 --activation-checkpointing

Repeat with dense and lowrank, changing only mode and output path. This
10M–50M-token pilot is a systems check, not a scientific quality verdict:
verify numerical stability, memory, throughput, checkpoint/restart and
constructor overhead before committing to the full run.

## Full run

Default token budget is 20B. Example:
    torchrun --standalone --nproc_per_node=8 experiments/e49_train.py \
      --mode latent --data-dir data/e49_tokens --output-dir runs/e49_latent \
      --vocab-size 32768 --tokens 20000000000 --micro-batch 1 \
      --grad-accum 8 --activation-checkpointing

Run dense and lowrank with identical controls. Keep seed, world size, global
batch and data split fixed. FSDP checkpoints include sharded model and optimizer
state under output-dir/latest; use --resume to continue.

A 20B-token comparison is substantial but not Chinchilla-optimal pretraining
for a 7B model. Treat it as an architecture comparison at a fixed token budget.

## Required measurements

Record exact source/tokenizer hashes, config, seed, GPU model/count, software
versions, world size, train/validation loss vs tokens, throughput, wall time,
peak allocated/reserved memory, OOM/restarts, parameter breakdown, checkpoint
bytes, optimizer state bytes, and latent factor-generation time/temporary memory.
Evaluate test.bin once after model selection. Compare equal-token and equal-compute
frontiers; both are needed to distinguish data exposure from compute efficiency.

## Caveats

- Validate FSDP and torch.distributed.checkpoint against the exact PyTorch build
  on the chosen cluster before long runs.
- The preflight memory estimate excludes activations, all-gathers, temporary
  generated factors, CUDA context and fragmentation.
- The constructor work is real compute and must be counted.
- This initial runner logs validation but does not yet automate held-out test
  evaluation or detailed hardware telemetry.
- Data preparation reads each source file as one document; split extremely
  large files into manageable source documents before preparation.

## Decision rule

Continue only if latent/shared-constructor demonstrates a meaningful,
repeatable advantage on at least one resource-quality frontier without
unacceptable degradation elsewhere. If not, archive the negative result and
stop. A clean negative answer is a successful completion of this question.
