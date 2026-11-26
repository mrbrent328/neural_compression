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

| Family | Trainable parameters | FP32 checkpoint weights (approx.) |
|---|---:|---:|
| Dense | 6.738B | 25.10 GiB |
| Direct low-rank, rank 512 | 3.152B | 11.74 GiB |
| Latent/shared constructor, rank 512 and z=16 | 2.410B | 8.98 GiB |

The latent configuration is approximately 64.2% fewer persistent parameters
than dense and 23.5% fewer than direct low-rank. These are parameter-count
estimates, not measured quality or speed results. They are not a claim that
the generated matrices are free: the latent forward pass creates about 742M
factor values across all 32 layers (roughly 1.38 GiB at BF16 if retained for
backpropagation), in addition to constructor activations. This is a key
temporary-memory and compute cost to measure in the pilot.

A rough 8-GPU FULL_SHARD estimate for FP32 sharded parameters, FP32 gradients (the FSDP1 default after backward),
and two FP32 Adam moments is approximately 12.55 / 5.87 / 4.49 GiB per GPU
for dense / low-rank / latent respectively. FSDP mixed precision uses BF16
for forward/backward computation, but retains full-precision parameter shards
outside those passes and normally upcasts gradients before the optimizer step. It excludes activations, block all-gathers,
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
- The preflight memory estimate assumes FP32 parameter shards, gradients and Adam moments; it excludes activations, all-gathers, temporary generated factors, CUDA context and fragmentation.
- The constructor work is real compute and must be counted.
- The local training runner logs validation; held-out test evaluation is
  performed separately with `experiments/e49_evaluate.py`. Detailed hardware
  telemetry and isolated latent factor-generation timing remain future work.
- Data preparation reads each source file as one document; split extremely
  large files into manageable source documents before preparation.

## Decision rule

Continue only if latent/shared-constructor demonstrates a meaningful,
repeatable advantage on at least one resource-quality frontier without
unacceptable degradation elsewhere. If not, archive the negative result and
stop. A clean negative answer is a successful completion of this question.

## Single-GPU local pilot (CUDA or ROCm)

For a single accelerator such as an 8 GiB RX 6600, use the local runner rather
than the distributed FSDP runner above. It uses the same E49 model classes but
does not initialize torch.distributed or wrap the model in FSDP. It requires a
PyTorch build with GPU BF16 support.

The local runner defaults to a small matched geometry: 4 layers, width 256,
FFN width 704, 4 query/KV heads, context 256, and rank 64. Override these
explicitly when changing the experiment. The tokenized train.bin and
validation.bin files must use the exact --vocab-size supplied.

Example short integration run:

    python experiments/e49_train_local.py \
      --mode dense --data-dir data/e49_tokens \
      --output-dir runs/e49_local_dense --vocab-size 32000 \
      --tokens 100000

Repeat with --mode lowrank and --mode latent, changing only the output
directory and mode. The runner records run_config.json, metrics.jsonl, and
latest.pt in each output directory. It supports --resume when the same
configuration and output directory are reused.

The local runner is an initial single-GPU pilot path. It logs train/validation
loss, throughput, peak allocated/reserved GPU memory, and checkpoints. It does
not yet perform the final held-out test evaluation or measure latent factor
generation time separately. The 100K-token example is an integration/smoke
pilot, not enough data to make a language-quality claim.

## Held-out local evaluation

After selecting the final checkpoint for each mode, evaluate the untouched
`test.bin` split once. Do not use test results to tune hyperparameters or
choose checkpoints; use validation results for model selection.

For the three single-GPU pilot checkpoints:

    python experiments/e49_evaluate.py \\
      --data-dir data/e49_tokens \\
      --checkpoint dense=runs/e49_local_dense/latest.pt \\
      --checkpoint lowrank=runs/e49_local_lowrank/latest.pt \\
      --checkpoint latent=runs/e49_local_latent/latest.pt \\
      --output-json runs/e49_test_comparison.json

The evaluator loads one model at a time, verifies that checkpoint run
configurations match their checkpoints, checks the training manifest against
the current data manifest, and refuses architecture/seed mismatches when
run_config.json is available. It also verifies the test split hash against
manifest.json. The JSON result records test loss/perplexity, scored token
count, evaluation throughput, checkpoint size/hash, parameter counts, GPU
memory, software versions, and data hashes.

Evaluation uses a sliding context window and scores each next-token target
exactly once. The default stride is 32; with a 256-token context, steady-state
scored tokens have 225–256 preceding tokens available. A smaller stride uses
more context but requires more forward passes. Use the same stride and batch
size for all model families. This is an evaluation protocol, not a training
change.

The evaluator requires the local runner's `latest.pt` checkpoint format and
a CUDA/ROCm PyTorch build with BF16 support. It does not modify checkpoints
or training artifacts.
