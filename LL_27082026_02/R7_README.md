# R7 — Local Token Prediction

## Purpose

Test whether the local-state/shared-interaction approach used in the CIFAR experiments can be transferred to causal language prediction without adopting a conventional multi-layer language model.

## First-run baseline

- Seed: 314
- Context: 4 tokens
- Embedding: 48
- FFN: 512
- Heads: 1
- Interactions: 1
- Batch: 16
- Epochs: 100
- Learning rate: 0.04
- Optimizer: SGD

These values are a transfer baseline from the current character-model experiments. They are not claimed to be optimal for language.

## Data path

`text -> deterministic tokenization -> token IDs -> local embeddings -> causal shared interaction -> next-token prediction`

The first tokenizer is intentionally simple and deterministic: words, numbers, and punctuation become tokens. This is a laboratory implementation, not a claim that it reproduces the historical R5 tokenizer exactly.

## Diagnostics

Each run records configuration, vocabulary, training history, best checkpoints, local validation states, and R4-inspired latent-state statistics.

## Experimental rule

After the first baseline, change only one configuration value per experiment. Bug fixes use version suffixes rather than new experiment numbers.
