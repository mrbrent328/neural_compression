# S7E1 — Relational Language Model v1

This is the first complete language-learning implementation of the relational architecture proposed after S6E2.

## Architecture

Persistent token identity embeddings feed per-occurrence contextual states. A shared causal relational field performs learned interaction between active states. A tied prediction head predicts the next byte/token. No semantic graph, coordinates, or target geometry are supplied.

This first implementation intentionally uses raw bytes as the vocabulary. That keeps tokenization from becoming another experimental variable. A later experiment can introduce subword tokenization after the core system is validated.

## Exact files needed

Download only these files from `main`:

`memory_network/rlm_v1/__init__.py`

`memory_network/rlm_v1/model.py`

`memory_network/rlm_v1/train.py`

You do not need to download the entire repository to run this first version.

## Requirements

Python three point ten or newer and PyTorch.

From the repository root, install PyTorch using the appropriate command for your machine, then run:

`python -m memory_network.rlm_v1.train path/to/training_data.txt --steps 1000`

The data file is treated as raw bytes, so ordinary UTF-8 text works without any additional tokenizer dependency.

## First run

Use a small text file first. The purpose of the first run is an implementation smoke test, not a benchmark.

`python -m memory_network.rlm_v1.train path/to/training_data.txt --steps 100`

Then run a larger corpus with the same architecture and report the printed loss/perplexity curve and checkpoint output.

## Current scope

This version establishes the complete trainable path. It does not yet implement explicit external relational-memory writes, emergent geometry analysis, or a conventional baseline. Those are follow-up measurements, not hidden assumptions in S7E1.
