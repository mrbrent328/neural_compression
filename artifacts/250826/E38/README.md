# E38 — Relational-Assisted Local Inference

## Question

Can a persistent relational structure reduce the context presented to a local language model while preserving answer accuracy?

E38 is the first product-oriented continuation of E37. It does not modify or retrain the language model. Instead, it tests whether persistent structure can replace repeated raw history at inference time.

## Conditions

`raw_history` gives the local model the complete observed fact stream.

`relational` gives the local model only a compact relational neighborhood retrieved from persistent memory.

The next planned condition is `relational_only`, where deterministic graph operations answer questions without an LLM when the relation structure is sufficient.

## Files

`e38_memory.py` — SQLite-backed persistent entities and relations.

`e38_corpus.py` — deterministic controlled fact/question stream.

`e38_model.py` — llama.cpp and LM Studio backends.

`run_e38.py` — end-to-end ingestion, retrieval, inference, and measurement harness.

## Running with llama.cpp

From this directory, use a local GGUF model:

```bash
python run_e38.py --backend llama --model /path/to/model.gguf
```

The default experiment uses ninety-six generated fact observations and twenty-four questions. Increase these only after the smoke test works.

## Running with LM Studio

Start its local OpenAI-compatible server, then:

```bash
python run_e38.py --backend lmstudio --model local-model
```

The memory and evaluation code are identical between backends.

## What matters

The first useful signal is not simply whether relational answers are correct. We care about the tradeoff between accuracy and neural workload:

`answer accuracy + prompt size + inference latency + persistent memory size`

A successful result would show similar accuracy with substantially less context. A stronger result would show that some questions can be answered directly from persistent relations, removing the model call entirely.

## Experimental boundary

The controlled corpus uses deterministic fact ingestion so that E38 can isolate retrieval/inference efficiency. It is not evidence that an LLM has learned the relational structure itself. Natural-language extraction and generalization are later tests.
