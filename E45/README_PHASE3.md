# E45 — Persistent Latent Memory (Phase 3A: Shared Codec)

## Question

Can one trained, frozen memory codec write and read multiple independent memory stores without retraining, while the total storage cost is lower than storing raw key/value vectors or duplicating the codec for every store?

This phase isolates **codec reuse**. It does not yet test cross-domain memory or pretrained-language-model integration.

## Experimental design

1. Train one codec on a dedicated training episode.
2. Save its weights once and freeze it.
3. Generate multiple independent memory stores with new random keys, new random values, and arbitrary key/value pairings.
4. Use the same frozen codec to write every store. No store-specific training or fine-tuning is allowed.
5. Exit the writer process.
6. Start a separate Python process, reload the codec and all stores, and evaluate queries.

The training episode and all memory stores use the same declared low-dimensional value domain: the same fixed projection from random 8-dimensional factors into 32-dimensional values. The actual factors, keys, and key/value associations are independently generated. This tests generalization to unseen associations and keys within one domain, not arbitrary unseen domains.

## Architecture

- **Shared key writer:** maps a key to a compact address code.
- **Shared query encoder:** maps a later key-only query into the same address space.
- **Shared value writer:** maps a value into a compact payload code.
- **Shared reader:** reconstructs the value from its payload.
- **Independent stores:** each stores only its own latent records and small metadata.
- **Pooled evaluator:** queries across all stores together, checking whether each query resolves to its correct record and correct store.

Default dimensions: 32-dimensional keys, 32-dimensional values, 8-dimensional addresses, and 8-dimensional payloads. Address and payload values are serialized as float16.

## Measurements

- Pooled top-1 record accuracy.
- Pooled top-1 store accuracy (cross-store isolation).
- Mean cosine reconstruction.
- Per-store retrieval and reconstruction.
- Raw key/value storage.
- One shared codec checkpoint plus all memory files and metadata.
- A duplicated-codec accounting baseline that charges the same checkpoint size once per store.
- Shared bytes per record and raw-to-shared-latent storage ratio.

The duplicated-codec comparison is an accounting baseline, not a separate model-quality experiment. Runtime and filesystem allocation overhead are excluded.

## Run

Requires Python, PyTorch, and NumPy.

Quick smoke test:

```bash
python E45/e45_latent_memory_phase3.py --device cpu --smoke
```

Default run: one codec trained on 8,192 associations, then reused across 8 independent stores of 1,024 associations each:

```bash
python E45/e45_latent_memory_phase3.py
```

CPU:

```bash
python E45/e45_latent_memory_phase3.py --device cpu
```

Outputs are written to `runs/E45_phase3A/` and are not committed automatically.

## Decision gates

A useful positive result requires:
- high pooled record accuracy;
- high pooled store accuracy, showing that queries do not drift into another store;
- strong reconstruction across every store;
- no model updates after the training episode;
- complete storage accounting showing the benefit of sharing one codec.

If the pooled result fails while single-store retrieval is strong, investigate address collisions and cross-store interference before changing the codec architecture. If value reconstruction fails across new stores, investigate whether the value codec has learned only the training episode rather than the shared value domain.

## Interpretation boundary

A positive result supports reusable codec behavior only for this synthetic data domain. It does not establish cross-domain semantic memory, pretrained-LM integration, portability across different codec architectures, continual learning, or superiority over conventional vector databases.
