# E45 — Persistent Latent Memory (Phase 2)

## Question

Does a compact learned memory preserve arbitrary per-record key/value associations, retrieve the correct value using only a key, survive a genuine process restart, and become storage-efficient as record count grows?

Phase 2 corrects a weakness in Stage 1: Stage 1 generated values from a shared deterministic key-to-value function, so the model could potentially learn that function instead of depending on stored associations. Here, keys and per-record values are independently randomized. Values have low-dimensional structure so payload compression is measurable, but the association between each key and value is arbitrary.

## Record architecture

Each stored record contains two learned parts:

- **Address code:** generated from the key by the writer-side key encoder.
- **Payload code:** generated from the associated value by a separate value encoder.

A later key-only query is mapped into the address space. The closest stored address selects a payload, which the reader decodes into the value.

The default record uses 8 address values + 8 payload values, stored as float16 (32 bytes total). The raw baseline stores a 32-value float32 key and 32-value float32 value (256 bytes total).

## Integrity controls

1. **Aligned memory:** normal retrieval after saving and reloading.
2. **Row-order permutation:** changes physical record order while preserving each address/payload pair. Retrieval should remain unchanged.
3. **Key/payload mismatch:** preserves all addresses and all payloads but permutes payloads across addresses. Address lookup should remain high while value reconstruction should collapse.

The third control is the critical one: it tests whether the retrieved association matters, rather than merely whether the system can produce plausible values.

## Genuine process restart

The default `--phase all` mode trains and writes artifacts in the parent process, then launches a separate Python interpreter to load and evaluate them. This is stronger than merely constructing a second model object in the same process.

## Run

Requires Python, PyTorch, and NumPy.

Quick correctness smoke test:

```bash
python E45/e45_latent_memory_phase2.py --device cpu --smoke
```

Default scaling run (3 seeds × 4 memory sizes):

```bash
python E45/e45_latent_memory_phase2.py
```

CPU run:

```bash
python E45/e45_latent_memory_phase2.py --device cpu
```

Outputs and checkpoints are written under `runs/E45_phase2/`; they are not committed automatically. Use `--phase train` to write artifacts without running the separate-process evaluator, or `--phase eval` to evaluate a previously written run.

## Storage accounting

The report includes:
- raw key/value float32 bytes;
- latent-record tensor bytes;
- serialized memory-file bytes;
- codec checkpoint bytes;
- total latent system bytes (memory file + codec checkpoint);
- total bytes per record;
- raw-to-total-latent storage ratio.

A ratio above 1 means the complete latent system is smaller than the raw key/value baseline at that record count. Small memories are expected to lose because the codec checkpoint is a fixed cost. The break-even point and savings at larger scales are the target observations.

This is still a per-run accounting model: the codec checkpoint is charged to each run. Reusing one trained codec across multiple independent memory stores could amortize that cost further, but is not measured here.

## Interpretation boundary

This remains a synthetic vector-memory experiment. It does not demonstrate natural-language memory, pretrained-LM integration, cross-model portability, superiority to text or vector databases, continual learning, or restoration of a model's complete internal state. The value vectors have shared low-dimensional structure; the key/value pairings themselves are randomized.

A positive result supports only the narrower claim that a learned address/payload representation can preserve and retrieve arbitrary associations in this controlled setting.
