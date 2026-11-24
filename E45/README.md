# E45 — Persistent Latent Memory (Stage 1)

## Motivation

E43/E44 ask whether learned latent packets can carry task-relevant information
between independent networks. E45 asks a related but distinct question:

> Can a learned latent representation preserve task-relevant information across
> a process boundary, be saved to persistent storage, reloaded, and queried later?

The first stage isolates the memory substrate before adding a pretrained
language model. It is intentionally not described as pretrained-LM memory.

## Architecture

- **Writer:** encodes an observed key/value pair into a compact latent record.
- **Address encoder:** maps a later key-only query into the latent address space.
- **Reader:** reconstructs the associated value from the retrieved latent.
- **Persistent store:** serializes latent records separately from the codec weights.
- **Reload test:** constructs a fresh codec instance, loads its checkpoint and
  the saved memory, then performs retrieval.

The model learns a content-addressed memory. It does not receive row indices
when answering queries.

## Run

Requires Python, PyTorch, and NumPy:

```bash
python E45/e45_latent_memory.py
python E45/e45_latent_memory.py --device cpu --seed 45
```

Outputs are written to `runs/E45/` and are not committed automatically.

## Controls and measurements

- Learned latent memory after save/reload.
- Shuffled-record control, preserving the records while breaking their
  association with the query keys.
- Cosine reconstruction quality and top-1 address rate.
- Raw key/value float32 bytes versus latent-record bytes.
- Codec checkpoint size reported separately so model cost is not hidden.

## Interpretation boundary

This is a synthetic associative-memory substrate test. It does not yet show
that a pretrained language model can use latent memory, that latent memory
beats text or conventional vector databases, or that a model can restore its
complete internal computational state.

## Next stages

1. Validate save/reload and retrieval across multiple seeds and record counts.
2. Add continual writes, duplicate/conflicting observations, and interference
   tests; measure forgetting and update cost.
3. Compare against raw vectors, conventional embeddings, and text records with
   complete storage and compute accounting.
4. Add a thin adapter for a frozen pretrained model. The model supplies
   contextual hidden states; a learned memory interface writes compact records
   and returns retrieved context through a model-compatible conditioning path.
5. Test process restart and model restart separately. Cross-architecture memory
   portability is a separate, harder claim.

A useful target metric is task-relevant information recovered per total
persistent byte, including the codec checkpoint and required metadata.
