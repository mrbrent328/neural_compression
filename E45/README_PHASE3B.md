# E45 — Persistent Latent Memory (Phase 3B: Cross-Domain Frozen Codec)

## Question

Can one frozen codec trained on several synthetic value domains write and retrieve memories from entirely new value domains, without receiving a domain identifier, projection matrix, or per-domain fine-tuning?

Phase 3A established codec reuse across independent stores that shared one value domain. Phase 3B changes the value-generating projection itself. It tests generalization across held-out projection domains, not natural-language semantics.

## Experimental design

1. Generate several training domains. Each domain is defined by an independently seeded random projection from 8-dimensional latent factors into 32-dimensional values.
2. Train one codec on a mixed stream of records from all training domains.
3. Freeze and save the codec once.
4. Generate new stores for:
   - **Seen domains:** the same projection domains used during training, but with new keys, factors, and key/value associations.
   - **Unseen domains:** independently seeded projection domains never used during training.
5. Use the same frozen codec to write every store. The codec receives no domain ID or projection matrix.
6. Exit the writer process. A separate evaluator process reloads the codec and all stores.
7. Pool all seen- and unseen-domain stores together and evaluate record retrieval, store isolation, and value reconstruction.

The seen/unseen comparison separates ordinary new-association generalization from projection-domain generalization.

## Architecture

The shared codec retains the Phase 3A structure:

- Key writer: key → compact address.
- Query encoder: key-only query → address.
- Value writer: value → compact payload.
- Reader: payload → reconstructed value.

Default dimensions are 32-dimensional keys and values, 8-dimensional addresses, and 8-dimensional payloads. Address and payload values are serialized as float16. Each store is independent; one codec checkpoint is shared across all stores.

## Default protocol

- 8 training projection domains.
- 8 held-out projection domains.
- 1,024 training records per training domain (8,192 total).
- 1,024 records per evaluation store.
- 16 evaluation stores total: 8 seen-domain stores and 8 unseen-domain stores.
- 120 training epochs.
- One pooled evaluation across all 16 stores.

## Measurements

- Pooled top-1 record accuracy and store accuracy.
- Separate seen-domain and unseen-domain retrieval accuracy.
- Mean cosine reconstruction for seen and unseen domains.
- Unseen-minus-seen reconstruction difference.
- Per-store retrieval and reconstruction.
- Complete shared-codec + memory + metadata storage compared with raw float32 key/value storage.

## Run

Requires Python, PyTorch, and NumPy.

Smoke test:

```bash
python E45/e45_latent_memory_phase3b.py --device cpu --smoke
```

Default run:

```bash
python E45/e45_latent_memory_phase3b.py
```

CPU:

```bash
python E45/e45_latent_memory_phase3b.py --device cpu
```

Outputs are written to `runs/E45_phase3B/` and are not committed automatically. Use `--seed 46` or another seed for replication; each seed creates a distinct training and held-out domain set.

## Decision gates

A strong positive result requires:

- High record and store accuracy in the pooled evaluation.
- Strong reconstruction on both seen and unseen domains.
- A small seen-to-unseen reconstruction gap.
- No codec updates after training.
- Storage benefit after charging the shared checkpoint once.

If retrieval remains high but unseen reconstruction falls, the address system is generalizing while the value codec is domain-sensitive. If both seen and unseen reconstruction are poor, the multi-domain training distribution may be too broad for the current payload capacity or reader. If pooled record/store accuracy falls, investigate address collisions separately from value-domain effects.

## Interpretation boundary

Unseen domains are new random projections drawn from the same synthetic generator family. This is a controlled distribution-shift test, not arbitrary real-world domain generalization. It does not establish natural-language memory, pretrained-LM integration, continual learning, conflict resolution, deletion, or superiority to vector databases. The codec has only been trained and tested within this specified synthetic family.


## Phase 3B result (seed 45)

The default run completed with a separate writer and evaluator process. Pooled top-1 record accuracy and store accuracy were both 100% across 16,384 queries and 16,384 pooled records. Seen-domain mean cosine reconstruction was 0.734798; unseen-domain mean cosine was 0.450154, a gap of -0.284644. All seen and unseen stores retained 100% record/store retrieval. Per-store reconstruction cosine ranged from 0.7008 to 0.7703 for seen stores and 0.3519 to 0.4956 for unseen stores.

The full shared system used 642,850 bytes versus 4,194,304 raw key/value bytes (6.525× raw-to-latent ratio; 39.24 bytes/record). This is a strong storage result but must be interpreted alongside the unseen-domain reconstruction degradation.

**Interpretation:** Address retrieval generalizes across the tested held-out projections, but value reconstruction is domain-sensitive. This is one seed and one payload dimension; it does not isolate whether the gap is caused by training-domain coverage, payload bottleneck, or model capacity.

**Next:** Run a multi-seed train-domain-count × payload-dimension ablation and include a matched 32-dimensional payload baseline before deciding how to change the codec.
