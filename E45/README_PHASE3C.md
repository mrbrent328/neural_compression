# E45 — Persistent Latent Memory (Phase 3C: Value Representation Generalization)

## Question

What explains the Phase 3B reconstruction gap between seen and unseen projection domains: limited training-domain diversity, a narrow payload bottleneck, or a limitation in the current value encoder/reader?

Phase 3B retained perfect record/store retrieval but produced mean cosine reconstruction of 0.7348 on seen domains and 0.4502 on unseen domains (gap −0.2846). Phase 3C is a controlled factorial ablation of the value path. It is not a natural-language or general semantic-memory test.

## Experimental design

A fresh codec is trained for every cell in a 5 × 4 × 3 design:

- Training-domain counts: **2, 4, 8, 16, 32**.
- Payload dimensions: **8, 16, 24, 32**.
- Seeds: **45, 46, 47**, each with an independently seeded family of projection domains.
- Total: **60 runs**.

The total training-record budget is fixed at 8,192 in every cell and divided evenly across the selected training domains. This isolates domain count from total training-example count. Each domain is an independently seeded random projection from 8-dimensional factors into 32-dimensional values.

Every cell evaluates the same number of stores: two seen-domain stores and two held-out-domain stores, with 1,024 records per store. Seen evaluation domains are selected evenly across the training-domain range. Unseen domains use new projection seeds and are never used during training. The frozen codec receives no domain ID or projection matrix.

The writer and evaluator run as separate Python processes. The evaluator reloads the checkpoint and serialized stores before scoring pooled retrieval and reconstruction.

## Payload comparison

Payload dimensions 8, 16, and 24 test progressively wider learned bottlenecks. The **32-dimensional payload** is the no-dimensionality-reduction reference: its learned payload has the same dimensionality as the original value. It is still a learned encoder/decoder and float16 serialization, not an identity transform or bit-identical raw-value oracle.

The address dimension remains fixed at 8 across all cells. The key/address pathway and evaluation protocol are otherwise unchanged.

## Measurements

- Pooled top-1 record and store accuracy.
- Seen and unseen mean cosine reconstruction.
- Seen and unseen mean squared reconstruction error (MSE).
- Unseen-minus-seen cosine and MSE gaps.
- Per-store retrieval and reconstruction.
- Complete shared checkpoint + memory + metadata storage, compared with raw float32 key/value storage.
- Across-seed mean, sample standard deviation, minimum, and maximum for each factorial cell.

Quality and storage must be considered jointly. A wider payload may improve reconstruction while reducing the storage advantage.

## Run

Requires Python, PyTorch, and NumPy.

Smoke test (two quick cells; checks execution and artifact structure, not model quality):

    python E45/e45_latent_memory_phase3c.py --device cpu --smoke

Full sweep:

    python E45/e45_latent_memory_phase3c.py --device cpu

Or use another supported PyTorch device:

    python E45/e45_latent_memory_phase3c.py --device cuda

The default sweep runs 60 cells. To run a smaller diagnostic subset:

    python E45/e45_latent_memory_phase3c.py --device cpu --seeds 45 --domain-counts 2 8 32 --payload-dims 8 32

Optional --epochs changes training duration; default is 120. Outputs are written under runs/E45_phase3C/ and are not committed automatically. Each cell has its own directory, checkpoint, stores, and result.json. partial_results.json is updated after each completed cell so an interrupted sweep still preserves completed results. The final aggregate is e45_phase3C_summary.json.

## Interpretation guide

- If more training domains improve unseen reconstruction at fixed payload dimension, domain diversity is helping.
- If wider payloads improve seen and unseen reconstruction, the original 8-dimensional payload was a meaningful bottleneck.
- If 32-dimensional payloads still show a large unseen-domain gap, the current learned value encoder/reader or training objective may be limiting cross-domain generalization.
- If retrieval accuracy drops, investigate the address path separately; reconstruction conclusions are not clean when the wrong record is selected.
- If reconstruction improves only as storage expands, report the quality/storage trade-off rather than calling it a compression win.

## Limitations

- Synthetic vectors only; held-out domains are random projections from the same generator family.
- The 32-dimensional reference removes dimensionality reduction but remains a learned float16 autoencoder.
- Two seen and two unseen stores are evaluated per run; the three seeds provide an initial replication estimate, not a definitive statistical conclusion.
- Storage excludes runtime, filesystem allocation, and indexing overhead beyond listed metadata.
- Continual writes, deletion, conflicts, interference under updates, forgetting, natural language, pretrained models, and vector-database comparisons remain untested.
