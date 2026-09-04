# Neural Compression — Context Handoff — 2026-09-03

## E40 Closeout

### Experiment

**E40 — Adaptive Residual Codec / Local Predictive Representation**

### Question tested

Can a codec use a locally adaptive predictive representation to choose the cheapest representation for each region, without an oracle, while maintaining exact reconstruction?

### What was implemented

E40 implemented a real adaptive residual codec with:

- multiple genuinely different predictive representations;
- local region signatures based on observed data;
- candidate generation for every representation;
- complete end-to-end cost comparison including representation metadata;
- deterministic selection of the cheapest candidate;
- synchronized encoder/decoder predictor state;
- exact reconstruction;
- malformed, truncated, invalid-representation, and trailing-stream validation;
- a real file-oriented CLI for encoding and decoding arbitrary files.

The representation selector is part of the actual codec rather than an oracle or precomputed strategy map.

The core architecture is intentionally modular: a `Predictor` can be replaced by another deterministic base model/codec while E40 supplies the local residual/representation-selection layer.

### Validation

The committed E40 validation suite passed all 8 tests:

1. exact roundtrip;
2. mixed regions use real adaptation;
3. signatures are local;
4. complete candidate cost includes metadata;
5. stateful predictor remains synchronized;
6. trailing bytes are rejected;
7. invalid representation IDs are rejected;
8. truncated payloads are rejected.

The separate CLI integration test also passed: `PASS test_cli_file_roundtrip`.

### Real-file results

| Input | Original | E40 | Change |
|---|---:|---:|---:|
| Text | 8,510 B | 4,736 B | -44.35% |
| MP3 | 5,856,348 B | 5,980,021 B | +2.11% |
| MP4 | 7,893,863 B | 7,999,438 B | +1.34% |
| ZIP | 1,067,660 B | 1,091,188 B | +2.20% |

All real-file tests reconstructed byte-for-byte exactly.

Adaptive versus best fixed representation:

- MP4: 131 bytes better
- MP3: 1 byte better
- ZIP: 6 bytes better
- Text: 1 byte better

### Observations

The adaptive selector behaves coherently according to local structure.

MP4 selected 7,599 compact, 28 balanced, 13 deep, and 69 raw regions out of 7,709.

MP3 selected 5,688 compact, 6 balanced, 0 deep, and 26 raw regions out of 5,720.

ZIP selected 1,041 compact, 1 balanced, 0 deep, and 1 raw region out of 1,043.

Text selected 1 compact, 1 balanced, 0 deep, and 7 raw regions out of 9.

Already-compressed MP3/MP4/ZIP data provides essentially no useful additional byte-level predictive structure for E40's current reference predictor. The system overwhelmingly selects its cheapest representation in those cases.

Text contains substantially more exploitable local structure, producing a 44.35% reduction under this codec.

### Interpretation

E40 does **not** establish a competitive general-purpose compressor. That is no longer the primary research objective.

The stronger result is that E40 demonstrates a working mechanism for selecting a minimal local predictive representation based on the structure actually present in each observation, while preserving exact reconstruction and allowing a pluggable predictor/model layer.

The current reference predictor is primitive. Its weak results on already-compressed formats are therefore primarily evidence about the predictor's limits, not a rejection of the adaptive architecture.

The selector itself is real and functioning: it measures actual candidates, includes representation metadata in the cost, chooses deterministically, and synchronizes the chosen path with the decoder.

### What E40 does not establish

E40 does not yet demonstrate:

- cross-observation persistence;
- useful learned latent structure;
- improvement to another model's predictions;
- generalization of a representation across related observations;
- that the current byte-level predictive representations are the right mechanism for model aiding;
- that the adaptive architecture is universally useful.

### Research-direction transition

The project should no longer frame the primary objective as "build a better compressor."

Compression/prediction is now better treated as an experimental measurement environment for a broader model-aiding question:

> **Can a small persistent relational representation capture structure that helps another model predict or explain observations?**

This remains aligned with the project North Star:

> **What is the smallest persistent relational structure capable of explaining multiple observations of the same underlying system?**

A useful conceptual target is:

> **What is the cheapest persistent representation that explains an observation or region well enough to minimize the complete information required for prediction/reconstruction?**

### Current model

The accumulated evidence continues to support the working hypothesis that a compact persistent representation may be more useful when it preserves reusable relationships between local observations than when it independently preserves each observation.

E40 adds an implementation-level bridge: a base predictor can explain what it can, E40 can characterize and encode the residual locally, and representation capacity can be selected from measured end-to-end cost. The next step is to determine whether the persistent representation itself can become useful to another predictive model across multiple observations.

### Next decision — deliberately unresolved

Do **not** preselect E41 yet.

The next conversation should review the persistent project state and decide the smallest experiment that moves from E40's local predictive/codec mechanism toward a genuine **model-aiding / multi-observation persistent representation** test.

The key discriminator should be whether an auxiliary persistent structure can capture information that is reusable across observations and improves prediction/explanation beyond what an individual observation or base predictor can provide.

Candidate conceptual directions may include a controlled multi-observation system, a persistent local representation with relations, or another minimal test of reusable structure. The exact E41 design should be proposed and agreed in the next experiment conversation rather than committed here prematurely.

### Important implementation lesson

For future system-building experiments, implement the mechanism as a real reusable system from the beginning. Do not create temporary oracle scaffolding with the intention of replacing it later. Representation selection, persistence, model interaction, and decoding/reconstruction boundaries should be genuine parts of the artifact whenever they are part of the hypothesis.

### Relevant committed code

- `E40_adaptive_residual_codec.py` — E40 core codec
- `test_E40_adaptive_residual_codec.py` — 8-test validation suite
- `E40_cli.py` — file-oriented CLI
- `test_E40_cli.py` — CLI exact-roundtrip validation

### Commits

- `685a1896fd626910886ff68e953998cb931b5540` — `fix: complete E40 adaptive residual codec`
- `d564cae541688f38d83f84311b80c2340b4311ca` — `feat: add E40 file codec CLI`

Canonical branch: `main`.

### Handoff state

E40 is closed. The repository now contains a real, exact, locally adaptive predictive/residual codec and file interface. The research question is moving beyond compression toward reusable persistent structure and model aiding. E41 remains an open design decision for the next research conversation.
