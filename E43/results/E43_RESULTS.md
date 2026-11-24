# E43 Phase 1 — Ten-Seed Replication Results

**Status:** Confirmed descriptive replication of the synthetic protocol effect; not a pretrained-LM result and not a compression comparison.

## Protocol
- Seeds 43–52; 60 epochs per seed; one fresh process per seed.
- Phase 1 v2 architecture, optimizer, task, split, and channel unchanged.
- Each run evaluated 512 held-out fact identities.
- The same underlying fact-to-class mapping and same held-out identities were reused across all seeds. This varies initialization/training randomness, not dataset assignment.

## Accuracy

| Condition | Mean | Sample SD | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Learned quantized latent | 68.9453% | 4.2443 pp | 68.0664% | 62.8906% | 77.3438% |
| Shuffled quantized latent | 1.5430% | 0.4735 pp | 1.5625% | 0.5859% | 2.5391% |
| Random quantized latent | 1.4063% | 0.4199 pp | 1.2695% | 0.9766% | 2.1484% |
| Zero quantized latent | 1.5625% | 0 pp | 1.5625% | 1.5625% | 1.5625% |
| No message | 1.5625% | 0 pp | 1.5625% | 1.5625% | 1.5625% |

Chance for 64 balanced classes: 1.5625%.

## Learned-channel accuracy by seed

| Seed | Accuracy |
|---:|---:|
| 43 | 67.3828% |
| 44 | 77.3438% |
| 45 | 66.4063% |
| 46 | 72.0703% |
| 47 | 68.7500% |
| 48 | 70.1172% |
| 49 | 62.8906% |
| 50 | 65.6250% |
| 51 | 72.8516% |
| 52 | 66.0156% |

## Paired controls

Learned accuracy was higher in all 10/10 runs against each control:
- Learned minus shuffled: +67.4023 percentage points mean.
- Learned minus random: +67.5391 percentage points mean.
- Learned minus zero: +67.3828 percentage points mean.
- Learned minus no-message: +67.3828 percentage points mean.

These are descriptive paired differences, not significance tests.

## Interpretation

The result supports a learned, task-specific information path through the explicit quantized packet in this synthetic setting. It is robust across ten independent model/training seeds on the fixed task mapping.

It does **not** show generalization across independently generated datasets, pretrained-model communication, semantic compression, or a compression advantage. The 20-byte packet excludes transport framing, and adapter/base parameter and checkpoint costs remain part of the system cost.

**Next discriminating step:** E44 independently randomizes the balanced fact-to-class mapping and crosses dataset seeds with model seeds. This tests whether E43's strong result depends on its fixed `fact_id % 64` structure.
