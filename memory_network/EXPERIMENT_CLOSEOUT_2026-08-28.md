# Experiment Closeout — Out-of-Distribution Structural Delta Selection

**Date:** 2026-08-28
**Branch:** `main`
**Experiment:** Out-of-distribution structural delta selection generalization
**Code:** `memory_network/generalization_delta.py`

## Question tested

Can the small nearest-centroid structural selector learned from the original `random`, `clustered`, and `similar` delta families choose efficient lossless representations for genuinely unseen change structures?

## Implementation

The experiment reused the existing lossless representations and selector from `adaptive_delta.py`. Training used seed `1001`, a state of 1024 values, seven change counts (1, 4, 16, 64, 256, 512, 1024), three training families, and 100 cases per cell, for 2100 training cases.

The selector observed only six structural statistics: change density, run count, average index gap, gap variance, value mean, and value standard deviation. After training, it was frozen.

Testing used seed `2003` and 100 cases per cell. Known controls were `random`, `clustered`, and `similar`. Novel structures were `strided`, `alternating`, `multi_cluster`, `edge_biased`, and `clustered_similar`. The adaptive representation was selected before candidate encoded sizes were inspected. The oracle was evaluated independently afterward. Exact reconstruction remained a hard requirement.

## Actual result

The complete test contained 5600 cases: 2100 known-pattern controls and 3500 novel cases.

Known cases produced an 83.0952% oracle-choice match rate, a mean efficiency ratio of 1.0112543, mean regret of 1.7571 bytes, and maximum regret of 38 bytes.

Novel cases produced an 82.1143% oracle-choice match rate, a mean efficiency ratio of 1.0558700, mean regret of 40.1143 bytes, and maximum regret of 821 bytes.

Across all 5600 cases, the selector matched the oracle on 82.4821% of cases. The median efficiency ratio was 1.0, mean efficiency ratio was 1.0391391, mean regret was 25.7304 bytes, maximum regret was 821 bytes, and 4619 cases had zero regret.

The adaptive result was better than fixed `absolute` in 3789 cases, equal in 1799 cases, and worse in only 12 cases. Mean absolute size was 1448.3382 bytes; mean adaptive size was 1072.6602 bytes.

## Per-pattern result

`random`: 99.5714% match, 1.0001216 mean ratio, 0.0057-byte mean regret, 2-byte maximum regret.

`clustered`: 88.5714% match, 1.0116487 mean ratio, 0.6486-byte mean regret, 37-byte maximum regret.

`similar`: 61.1429% match, 1.0219927 mean ratio, 4.6171-byte mean regret, 38-byte maximum regret.

`strided`: 95.1429% match, 1.0063582 mean ratio, 0.27-byte mean regret, 8-byte maximum regret.

`alternating`: 99.5714% match, 1.0001229 mean ratio, 0.0043-byte mean regret, 1-byte maximum regret.

`multi_cluster`: 87.2857% match, 1.0028040 mean ratio, 2.7671-byte mean regret, 83-byte maximum regret.

`edge_biased`: 100% match, 1.0 mean ratio, 0-byte mean regret, 0-byte maximum regret.

`clustered_similar`: 28.5714% match, 1.2700650 mean ratio, 197.53-byte mean regret, 821-byte maximum regret.

## Main observation

The selector generalized surprisingly well to novel structural families. Novel-pattern accuracy remained close to the known-pattern control despite the selector never being trained on those families. Strided, alternating, multi-cluster, and edge-biased structures were handled well.

The dominant failure was the composed `clustered_similar` structure. The selector frequently chose `absolute` while the oracle chose `ranges`, especially at 512 changed values. This produced the largest observed regrets, up to 821 bytes.

The result therefore suggests that the selector is not simply memorizing the names or exact samples of the original pattern families. Its structural features transfer to several unseen geometries. However, the current nearest-centroid model does not reliably represent interactions between structural dimensions when familiar properties are combined in a new way.

## What this establishes

The experiment supports the narrower claim that cheap structural statistics can provide transferable information for selecting among lossless delta representations. A very small frozen selector retained substantial performance on genuinely unseen change structures.

It does not establish arbitrary or universal generalization. The composed `clustered_similar` family exposes a clear boundary, and the test distribution is still synthetic and limited to the structures implemented here.

## Implementation and workflow lessons

The experiment correctly kept the selector frozen and separated adaptive decisions from oracle measurement. Exact reconstruction was preserved throughout.

The result should not be tuned retroactively. The `clustered_similar` failure is useful evidence and should remain visible in the record.

For future work, inspect the failed cases before changing the selector. Determine whether the failure comes from missing information in the six structural features or from the nearest-centroid model being unable to represent feature interactions. Those are distinct hypotheses and should be separated experimentally.

## Unresolved next decision

Before implementing another selector, determine why `clustered_similar` is misclassified. Compare its feature vectors and nearest-centroid geometry against `clustered` and `similar` cases. The next experiment should distinguish **missing structural information** from **insufficient interaction modeling**.
