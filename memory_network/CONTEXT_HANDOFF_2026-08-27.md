# Neural Compression — Context Handoff

**Date:** 2026-08-28
**Branch:** `main`

## Purpose

This is the durable checkpoint for resuming Neural Compression research in a fresh conversation. The repository is authoritative. Follow `PROJECT_WORKFLOW.md`: one experiment per conversation, discuss before coding, implement only after approval, commit to `main`, use the user's real run for analysis, then close out and update this handoff.

## Current research direction

The project is increasingly centered on:

> If two systems already share the same state, what is the smallest persistent/relational structure needed to communicate a change between them exactly?

Earlier work established exact synchronized-state communication and showed that different structures of change favor different descriptions. The project does not claim a complete theory of neural structure; it is looking for concrete, reusable mechanisms.

## Established delta findings

The structured-delta work uses five lossless representations: `absolute`, `index_delta`, `ranges`, `factored_values`, and `combined`. Exact reconstruction is a hard requirement. The experiments established that representation efficiency depends on the structure of the change: sparse/random changes often favor index deltas, while clustered changes increasingly favor ranges.

The Adaptive Delta experiment then trained a small nearest-centroid selector using six structural statistics: change density, run count, average index gap, gap variance, value mean, and value standard deviation. On 2100 known-distribution test cases it matched the oracle on **83.2857%** of cases, had **1.0115999** mean adaptive/oracle ratio, **1.835 bytes** mean regret, and reduced mean size by about **26%** versus fixed absolute encoding.

## Completed experiment: Out-of-Distribution Structural Delta Selection

### Question

Can the small structural selector generalize to change structures that were not represented by its training families?

### Implementation

`memory_network/generalization_delta.py` reused the existing lossless representations and nearest-centroid selector. Training used the original `random`, `clustered`, and `similar` families with seed `1001`, state size 1024, seven change counts (1, 4, 16, 64, 256, 512, 1024), and 100 cases per cell: **2100 training cases**.

The selector was frozen before testing. Test seed was `2003`, with 100 cases per cell. Known controls were `random`, `clustered`, and `similar`. Novel structures were `strided`, `alternating`, `multi_cluster`, `edge_biased`, and `clustered_similar`.

Adaptive decisions were made before candidate encoded sizes were inspected. The oracle was evaluated independently afterward. Exact reconstruction remained mandatory.

### Actual user result

The user ran the experiment locally and supplied the complete output.

Known controls: **2100 cases**, **83.0952%** oracle-choice match, **1.0112543** mean ratio, **1.7571 bytes** mean regret, **38 bytes** maximum regret.

Novel structures: **3500 cases**, **82.1143%** oracle-choice match, **1.0558700** mean ratio, **40.1143 bytes** mean regret, **821 bytes** maximum regret.

Overall: **5600 cases**, **82.4821%** oracle-choice match, **1.0391391** mean ratio, **25.7304 bytes** mean regret, **821 bytes** maximum regret, and **4619 zero-regret cases**. Adaptive output was better than absolute in **3789** cases, equal in **1799**, and worse in only **12**. Mean absolute size was **1448.3382 bytes**; mean adaptive size was **1072.6602 bytes**.

### Per-pattern result

`random`: 99.5714% match, 0.0057-byte mean regret.

`clustered`: 88.5714% match, 0.6486-byte mean regret.

`similar`: 61.1429% match, 4.6171-byte mean regret.

`strided`: 95.1429% match, 0.27-byte mean regret.

`alternating`: 99.5714% match, 0.0043-byte mean regret.

`multi_cluster`: 87.2857% match, 2.7671-byte mean regret.

`edge_biased`: 100% match, 0-byte mean regret.

`clustered_similar`: **28.5714% match**, **197.53-byte mean regret**, **821-byte maximum regret**.

### Main observation

The selector generalizes substantially to genuinely unseen structures. Its novel-pattern match rate remains close to the known-pattern control, and several unseen geometries are handled extremely well. This is evidence against simple memorization of the original pattern labels.

The dominant failure is the composed `clustered_similar` structure. The worst cases occur at 512 changed values, where the selector chooses `absolute` while the oracle chooses `ranges`, causing regrets up to 821 bytes.

The current evidence therefore supports a narrower claim: cheap structural statistics contain transferable information for choosing among lossless delta representations. It does **not** establish universal generalization.

### Interpretation boundary

Do not retroactively tune the selector based on this result. The failure is part of the evidence.

The current leading hypothesis is that the selector handles individual structural dimensions well but does not reliably model interactions between them when familiar properties are combined in a new way. This is a hypothesis, not yet an established cause.

## Important implementation lessons

Keep exact reconstruction checks intact. Keep adaptive decisions separate from oracle measurements. Review the actual repository version before making implementation claims. Preserve the distinction between observed results and interpretation.

The `clustered_similar` failure should be investigated rather than patched immediately.

## Current unresolved question / next decision

Before changing the selector, determine **why `clustered_similar` fails**.

The next experiment should distinguish two competing explanations:

**Missing information:** the six current structural features do not contain enough information to distinguish the relevant representation interaction.

**Insufficient model:** the features contain the needed information, but a nearest-centroid classifier cannot represent the interaction between them.

The first step in the next conversation should be to inspect the failed cases and compare the feature-space geometry of `clustered`, `similar`, and `clustered_similar`. Then propose exactly one experiment that cleanly separates those explanations.

## Repository artifacts

The OOD experiment code is `memory_network/generalization_delta.py`.

The actual summarized result is recorded in `memory_network/generalization_delta_result_2026-08-28.json`.

The formal experiment closeout is `memory_network/EXPERIMENT_CLOSEOUT_2026-08-28.md`.

The canonical workflow is `PROJECT_WORKFLOW.md`.
