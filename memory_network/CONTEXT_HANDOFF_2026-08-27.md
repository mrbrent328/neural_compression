# Neural Compression — Context Handoff

**Date:** 2026-08-28
**Branch:** `main`

## Why this file exists

This is the handoff checkpoint for starting a fresh conversation without losing the current research thread. Preserve the experimental direction and workflow below.

## Current research direction

The project has increasingly focused on a simpler underlying question:

> If two systems already share the same state, what is the smallest persistent/relational structure needed to communicate a change between them exactly?

This grew out of earlier neural-compression/model-synchronization experiments where synchronized models could receive a small latent/delta description of changes and reproduce the changed state. The broader project still does **not** claim a breakthrough or complete explanation of the neural structure. The goal is to find concrete, reusable things that work and then build from them.

## Important prior finding

Earlier compression work demonstrated that files/models can be represented through descriptions/deltas rather than simply storing the complete data. The neural-compression experiments also demonstrated synchronized-state communication and exact reconstruction. Earlier work included successful MP3/MP4 experiments with SHA-256 preservation and high compression throughput; those results remain part of the broader research history.

## Delta synchronization progression

### Delta communication experiment

Two synchronized states were compared by changing different numbers of values.

Example at 1024 state values:
- 1 changed value: 8-byte delta vs 4096-byte full state; 99.8047% reduction; synchronized=true.
- 4: 32-byte delta; synchronized=true.
- 16: 128-byte delta; synchronized=true.
- 64: 512-byte delta; synchronized=true.
- 256: 2048-byte delta; synchronized=true.
- 512: 4096-byte delta; synchronized=true.
- 1024: 8192-byte delta; synchronized=true.

This established the basic exact state-synchronization mechanism.

### Compressed delta experiment

We tested random, similar, and clustered changes and compressed the delta itself.

General result: compression improved the delta representation, especially as the number of changed values increased. The more important observation was that **different structures of change favor different descriptions**.

### Structured delta representation experiment

The same exact state change was represented five ways:

1. `absolute` — explicit changed indices + values.
2. `index_delta` — describe movement/gaps between changed indices.
3. `ranges` — describe contiguous changed regions.
4. `factored_values` — represent values around a shared base.
5. `combined` — range structure + factored values.

The experiment tested `random`, `clustered`, and `similar` changes at 1, 4, 16, 64, 256, 512, and 1024 changed values.

All representations eventually passed exact reconstruction after implementation bugs were fixed.

The result showed that the best representation depends on the structure of the change. In particular, sparse/random cases often favored index-delta representation, while clustered cases increasingly favored ranges.

Examples from the successful run:
- random, 256 changes: `index_delta` was 1291 compressed bytes vs 1452 for absolute.
- clustered, 256 changes: `ranges` was 1010 bytes vs 1019 for index_delta.
- clustered, 512 changes: `ranges` was 1936 bytes vs 1951 for index_delta.
- clustered, 1024 changes: `ranges` was 3789 bytes vs 3816 for index_delta.

The important conceptual result is not that one encoding wins universally; it is that **the structure of the relationship/change determines which description is cheap**.

## Implementation lessons from structured_delta.py

Several bugs occurred while building this experiment:

1. Range reconstruction initially used `np.concatenate()` and failed on empty/invalid range lists.
2. Local downloaded files sometimes had names like `structured_delta (1).py`; the user renamed them to files such as `structured_delta_4.py`. This caused confusion between the file being executed and the current GitHub source.
3. The range packet format was rewritten with explicit metadata rather than relying on inference.
4. `index_delta` initially failed to preserve the first index exactly.
5. Factored floating-point values initially were not lossless despite the experiment requiring exact reconstruction.
6. These were corrected, and the successful run demonstrated `exact=true` for all tested representations/cases.

**Workflow lesson:** when changing code, first inspect the actual current repository version and make sure the user's executed local file corresponds to it. Do not claim a fix was applied unless the repository was actually updated. Do not weaken exactness checks just to make a test pass.

## Experiment: Adaptive Delta Representation Selection

### Question

> How close can a simple system get to the optimal representation of a state change without trying every representation?

### Implementation

`memory_network/adaptive_delta.py` contains both the oracle and adaptive selector.

The oracle evaluates every currently supported representation, compresses it, verifies exact reconstruction, and selects the smallest valid result.

The adaptive system observes only structural statistics of the current delta: change density, run count, average index gap, gap variance, value mean, and value standard deviation. A small nearest-centroid classifier is trained on a separate population using oracle-selected representations as labels. The classifier is then frozen before evaluating a separate test population.

The test comparison also includes a fixed `absolute` baseline. Adaptive decisions are made before candidate encoding sizes are inspected. Oracle results are calculated independently afterward.

### Actual result

The user ran the committed experiment with training seed `1001` and test seed `2002`, using a state of 1024 values, three pattern families (`random`, `clustered`, `similar`), seven change counts (1, 4, 16, 64, 256, 512, 1024), and 100 cases per cell for both training and testing.

The test contained 2100 cases.

Adaptive selector matched the oracle on 1749 of 2100 cases: **83.2857% exact representation-choice match rate**.

The fixed absolute baseline matched the oracle on only **15.8095%** of cases.

Mean adaptive/oracle compressed-size ratio was **1.0115999**, meaning adaptive output was on average about **1.16% larger than the oracle**.

Median ratio was **1.0**. Worst ratio was **1.40217**.

Mean adaptive regret was **1.835 bytes** and maximum regret was **37 bytes**. There were **1749 zero-regret cases**.

Adaptive output was better than the absolute baseline in **1443 cases**, equal in **655 cases**, and worse in only **2 cases**.

Mean absolute-baseline size was **1430.035 bytes**. Mean adaptive size was **1058.786 bytes**, approximately a **26.0% reduction versus the fixed absolute baseline** on this test distribution.

### Observations

The main result is stronger than the raw 83.3% classification accuracy suggests. Most adaptive mistakes were cheap: the classifier was usually close to the oracle even when it selected a different representation.

The printed representative errors included clustered/4 changes, where adaptive chose `absolute` instead of oracle `ranges` for a 5-byte penalty, and similar/4, 16, and 64 changes, where adaptive chose `absolute` instead of `index_delta` for penalties of 1, 11, and 22 bytes respectively.

The adaptive system dramatically outperformed the fixed absolute baseline on this controlled distribution.

### What this does and does not establish

This experiment supports the claim that **structural statistics of a delta can predict which lossless representation is efficient**, and that a very small learned selector can approach the oracle closely on the tested distribution.

It does not establish that the selector generalizes to arbitrary unseen change structures. Training and test data used different random seeds but the same three pattern families and seven change-count levels.

The result therefore supports the structural-selection hypothesis while leaving distributional generalization as the important unresolved question.

## Important implementation/workflow note

The first committed version of `adaptive_delta.py` used hard-coded rules despite describing a training phase. That mismatch was caught during code review before the user's run. It was replaced with the current nearest-centroid selector, which actually learns from oracle labels on the training population and freezes before testing.

This reinforces the standing requirement to review code against the agreed experimental design before the user runs it.

## Standing project workflow

The canonical project workflow is maintained in [`PROJECT_WORKFLOW.md`](../PROJECT_WORKFLOW.md).

Every new research conversation should follow that procedure: review the persistent project state, propose exactly one experiment, discuss and agree on it, design the implementation, code it, commit it to the canonical branch, have the user run it, analyze the actual output, explicitly summarize the completed experiment, and leave a handoff for the next conversation.

Do not repeatedly re-plan an experiment after the user has explicitly approved it. Do not substitute assumed results for the user's real run. Preserve exactness requirements.

## Current branch/workflow

Canonical branch: `main`.

The structured-delta and memory-network work lives under `memory_network/`.

The user prefers the following workflow for this project:

1. Discuss the experiment enough to agree on the goal.
2. Review the proposed design if there is a real ambiguity.
3. **Code it.**
4. Put the code in GitHub and commit it.
5. Give the user the exact file/link and commit.
6. User runs it on their machine.
7. Analyze the actual output together.
8. Before ending the conversation, create an explicit experiment summary covering the question, implementation, actual result, main observation, limits, implementation lessons, and unresolved next decision.
9. Update the handoff with that summary and the next unresolved question.

Do not repeatedly re-plan after the user has explicitly approved the experiment. Avoid excessive sanity-checking that blocks progress. If something looks like it is working, proceed and test it; fix concrete failures when they appear.

## Context/workflow failure in an earlier conversation

A previous conversation became very long and context handling degraded. Near the end, the user explicitly asked to code and commit the adaptive experiment. Instead of executing, the assistant repeatedly described the plan and incorrectly implied that code had been/was being committed. The user noticed this as a context/workflow regression.

The current workflow was created specifically to prevent that failure: repository state is authoritative, one experiment is handled per conversation, approved designs are executed rather than repeatedly re-planned, and each completed experiment receives an explicit closeout summary and handoff.

## Immediate next decision

The next conversation should **not** automatically start coding. It should first review this handoff, `PROJECT_WORKFLOW.md`, and the current `main` branch, then propose exactly one next experiment based on the Adaptive Delta evidence.

The strongest unresolved question currently identified is whether the structural selector generalizes to **novel change structures that were not represented by the training/test pattern families**, rather than merely to new random instances of known families.
