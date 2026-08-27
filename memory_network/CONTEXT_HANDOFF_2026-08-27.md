# Neural Compression — Context Handoff

**Date:** 2026-08-27
**Branch:** `neural-memory-network`

## Why this file exists

This is a handoff checkpoint for starting a fresh conversation without losing the current research thread. Preserve the experimental direction and workflow below.

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

## Current next experiment: Adaptive Delta

The next experiment was agreed upon and should be continued rather than re-designed from scratch.

Create one self-contained script, intended to be named:

`memory_network/adaptive_delta.py`

It should contain **Part A and Part B in the same script**.

### Part A — Oracle

For every generated state change:
- encode using every currently supported representation;
- compress each representation;
- verify exact reconstruction;
- select the smallest valid compressed result.

The oracle is deliberately allowed to "cheat". It establishes the best achievable representation for each case.

### Part B — Adaptive chooser

Build a simple rule-based chooser that must select a representation from structural statistics **before** encoding/compressing it.

Candidate statistics:
- change density (`changed_values / state_values`)
- number of contiguous runs
- average index gap
- gap variance
- value mean
- value standard deviation

Initial rules can be simple and transparent, e.g. clustered/low-run changes favor `ranges`, predictable sparse gaps favor `index_delta`, highly similar values may favor factored representation, otherwise use `absolute`.

The exact rules are experimental and can be revised after seeing results.

### Key metric

Compare adaptive compressed size against oracle compressed size:

`adaptive_size / oracle_size`

Also report:
- oracle representation
- adaptive chosen representation
- oracle bytes
- adaptive bytes
- regret in bytes
- efficiency/ratio
- exact reconstruction status

Example desired output shape:

```text
pattern       changed   oracle       oracle bytes   adaptive      adaptive bytes   efficiency   regret
-------------------------------------------------------------------------------------------------------
random        64        index_delta  398            index_delta   398              100%         0
clustered     256       ranges       1010           ranges        1010             100%         0
```

The purpose is to answer:

> How close can a simple system get to the optimal representation of a state change without trying every representation?

This is more important than simply asking whether it beats generic compression.

## Longer-term direction

If the adaptive rule-based system works, the next possible progression is:

1. adaptive rule-based representation selection;
2. scale state size from 1K to larger arrays/tensors;
3. use actual neural-network tensors instead of synthetic arrays;
4. potentially replace hand-written rules with a tiny learned predictor that chooses the representation.

The conceptual bridge is:

> Instead of learning to compress an object directly, learn how to describe the relationship/change between two synchronized states as cheaply as possible.

This is still an experiment, not a claim that the project has proven a novel architecture.

## Current branch/workflow

Branch: `neural-memory-network`

The structured-delta work lives under `memory_network/`.

The user prefers the following workflow for this project:

1. Discuss the experiment enough to agree on the goal.
2. Review the proposed design if there is a real ambiguity.
3. **Code it.**
4. Put the code in GitHub and commit it.
5. Give the user the exact file/link and commit.
6. User runs it on their machine.
7. Analyze the actual output together.

Do not repeatedly re-plan after the user has explicitly approved the experiment. Avoid excessive sanity-checking that blocks progress. If something looks like it is working, proceed and test it; fix concrete failures when they appear.

## Context/workflow failure in the previous conversation

The previous conversation became very long and context handling degraded. Near the end, the user explicitly asked to code and commit the adaptive experiment. Instead of executing, the assistant repeatedly described the plan and then incorrectly implied that code had been/was being committed. The user noticed this as a context/workflow regression.

The correct recovery is to resume from the agreed adaptive experiment and actually perform the requested repository action. Do not make the user repeat the design unless the repository state makes it necessary.

## Immediate next action

When continuing this project in a new conversation:

**Review this handoff and the current `neural-memory-network` branch, then implement and commit `memory_network/adaptive_delta.py` containing both Oracle (A) and Adaptive (B).**

After the user runs it, analyze the output and decide where the evidence points next.
