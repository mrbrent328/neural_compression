# Developmental Stream — Experiment Closeout

**Date:** 2026-08-31  
**Branch:** `main`

## Question

What persistent structures and behavior emerge when a relational neural system continuously experiences an environment, maintains finite relational memory, and learns what information to retain without predefined semantic concepts?

## What was implemented

The experiment implemented a small continuous numerical world and a continuously learning relational system with:

- fixed persistent identity vectors for 16 entities;
- mutable 32-dimensional contextual state;
- an encoder and shared interaction mechanism for contextual state transitions;
- prediction and prediction-error measurement;
- bounded short-term experience memory (capacity 32);
- bounded long-term memory (capacity 12);
- similarity-weighted long-term retrieval fed back into prediction;
- an online consolidation network trained from prediction consequences;
- periodic instrumentation and a JSON result artifact.

The intended runtime loop was:

`observation → prediction from current state + recalled long-term memory → prediction error → short-term insertion → contextual state transition → learned consolidation choice → finite-capacity long-term update → next observation`

The persistent identity carrier remained fixed throughout the run. The predictor and consolidation policy learned online.

## Actual run

The user's real run used:

- seed: `314159`
- steps: `5000`
- entities: `16`
- state dimension: `32`
- short-term capacity: `32`
- long-term capacity: `12`

Reported results:

- total consolidations: **5000**
- final long-term memory size: **12**
- mean prediction error: **0.27975707083093004**
- mean prediction loss: **0.2327067244131118**
- mean consolidation loss: **0.591435519683361**
- persistent identity updates during run: **false**
- online learning: **true**
- retrieval affects prediction: **true**

The result artifact is `memory_network/developmental_stream_result.json`.

## Main observation

The continuous state/memory loop operated successfully, including finite-capacity long-term memory, replacement, retrieval, contextual state evolution, and online learning.

However, every one of the 5,000 experiences was selected for consolidation. This means the experiment did **not** actually test whether the system can decide that an experience should not be retained.

The implementation's consolidation mechanism ranks available candidates and always selects the highest-scoring candidate. There is no explicit abstain / retain-nothing action. Therefore the experiment tested **candidate ranking under mandatory consolidation**, not full selective memory formation.

The consolidation scores also remained broadly near the neutral region rather than demonstrating a clear, independently established usefulness scale. This makes the next question whether the scorer learned meaningful relative preference among candidates, rather than immediately adding a threshold.

## What the result supports

- A continuous relational state/memory loop can operate without predefined semantic labels.
- Persistent identity can remain stable while contextual state changes.
- Finite-capacity long-term memory can be maintained and replaced during continuous experience.
- Retrieved long-term states can influence subsequent prediction.
- The predictor and consolidation mechanism can learn online from the stream.
- The current architecture provides a workable substrate for testing adaptive memory selection.

## What the result does not establish

- selective learned memory formation;
- semantic understanding;
- consciousness or agency;
- human-like cognition or biological brain organization;
- that retained memories are intrinsically meaningful concepts;
- that the architecture is minimal or optimal;
- useful generalization to novel conditions.

## Important implementation lesson

The experiment exposed an architectural distinction that should remain explicit in future work:

**ranking candidates is not the same as deciding whether anything should be remembered.**

A true selective-memory experiment requires an explicit decision space that permits the system to retain an experience, retain another candidate, or retain nothing. This distinction should be tested rather than silently patched into the present result.

## Next unresolved question

Before introducing an explicit remember/forget threshold, analyze whether the learned consolidation scorer contains meaningful information at all.

Specifically examine:

1. score distributions over time;
2. whether scores correlate with actual prediction-error improvement;
3. whether candidate rankings consistently favor more useful memories;
4. whether ranking becomes more discriminating with experience;
5. whether replacement preferentially preserves memories with measurable predictive value;
6. what structural patterns are present in the final finite memory.

If meaningful ranking exists, the next experiment can add an explicit abstention/forgetting decision. If it does not, the consolidation learning mechanism itself needs to be reconsidered first.

## Interpretation discipline

This result should be preserved as-is. Do not retrofit a threshold into this run and call the resulting behavior evidence from the original experiment. Do not interpret state geometry as semantic content. Novel-condition generalization remains a separate future test.
