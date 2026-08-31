# Developmental Stream — Implementation Design

The experiment uses a small continuous world and the existing relational-memory ideas as a substrate. The first implementation is intentionally minimal.

## Runtime loop

`observation → prediction from current state + recalled long-term memory → prediction error → short-term insertion → contextual state transition → learned consolidation choice → finite-capacity long-term update → next observation`

## Stable versus mutable state

Persistent entity identity vectors are initialized once and remain fixed for the lifetime of this experiment. Contextual states change after observations. Long-term memory stores contextual snapshots associated with persistent entities. The consolidation policy is the only trainable component updated during the run.

## Memory

Short-term memory is a bounded recent-experience buffer. Candidate generation is intentionally simple and broad. A learned consolidation network scores candidates using current internal state and candidate state. Long-term memory has a hard capacity limit. When full, the lowest-scoring stored memory is replaced only when the selected candidate has a higher score.

Long-term memory is read back through similarity-weighted retrieval and is included in the prediction input, so retained memories can influence later predictions.

## Learning signal

The consolidation policy receives a weak delayed self-supervised signal based on change in prediction error across successive experiences. This does not encode semantic importance. It gives the memory-selection mechanism a consequence-sensitive signal while keeping the underlying identity/context representation stable in the first experiment.

## Environment

The environment is generated from numerical constraints and contains no semantic labels. It produces continuous bounded values and local observations. This is deliberately a tiny developmental substrate, not a simulated human world.

## Instrumentation

The program records periodic snapshots, prediction error, context magnitude, consolidation events, source experience, memory size, and final long-term memory contents. The JSON result is the authoritative run artifact.

## Required interpretation discipline

Do not infer semantics or consciousness from state geometry or complexity. The first run establishes whether the continuous state/memory loop is operational and whether adaptive finite memory produces stable, changing, inspectable behavior. Novel-condition generalization is a follow-on analysis/experiment if the first run provides a useful baseline.
