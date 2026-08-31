# Developmental Stream — Experiment Proposal

## Question

What persistent structures and behavior emerge when a relational neural system continuously experiences an environment, maintains finite relational memory, and learns what information to retain without predefined semantic concepts?

## Core design

The system receives a continuous stream of heterogeneous observations from an environment generated from explicit constraints rather than semantic labels. It is not given language, coordinates, predefined concepts, or a fixed supervised relation-prediction task.

The system maintains short-term experience and finite-capacity long-term relational memory. Long-term memories can be modified, replaced, or forgotten. A learned consolidation mechanism chooses what survives when capacity is constrained.

The first implementation keeps the persistent identity carrier stable. Continuous learning occurs through mutable contextual state, memory retrieval, and adaptation of the consolidation policy. Changing the underlying representation weights continuously is deliberately left for a later experiment so the first test does not confound multiple learning mechanisms.

## Observation

The experiment records predictions, prediction error, contextual state changes, consolidation events, memory replacement, memory size, and persistent memory contents. A result artifact is written for later analysis.

## Interpretation boundary

Interesting behavior is evidence about the architecture, not evidence of consciousness, semantic understanding, or reproduction of biological brain organization. Generalization must be established by behavior on novel situations rather than by interpreting internal vectors as human concepts.

## Workflow

This experiment follows `PROJECT_WORKFLOW.md`: proposal → discussion/approval → implementation → commit to `main` → user's real run → analysis → closeout and handoff.
