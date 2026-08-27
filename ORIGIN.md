# Neural Memory Network

This branch is a clean implementation track derived from the Neural Compression research.

## Why this exists

The parent research repeatedly explored persistent state, local representations, relationships between local units, and compact descriptions of change. The purpose of this branch is to stop treating those ideas only as experiments and try to assemble them into a usable system.

## Working hypothesis

A distributed collection of small persistent models or memory units may be able to answer queries by communicating compact state changes rather than repeatedly exchanging complete states.

Conceptually:

    query -> local memory -> delta -> local memory -> delta -> ... -> answer

A synchronized reference state may allow two units to communicate only what changed, building on the earlier Neural Compression experiments.

## First implementation goals

1. Represent one persistent local memory independently.
2. Establish a compact state-delta format between synchronized peers.
3. Verify that a receiver can apply a delta and recover the sender's state/behavior.
4. Extend to multiple memories and query propagation.
5. Measure useful behavior, memory growth, communication cost, and propagation depth.

## Research rule

This branch is exploratory. A working result is enough reason to build on it. We will investigate failures when they matter, but will not spend experiments eliminating hypothetical ambiguities that do not change the next engineering decision.
