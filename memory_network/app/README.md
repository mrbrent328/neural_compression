# S6E2 — Working Relational Memory System

First application-oriented system built from the Neural Compression relational-memory hypothesis.

## Design

- SQLite stores exact observations and persistent entity records.
- Neural memory keeps learned persistent identities separate from mutable relational context.
- A shared interaction network updates context from observed edges.
- Relation prediction provides learning pressure.
- A checkpoint persists learned weights and current derived context across restarts.
- Without a checkpoint, the neural state can be reconstructed by replaying durable observations.
- No target embeddings or coordinates are supplied.

## Run

From the repository root:

```bash
python3 -m memory_network.app.interface.cli
```

Example:

```text
> remember apple is_a fruit
> remember apple color red
> remember fruit category food
> inspect apple
> query apple is_a
> train 1000
> query apple category
```

Default files are `data/memory.db` and `data/memory.pt`.

This is deliberately a small inspectable system, not a finished NLP product. Structured observations are used first so the relational mechanism can be tested independently of language parsing.
