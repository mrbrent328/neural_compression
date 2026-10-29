# Relational Cognitive Engine v0

This prototype promotes the persistent-relational ideas from the Neural Compression experiments into a candidate cognitive substrate.

## Core hypothesis

A useful local intelligence may be cheaper if its persistent internal model is an explicit, continuously changing network of concepts, relations, beliefs, and provenance, while neural models are treated as optional perception/language adapters rather than the place where all knowledge lives.

The intended cycle is:

`perception -> concept activation -> relational propagation -> inference -> model update -> changed future interpretation`

Memory is therefore not a separate retrieval store. Durable memory is structural change in the model itself.

## v0 goals

The first version intentionally avoids a large LLM. It tests whether the substrate itself can:

- create and update persistent concepts and relations;
- strengthen repeated observations rather than storing duplicate episodes;
- represent support, contradiction, and uncertainty;
- spread activation through only a local neighborhood;
- produce derived beliefs from multi-hop relational overlap;
- retain provenance so conclusions are inspectable;
- save and reload the changed model;
- demonstrate that later inference differs because earlier experience changed the structure.

## Why this is separate from E41

E41 is a controlled experiment asking whether learned persistent relational memory can aid a frozen predictor under ambiguity. This project asks a broader architectural question: can persistent relational structure itself become the continuously changing world model used for cognition?

E41 remains evidence and a possible learned component. It is not modified by this branch.

## Running

```bash
python cognitive_engine/rce_v0.py --demo
```

The demo teaches a tiny world, asks a question, shows the activated subgraph and provenance, then introduces a contradiction and asks again. It also performs a save/reload check.

No external packages are required for v0.

## Deliberate limitations

This is not yet an LLM, a claim of consciousness, or a complete natural-language understanding system. The parser is intentionally tiny and deterministic. Text, vision, audio, and richer generation should later be adapters that map into/out of the same persistent conceptual state.

## Near-term path

1. Validate structural learning and provenance.
2. Add abstraction/compression: repeated relational motifs become reusable higher-order concepts.
3. Add uncertainty-aware contradiction resolution and temporal context.
4. Add a small local language adapter while keeping cognition observable.
5. Add image embeddings/object detections as another input adapter resolving into the same concept nodes.
6. Benchmark memory growth, activation cost, inference accuracy, and forgetting against simple baselines.
