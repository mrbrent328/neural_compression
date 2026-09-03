# Product Direction — 2026-09-03

**Status:** Strategic assessment informed by E39. This document is not an experimental closeout and does not make a novelty or broad generalization claim. E39 provides controlled evidence that strengthens the product architecture hypothesis, but real-workspace validation remains outstanding.

## Executive conclusion

The strongest first product is a **local-first, verifiable relational-memory gateway for local AI**.

It should turn trusted workspace facts into a compact, persistent, provenance-backed relationship graph; resolve direct factual and multi-hop questions deterministically where possible; and otherwise give a local language model only the smallest relevant, cited subgraph.

E39 strengthens this direction by showing, in a controlled held-out setting, that the relational representation itself can preserve answerable information after the original observations are hidden. This means the product should be understood not merely as a retrieval optimization, but as a **persistent, inspectable information representation between accumulated observations and the language model**.

The immediate user is a developer or researcher running private local-model workflows who needs durable workspace memory without replaying full chat histories or documents on every query.

The product promise is:

> Give local AI durable, inspectable memory with less context, less latency, and evidence for every factual answer.

This recommendation deliberately does **not** make the neural archive compressor the first product.

## Repository evidence

### E37 / E38 / E39 form the product spine

[E37](artifacts/250826/E37) established the relevant architectural signal in a controlled synthetic setting: retaining relations did not change local reconstruction error, but it materially improved next-observation prediction. The recorded full-relation condition reached pass-two prediction F1 of approximately 0.553; the no-relations ablation was 0. The relation-budget sweep recorded approximately 0.516 F1 with 64 retained relations.

[E38](artifacts/250826/E38) turns that into a direct product question:

> Can a persistent relational structure reduce the context presented to a local language model while preserving answer accuracy?

Its implementation already contains the right primitives:

- a small SQLite-backed relation store;
- compact multi-hop neighborhood retrieval;
- raw-history versus relational-context conditions;
- local llama.cpp and LM Studio inference adapters;
- measurements for answer correctness, prompt size, latency, and memory size.

The E38 harness is therefore the best foundation for a real product. It is intentionally controlled and currently has no tracked results artifact, so it is a product hypothesis rather than proof of user value.

[E39](artifacts/250826/E39) then tests the representation itself rather than the LLM layer:

> How much of an information stream can be replaced by a persistent relational representation while preserving the ability to answer previously unseen questions about that stream?

E39 builds a relational representation from source observations, hides the original observations from the evaluated condition, asks held-out direct and two-hop questions, and progressively removes relational facts. The full representation answered all 144 held-out questions correctly with explicit supporting evidence chains. At 75% of facts retained, accuracy remained 72.22%; at 50%, it fell to 45.14%; at 25%, 19.44%; and at 12.5%, 9.72%.

The important product implication is not the exact synthetic percentages. It is that **the persistent relational representation retained useful answering capability after the original observations were removed, and degradation followed information removal rather than hallucinated answers**. When required relations were absent, the deterministic system returned no answer and no supporting chain. This makes inspectability and bounded failure a core architectural property rather than an afterthought.

E39 also suggests that useful information is not uniformly distributed across stored facts. The capability curve falls substantially as the relation budget is reduced, which supports treating structured memory as a selective representation of what deserves to persist rather than indiscriminately storing or replaying everything.

### S6E2 contributes useful concepts, not a shippable base

The [S6E2 app](memory_network/app) contributes durable observations, persistent entities, mutable relational context, and a simple CLI. Those are valuable design inputs.

It is not ready to become the application unchanged:

- the documented repository-root command is blocked by the top-level `memory_network.py` file shadowing the `memory_network/` application directory;
- there is no dependency or package manifest;
- the data model does not retain source provenance, claim lifecycle, conflicts, or review state;
- its learned ranking scores are not calibrated evidence and should not determine factual answers.

Treat its neural components as future experiments or optional ranking signals, never as the sole source of truth in the first release.

### NCA remains research, not the first commercial direction

[NCA-01](NCA-01_local_adaptive_archive.py) is a credible exact-round-trip archive prototype. The recorded baseline run reports a 63.1376% reduction on a 3.18 MB Calgary corpus and exact reconstruction verification.

However, it is not a product-ready compression engine:

- the reported run took about 42 minutes;
- the adaptive selector chooses a strategy from structural rules, then only falls back when it expands against raw data; it does not compare the selected neural candidate against the classical candidate;
- NCA-02 produced a worse recorded compression result despite its stronger neural configuration;
- portability depends on deterministic online training and codec reproducibility.

Keep the archive line alive as research. Its useful product lesson is **adaptive selection of what deserves structured memory**, not an immediate replacement for conventional compressors.

## Product: Local Relational Memory Gateway

### User-facing behavior

1. A user imports notes, Markdown, transcripts, or structured records.
2. The system proposes facts with exact source spans; users approve, edit, reject, or expire them.
3. Facts are stored locally in a versioned relational graph.
4. On a question, the router first tries deterministic graph resolution.
5. If explanation or synthesis is needed, it sends a bounded, source-cited subgraph to the local model.
6. The answer identifies whether it was deterministically resolved or model-generated, shows supporting evidence, and exposes the context reduction achieved.

### First-release requirements

- Local SQLite storage by default.
- OpenAI-compatible proxy/API for local model hosts.
- Markdown/text ingestion plus a manual structured-fact editor.
- Provenance for every claim: source, span, timestamp, scope, confidence, and lifecycle state.
- Direct and two-hop graph queries.
- A review inbox so untrusted extraction cannot silently mutate memory.
- An answer view with supporting facts and source excerpts.
- A benchmark mode that compares raw-document context with relational retrieval on the same question set.

### Explicit non-goals for version one

- No autonomous, unreviewed memory writes.
- No claim of semantic understanding or “self-improving” cognition.
- No neural score as the authoritative answer mechanism.
- No general-purpose file compressor.
- No broad enterprise knowledge graph before the local workflow is useful.

## Technical shape

```
documents / chats
        |
candidate fact extraction + user review
        |
versioned local fact graph (SQLite + provenance)
        |
question router
   /                    \\
deterministic graph       compact cited subgraph
answer                    + local LLM explanation
   \\                    /
      evidence, confidence, context/latency metrics
```

The existing E38 relation store and local-model adapters can seed this architecture. E39 adds a more fundamental design principle: the graph should be treated as a persistent representation in its own right, with deterministic answering and explicit evidence chains where possible. The E37 and E39 relation-budget results provide useful future controls: retrieval and storage should be bounded and measured rather than indiscriminately stuffing every connected fact into context.

## Validation gates

Do not treat the product direction as validated until it meets all of the following on real user workspaces:

1. At least 60% context reduction versus raw history on the target tasks.
2. No material loss in answer accuracy or factual support quality.
3. Every factual answer has inspectable provenance.
4. Direct graph questions bypass the model reliably.
5. Users can correct or revoke a memory without fighting the system.

Measure accuracy, evidence coverage, context tokens/characters, latency, direct-resolution rate, and correction rate. Keep the raw-history baseline in every evaluation.

E39 should be treated as architectural evidence, not as satisfaction of these product gates. Its controlled synthetic corpus does not establish performance on real user workspaces.

## Market posture

“AI memory” is already an established category: Mem0 presents a general memory layer, and Letta presents a platform for stateful agents. The differentiated wedge here is narrower:

> Auditable, local-first relational knowledge for people who run local models and need to know exactly what their AI remembers and why it answered.

Local inference integration is practical because both LM Studio and llama.cpp expose developer-oriented local APIs.

Related context, not evidence from this repository:

- [Mem0](https://docs.mem0.ai/introduction)
- [Letta](https://docs.letta.com/)
- [LM Studio developer documentation](https://lmstudio.ai/docs/developer)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Recommended next build decision

Productize the E38 architecture as a small, deterministic, evidence-first vertical slice, now informed by E39's representation result:

`ingest reviewed facts → persist source-backed relations → direct graph answer or compact local-LLM answer → show evidence and savings`

The key product primitive is therefore not simply “retrieval.” It is a **persistent relational memory representation with bounded, inspectable access to a local model**.

That smallest artifact is both useful and falsifiable. If it does not preserve answer quality while cutting context in real workspaces, the product hypothesis should be revised before expanding the model, graph, or UI.
