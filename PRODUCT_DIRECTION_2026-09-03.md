# Product Direction — 2026-09-03

**Status:** Strategic assessment. This is not an experimental closeout and does not make a novelty, generalization, or performance claim.

## Executive conclusion

The strongest first product is a **local-first, verifiable relational-memory gateway for local AI**.

It should turn trusted workspace facts into a compact, persistent, provenance-backed relationship graph; resolve direct factual and multi-hop questions deterministically where possible; and otherwise give a local language model only the smallest relevant, cited subgraph.

The immediate user is a developer or researcher running private local-model workflows who needs durable workspace memory without replaying full chat histories or documents on every query.

The product promise is:

> Give local AI durable, inspectable memory with less context, less latency, and evidence for every factual answer.

This recommendation deliberately does **not** make the neural archive compressor the first product.

## Repository evidence

### E37 / E38 form the product spine

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
   /                    \
deterministic graph       compact cited subgraph
answer                    + local LLM explanation
   \                    /
      evidence, confidence, context/latency metrics
```

The existing E38 relation store and local-model adapters can seed this architecture. The E37 relation budget provides a useful future control: retrieval should be bounded and measured rather than indiscriminately stuffing every connected fact into context.

## Validation gates

Do not treat the product direction as validated until it meets all of the following on real user workspaces:

1. At least 60% context reduction versus raw history on the target tasks.
2. No material loss in answer accuracy or factual support quality.
3. Every factual answer has inspectable provenance.
4. Direct graph questions bypass the model reliably.
5. Users can correct or revoke a memory without fighting the system.

Measure accuracy, evidence coverage, context tokens/characters, latency, direct-resolution rate, and correction rate. Keep the raw-history baseline in every evaluation.

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

Productize E38 as a small, deterministic, evidence-first vertical slice before adding neural memory:

`ingest reviewed facts → persist source-backed relations → direct graph answer or compact local-LLM answer → show evidence and savings`

That smallest artifact is both useful and falsifiable. If it does not preserve answer quality while cutting context in real workspaces, the product hypothesis should be revised before expanding the model, graph, or UI.
