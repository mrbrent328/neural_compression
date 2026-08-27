# Research Log Addendum — 2026-08-27

## Decision: Stop Asking Only Questions; Build the Reusable Mechanism

After reviewing the recovered ROCM-test history, E14B, E18–E30, E36, and E37, the current assessment changed from “continue isolating the phenomenon” to “we have enough evidence to build the mechanism suggested by the evidence.”

This is not a claim of novelty, a breakthrough, or a universal theory. It is a change in research posture: the project has accumulated enough convergent observations that another round of increasingly similar diagnostic experiments risks producing diminishing returns without giving us a usable artifact.

### What the accumulated evidence now supports

The project repeatedly encountered the same structural pattern through different implementations:

`observation → compact local representation → persistence → reuse → relationships → prediction/reconstruction`

The early neural compressor demonstrated a particularly concrete version of persistence: encoder and decoder can maintain synchronized adaptive state because both reconstruct the same state trajectory from shared history.

E14B demonstrated the instinct to represent a large system as addressable local pieces rather than one monolithic representation.

The functional-latent experiments showed that strong behavioral reconstruction can coexist with poor parameter-level similarity, weakening the assumption that useful compression must preserve the original representation directly.

E36 made persistent local units and relations explicit under an information boundary.

E37 then supplied the strongest current evidence for the relational component: in its tested synthetic environment, reconstruction remained essentially unchanged while prediction improved as relational structure was retained. The reported relation-budget curve rose from zero prediction F1 with no relations to approximately 0.52 at sixty-four retained relations and approximately 0.61 at one hundred twenty-eight, while reconstruction error remained approximately 0.172. The full relational condition also achieved approximately 0.55 prediction F1 while the no-relations condition achieved zero.

These results do not prove that relations are universally necessary, nor that the tested number of relations is a true global minimum. They do establish that, in the tested setup, relational information carried predictive information that local reconstruction alone did not.

### New working hypothesis

> A compact persistent representation may be more useful when it preserves reusable relationships between local observations than when it independently preserves each observation.

This is now considered sufficiently supported to justify implementation as a reusable system component.

### New implementation direction

Build a minimal, inspectable persistent relational memory system rather than immediately adding another large neural experiment.

The first implementation should:

`observe → identify local unit → persist/update unit → record sparse relation → predict → compare → update`

The observer must not receive world IDs, absolute world coordinates, hidden geometry, or future observations. The simulator may know those things for evaluation, but the memory must not.

The first useful artifact should be small enough to inspect and run quickly. It should create persistent units from local observations, reuse units across a changed observation/traversal, maintain a sparse relational graph, and make predictions from that graph.

### Why build now

The research has reached a point where an artifact can provide more information than another isolated sweep. A reusable mechanism gives us something that can be placed in different environments, visualized, instrumented, and compared against simpler alternatives.

The goal is therefore no longer “prove the hypothesis before building anything.” The goal is “build the smallest falsifiable implementation of the hypothesis and let its behavior generate the next questions.”

### Guardrail

The artifact is not itself evidence that the hypothesis is correct. It is a testable implementation of the current model.

If it fails, inspect which assumption failed.

If it works only on the toy world, record that limitation.

If it generalizes to substantially different observations while maintaining a compact persistent structure, that becomes materially more interesting.

If it produces behavior we did not design for, preserve the result before explaining it.

### Current research posture

We have enough evidence to build.

We do not yet have enough evidence to claim universality.

That distinction is deliberate.
