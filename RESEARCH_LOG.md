# Neural Compression — Research Log

## Project North Star

**Current research question:**

> What is the smallest persistent relational structure capable of explaining multiple observations of the same underlying system?

This is the current compressed statement of the research direction. It is allowed to change as evidence changes.

## How to Read This Log

This file is the living research memory for the project. Git history preserves earlier versions of the file, so changes in our understanding are themselves part of the record.

The experiment number is not the research trajectory. Experiments are observations and tests attached to a smaller set of conceptual questions.

The project should distinguish between:

- **Observations:** what an experiment actually showed.
- **Invariants:** patterns that appear to survive changes in observation, traversal, viewpoint, seed, or implementation.
- **Hypotheses:** explanations we currently think might account for those observations.
- **Rejected explanations:** ideas that experiments have weakened or ruled out.
- **Open questions:** things we do not yet understand.
- **Current model:** the smallest coherent structural explanation we currently have.
- **Next decision:** the experiment or architectural change that most efficiently distinguishes competing explanations.

When an observation is unusual, surprising, or potentially useful later, record it even when its importance is not yet known.

## Reconstructed Trajectory

The early work can be understood as a sequence of increasingly constrained questions rather than as a sequence of unrelated architectures.

### 1. Can a tiny representation retain useful information?

The initial direction was not simply “compress a model.” The deeper question was whether a system could retain something recognizably about a much larger underlying system while storing dramatically less explicit information.

The recurring instinct was: if the same underlying thing is observed repeatedly, why should every observation require its own complete representation?

### 2. What information survives?

Very small latent representations produced organized behavior and geometry. A two-dimensional latent was especially interesting because it appeared capable of retaining structure associated with a much larger system.

The important interpretation is that the geometry was initially evidence to investigate, not necessarily the object of the research itself.

### 3. Is the surviving structure persistent?

Experiments across seeds and configurations suggested that organized latent structure could appear repeatedly, although its exact geometry varied.

This did not establish that latent geometry was itself “meaning.” It established that something structured appeared to survive severe compression and therefore deserved explanation.

### 4. Is that structure behaviorally meaningful?

Behavioral probes showed that the relationship between latent geometry and behavior was not simple or universal. Different seeds could produce different relationships, including apparent inversions.

This weakened the easy explanation that latent geometry directly equals semantic meaning.

That failure is valuable evidence.

### 5. Is the apparent structure an artifact of coordinates or observation?

Rotation and viewpoint experiments became important because a representation can appear structured while merely encoding one particular coordinate system or observation path.

The deeper question was not simply whether a geometric statistic survives rotation. It was:

> Does the representation describe the underlying system, or merely one way of looking at the system?

This distinction should remain explicit in future experiments.

### 6. What persists across observations?

Traversal changes, viewpoint changes, local representations, shared interaction layers, and information-leak tests increasingly point toward relationships between observations as a central object of study.

The research is therefore moving from asking what an individual latent contains toward asking what structure must persist across multiple observations of the same system.

### 7. Current direction: persistent relational structure

The current hypothesis is that the useful compressed object may not be an isolated representation of each observation at all. It may instead be a small persistent relational structure from which multiple observations can be explained or reconstructed.

The target is not merely a compact encoding. The target is the **minimum reusable structure** that accounts for the observed family without being tied to any particular presentation.

## Recent Evidence: E36–E37

The latest work makes the persistent-structure question concrete rather than purely latent.

### E36 — local observer / adaptive information memory

E36 tests whether a moving observer can build useful persistent knowledge from only locally visible information, rather than being given the whole world. The memory can update existing chunks, create new chunks, and merge compatible chunks. The experiment explicitly compares persistent knowledge cost with raw observation cost and asks whether information topology can emerge from traversal rather than being imposed. fileciteturn11file13

The later E36 relational version strengthens the information boundary: memory receives relative positions, local features, observer motion, and rotation changes, but not absolute world coordinates, world identifiers, or hidden geometry. It retains sparse relations between persistent local anchors and tests reuse across repeated traversal. fileciteturn11file16

This is important because it moves the question from “what shape does a latent space have?” toward “can persistent structure emerge from observations without being handed the underlying coordinate system?”

### E37 — Minimal Persistent Relational Structure

E37 makes the current research question explicit and turns it into an interpretable, non-neural baseline. It separates simulator knowledge from observer input and asks whether structure learned on one traversal can explain a second traversal with a different starting phase or direction. It measures reconstruction, next-observation prediction, structure cost, reuse, ablations, and a relation-budget Pareto curve. fileciteturn11file2

The experiment therefore gives us a concrete way to test the phrase **minimum persistent relational structure** rather than treating it as an architectural metaphor.

### Current interpretation of E36 → E37

E36 asks whether persistent structure can emerge from local observation.

E37 asks whether that persistent structure is sufficient to explain changed observations, and how much relational structure is actually necessary.

That progression is currently more important than the individual implementation details.

## Important Methodological Insight

A recurring failure mode in the project has been mistaking experimental machinery for the research question.

For example, a chain such as:

`geometry → rotation → transformed direction → latent intervention → model output → behavior metric`

can be useful for validating an implementation. But making that chain increasingly airtight does not necessarily answer the larger scientific question.

The project should therefore distinguish:

**“Is the experimental plumbing correct?”**

from

**“What new uncertainty does this experiment remove?”**

A technically clean experiment that does not discriminate between meaningful hypotheses may be lower value than a simpler experiment that directly tests an invariant or destroys a competing explanation.

## Current Conceptual Trajectory

The most useful compressed view of the project so far is:

```text
Can a tiny representation work?
        ↓
What information survives?
        ↓
Is there persistent structure?
        ↓
Is the structure behaviorally meaningful?
        ↓
Which parts are observation / coordinate artifacts?
        ↓
What survives changes in viewpoint / traversal?
        ↓
What relationships persist across observations?
        ↓
What is the minimum structure required?
```

This trajectory should be treated as a working interpretation, not a claim that every historical experiment was consciously designed around these exact questions.

## Architectural Context

The broader architecture is increasingly being evaluated through the same structural principle that has appeared repeatedly in earlier engineering work:

> Build the smallest general mechanism once, then allow many different things to pass through it rather than encoding each case separately.

This is an architectural instinct toward dynamic, reusable, implementation-agnostic structure. In the current research context, that instinct becomes a scientific question: what reusable relational mechanism can explain a family of observations without separately encoding each observation?

The implementation language is secondary to the structure being tested. C#, C++, Python, or another language is an implementation accent, not the conceptual architecture.

## Research Hygiene

Do not silently turn a metaphor into a technical definition. Terminology used in conversation may be an approximate handle for a structural idea. When terminology is ambiguous, reconstruct the operation or relationship being described before correcting the vocabulary.

Do not treat repeated experiments as automatically redundant. A superficially repeated experiment may be testing a different projection, viewpoint, traversal, representation, or invariant.

At the same time, do not repeat an experiment merely because it is familiar. Before rerunning something, identify what uncertainty the new version is intended to remove.

Do not confuse a good reconstruction score with discovery of the underlying structure. Explicitly audit information pathways and ask whether the system could succeed without discovering the intended reusable structure.

## Historical Reconstruction Status

This document begins with a compressed reconstruction rather than a complete catalog of the project's roughly hundred experiments. Earlier conversations, experiment artifacts, and notes should be incorporated gradually when they add durable information.

The goal is not to reproduce every conversation. The goal is to preserve the smallest useful relational structure of the research history: what we observed, what those observations changed, what explanations were rejected, and why the next experiment was chosen.

## Current State

**North star:** Find the smallest persistent relational structure capable of explaining multiple observations of the same underlying system.

**Primary concern:** Distinguish structure belonging to the underlying system from structure belonging only to a particular observation, coordinate system, traversal, seed, or implementation.

**Current architectural direction:** Test whether local representations connected through a shared relational / interaction mechanism provide a better abstraction than treating observations as independent latent objects.

**Recent concrete baseline:** E37 MPRS explicitly tests persistent units plus sparse relations across changed traversals and measures the cost/error tradeoff. fileciteturn11file2

**Highest-value next step:** Interpret the E37 result and its ablations before adding architectural complexity. In particular, determine whether relations provide explanatory power beyond persistent units alone, and identify the smallest tested structure that remains sufficient under changed observation.

**Known danger:** Getting absorbed in the machinery of measurement and losing the original question.

**Living rule:** Every new experiment should state what assumption it is testing, what observation would change our mind, and what part of the persistent structure it is intended to isolate.
