# Neural Compression — Research Log

## Project North Star

> What is the smallest persistent relational structure capable of explaining multiple observations of the same underlying system?

This is the current compressed research question. It may change as evidence changes.

## New Conceptual Note: Null State, Attention, and Distinction

A separate line of thought about geometric representation and game-engine data structures unexpectedly looped back into the project's compression question.

The proposed intuition is to treat **zero as a null state**, rather than merely as another value. A particular state can then be understood as a deviation from that null state: something exists as a state because relationships, constraints, or distinctions make it non-null.

This suggests a sharper formulation of the compression problem:

> **What is the minimum set of relationships that must remain active or persistent to prevent a state from collapsing back into the null state?**

This reframes compression from "how do we encode the object?" toward "what persistent structure is necessary to distinguish the object/state from nothing?"

A related philosophical intuition is that **attention maintains information in an active state**. Information may physically remain stored without attention, so this is not yet a literal claim about information theory. The useful research interpretation is narrower: a system's effective state may depend on which relationships or distinctions are currently represented and maintained. Removing a relationship from the active representation could make that distinction disappear from the system's effective state.

This may connect to the existing North Star because both formulations seek the smallest persistent relational structure needed to preserve identity across observations. It also potentially connects to mathematical questions involving transformations and convergence: rather than assuming every state must literally return to the number one, investigate whether a transformation causes every state to converge to a common relational structure relative to a null/reference state. This is only a hypothesis and should not be treated as a solution to any mathematical problem.

Important provenance: this idea arose while thinking about how geometry could be represented as relationships and generative descriptions in a game engine, not from an experiment in this repository. It is recorded here as a potentially useful conceptual bridge for future research, without retroactively assigning it experimental evidence.

## How to Read This Log

This file is the living research memory. Git history preserves changes in our understanding. Experiment numbers are not the research trajectory; experiments are observations attached to smaller conceptual questions.

Keep separate:

- **Observations:** what an experiment actually showed.
- **Invariants:** patterns that survive changes in observation, traversal, viewpoint, seed, or implementation.
- **Hypotheses:** explanations currently being tested.
- **Rejected explanations:** explanations weakened or ruled out.
- **Open questions:** unresolved uncertainty.
- **Current model:** smallest coherent structural explanation presently supported.
- **Next decision:** experiment that most efficiently separates competing explanations.

When something is surprising or potentially useful later, record it even before its importance is understood.

## Reconstructed Trajectory

The project is best understood as increasingly constrained questions rather than unrelated architectures.

### 1. Can a tiny representation retain useful information?

The initial instinct was broader than simply compressing a model: can a system retain recognizable information about a much larger underlying system while storing dramatically less explicit information?

A recurring idea was that repeated observations of the same underlying thing should not necessarily require independent complete representations.

### 2. What information survives?

Very small latent representations produced organized behavior and geometry. A two-dimensional latent was particularly interesting because it retained structure associated with a much larger system.

The geometry was evidence to investigate, not automatically the object of the research.

### 3. Is the surviving structure persistent?

Across seeds and configurations, organized latent structure appeared repeatedly, although its exact geometry varied. This established that structured information could survive severe compression; it did not establish that latent geometry itself was semantic meaning.

### 4. Is that structure behaviorally meaningful?

Behavioral probes showed that latent geometry and behavior were not related by one universal mapping. Different seeds could produce different relationships, including apparent inversions. This weakened the simple explanation that latent coordinates directly equal semantic meaning.

### 5. Is the apparent structure an artifact of coordinates or observation?

Rotation, viewpoint, and intervention experiments increasingly asked whether a representation describes the underlying system or merely one coordinate system or observation path.

#### Observation basis versus physical rotation

The newer rotation/angle idea should not be interpreted primarily as literal mechanical rotation. The intended variable is a change in **observation basis or observation state**: the same underlying system can produce different observed manifestations under different observation conditions.

Current hypothesis:

> Changing the observation basis can change observed form without changing the underlying system, and combining observations may expose relational information not explicit in an individual observation.

This remains a hypothesis, not an established neural mechanism.

### 6. What persists across observations?

Traversal changes, viewpoint changes, local representations, shared interaction layers, and information-boundary tests increasingly moved the question from what one latent contains to what structure must persist across multiple observations of the same system.

### 7. Current direction: persistent relational structure

The current hypothesis is that the useful compressed object may not be an isolated representation of each observation. It may instead be a small persistent relational structure from which multiple observations can be explained, reconstructed, or predicted.

The target is **minimum reusable structure**, not merely minimum representation size.

## Historical Archaeology: ROCM-test Recovery

On August twenty-six, twenty twenty-six, the repository received commit `7a29f7f` titled **“ROCM-test folder dump.”** This is the historical layer that was missing from the earlier GitHub snapshot. The prior repository state exposed the later experiment sequence beginning around E18; this dump restores the older working material and should now be treated as primary evidence for reconstructing the original trajectory.

The recovered material confirms that the original project had at least two distinct meanings of “compression.”

### Coding compression

The early work included direct neural byte-stream compression. One recovered implementation is a synchronized online neural range compressor: the encoder and decoder initialize the same model state, predict the next byte from a fixed context, range-code the prediction, then both update the model from the observed/decoded byte. The implementation verifies exact reconstruction with SHA-256.

The important conceptual ingredients are:

`observed stream → context → probability model → predicted symbol cost → encode → shared model update`

That branch forces explicit questions about predictability, bit cost, online adaptation, synchronization, and what information is actually reusable.

### The forgotten synchronization insight

The evolution of this code reveals something more important than the codec itself. The model does not need to be transmitted alongside the compressed data if **both sides can deterministically reconstruct the same evolving model state from the same history**. The compressed stream is therefore not merely carrying symbols; it is carrying information that drives two copies of a shared adaptive state machine along the same trajectory.

This is a strong early precursor to the project's later notion of persistent state. The system effectively turns history into state:

`shared initial state + observations → evolving internal state → next prediction`

and the decoder reproduces that state trajectory because it receives the same observations through decoding.

That should be treated as a foundational architectural idea, not merely an implementation trick. It is an early example of **persistent reusable structure emerging through sequential interaction rather than being re-sent from scratch**.

The later versions are explicitly optimizations of the same algorithm rather than new compression methods. This is historically useful because it separates **research evolution** from **implementation evolution**. The core idea stayed stable while the machinery changed around it.

### Representation / model compression

The other branch attempted to retain useful behavior or structure of a larger system using a much smaller latent or parameter representation. This branch eventually led through functional recovery, relational attention, latent geometry, viewpoint/rotation tests, persistent memory, and finally explicit relational structure.

The historical relationship is therefore better represented as:

`raw-data coding → learned predictability → synchronized adaptive state → representation/latent compression → functional recovery → relational attention → latent geometry → observation invariance → persistent structure → relational structure`

The coding branch should not be discarded as obsolete history. It is a concrete precursor to the current question because both branches ask what information is predictable, reusable, and worth preserving.

### Important archaeology rule

Do not retroactively force the current E37 interpretation onto early experiments. First record what the historical experiment actually tested. Then identify which later question it may have informed. An apparently dead-end result may contain a precursor to an idea rediscovered later under different terminology.

### Archaeology targets

As the ROCM-test dump is reviewed, specifically look for:
