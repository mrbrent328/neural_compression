# Neural Compression — Research Log

## Project North Star

> What is the smallest persistent relational structure capable of explaining multiple observations of the same underlying system?

This is the current compressed research question. It may change as evidence changes.

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

The early work included direct neural byte-stream compression. One recovered implementation is a batch-aware online neural compressor using a byte vocabulary, finite context windows, learned next-byte probabilities, explicit conversion of negative log probabilities into estimated bit cost, GPU-oriented minibatching, and online gradient updates. The same historical code compares the neural estimate against conventional compressors including gzip and zstandard. This is a genuine information-theoretic compression branch, not merely a latent-size experiment.

The important conceptual ingredients are:

`observed stream → context → probability model → predicted symbol cost → adaptive update`

That branch forces explicit questions about predictability, bit cost, online adaptation, and what information is actually reusable.

### Representation / model compression

The other branch attempted to retain useful behavior or structure of a larger system using a much smaller latent or parameter representation. This branch eventually led through functional recovery, relational attention, latent geometry, viewpoint/rotation tests, persistent memory, and finally explicit relational structure.

The historical relationship is therefore better represented as:

`raw-data coding → learned predictability → representation/latent compression → functional recovery → relational attention → latent geometry → observation invariance → persistent structure → relational structure`

The coding branch should not be discarded as obsolete history. It is a concrete precursor to the current question because both branches ask what information is predictable, reusable, and worth preserving.

### Important archaeology rule

Do not retroactively force the current E37 interpretation onto early experiments. First record what the historical experiment actually tested. Then identify which later question it may have informed. An apparently dead-end result may contain a precursor to an idea rediscovered later under different terminology.

### Archaeology targets

As the ROCM-test dump is reviewed, specifically look for:

- forgotten hypotheses that later reappeared;
- early distinctions between storage/coding cost and representation cost;
- places where decoder/model cost was ignored or explicitly counted;
- online or adaptive learning mechanisms;
- observations about reusable information;
- early information-leak or boundary concerns;
- evidence that multiple observations were being treated as views of one underlying system;
- any “holy shit, come back to this later” result that was never incorporated into the main trajectory.

The goal is not to reproduce every historical file. The goal is to recover durable relationships between experiments, observations, and changed hypotheses.

## E18–E22: Functional and Relational Turn

E18 explicitly compared optimizing a held-out latent for factor/weight MSE against optimizing it for the function of the recovered Q projection. The constructor remained fixed, so the experiment isolated the effect of the latent objective.

E19 then pushed the objective into **relational attention**, comparing candidate Q behavior against Q/K/V attention context rather than treating the target weight only as an isolated tensor. E20 continued this as a confirmation experiment, while E21 and E22 explored extreme latent capacity and relational capacity.

This is an important historical precursor to the present relational view: the project began asking whether preserving a component's **functional relationships** could matter more than minimizing raw parameter error before the later E36/E37 persistent-memory experiments made relations explicit.

## E23–E30: Latent Geometry Investigation

E23–E30 moved into L2 latent landscapes, coordinate geometry, eigenvector sweeps, cross-seed replication, geometry/behavior sensitivity, rotated geometry, principal directions, rotation invariance, and seed-regime clustering.

The durable conclusion is not that a particular latent coordinate system was discovered. Rather, these experiments established that structured geometry could be real and reproducible enough to investigate, while its exact orientation and behavioral interpretation could vary across seeds. That uncertainty motivated the later shift toward invariants and observation dependence.

## E36 — Local Observer / Adaptive Information Memory

E36 tests whether a moving observer can build useful persistent knowledge from only locally visible information instead of receiving the whole world. Memory can update existing chunks, create new chunks, and merge compatible chunks. The experiment compares persistent knowledge cost with raw observation cost and asks whether information topology can emerge from traversal rather than being imposed.

The later relational E36 version strengthens the boundary: memory receives relative positions, local features, observer motion, and rotation changes, but not absolute world coordinates, world identifiers, or hidden geometry. It retains sparse relations between persistent local anchors and tests reuse across repeated traversal.

The conceptual transition is:

`latent geometry question → can persistent structure emerge from local observation?`

## E37 — Minimal Persistent Relational Structure

E37 makes the current question explicit in an interpretable, non-neural baseline. It separates simulator knowledge from observer input and asks whether structure learned on one traversal can explain a second traversal with a different starting phase or direction.

It measures reconstruction, next-observation prediction, structure cost, reuse, ablations, and a relation-budget frontier.

The important progression is:

`E36: can persistent structure emerge?`

`E37: is persistent structure sufficient, and how much relational structure is necessary?`

The E37 result currently suggests that persistent units can preserve reconstructive information while relations provide the missing predictive structure. This should remain framed as evidence within the tested representation and cost model, not as proof of a universal minimum.

## Important Methodological Insight

A recurring failure mode is mistaking experimental machinery for the research question.

For example:

`geometry → rotation → transformed direction → latent intervention → model output → behavior metric`

may validate plumbing without resolving the scientific question.

Always distinguish:

**Is the experimental plumbing correct?**

from:

**What new uncertainty does this experiment remove?**

A technically clean experiment that does not discriminate between hypotheses may be less valuable than a simpler experiment that destroys a competing explanation.

## Current Conceptual Trajectory

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
What changes when the observation basis changes?
        ↓
What survives changes in viewpoint / traversal?
        ↓
What relationships persist across observations?
        ↓
What information appears only through combinations of observations?
        ↓
What is the minimum structure required?
```

This is a working reconstruction, not a claim that every historical experiment was consciously designed around these exact questions.

## Architectural Context

A recurring engineering instinct in the project is:

> Build the smallest general mechanism once, then allow many different things to pass through it rather than encoding each case separately.

In the research context this becomes a scientific question: what reusable relational mechanism can explain a family of observations without separately encoding each observation?

Implementation language is secondary. Python, C#, C++, or another language is an implementation detail, not the conceptual architecture.

## Research Hygiene

Do not silently turn a metaphor into a technical definition. Reconstruct the operation or relationship being described before correcting vocabulary.

Do not treat repeated experiments as automatically redundant. A repeated-looking experiment may test a different projection, viewpoint, traversal, representation, observation basis, or invariant.

Do not rerun an experiment merely because it is familiar. State what uncertainty the new version removes.

Do not confuse a good reconstruction score with discovery of the underlying structure. Audit information pathways and ask whether the system could succeed without discovering the intended reusable structure.

When using physical analogies, preserve the operational mapping. Mechanical rotation should not be substituted for a change in observation basis if the latter is the actual research variable.

## Historical Reconstruction Status

The log is a compressed research memory, not a catalog of every file. The goal is to preserve the smallest useful relational structure of the research history: what we observed, what changed because of it, what explanations were rejected, and why the next experiment was chosen.

The ROCM-test dump is now explicitly part of the reconstruction process. Historical interpretation should proceed from recovered evidence rather than memory of the later project state.

## Current State

**North star:** Find the smallest persistent relational structure capable of explaining multiple observations of the same underlying system.

**Primary concern:** Distinguish structure belonging to the underlying system from structure belonging only to a particular observation, coordinate system, traversal, seed, or implementation.

**Recent conceptual refinement:** Treat changes in observation basis/state as distinct from literal physical rotation. Test whether different observation states can yield different forms while sharing compact underlying relational structure, and whether mixed observations reveal relational information unavailable to individual observations.

**Historical refinement:** The original compression objective had both a coding-compression branch and a representation-compression branch. The coding branch is an important precursor because it makes predictability, bit cost, online adaptation, and reusable information explicit.

**Current architectural direction:** Test whether local representations connected through a shared relational / interaction mechanism provide a better abstraction than treating observations as independent latent objects.

**Recent concrete baseline:** E37 MPRS explicitly tests persistent units plus sparse relations across changed traversals and measures cost/error tradeoffs.

**Highest-value next step:** Continue historical archaeology through the ROCM-test dump before adding unnecessary architectural complexity. Identify early observations that later reappeared as latent geometry, functional recovery, relational attention, persistent memory, observation-basis changes, or information-boundary concerns.

**Known danger:** Getting absorbed in measurement machinery and losing the original question.

**Living rule:** Every new experiment should state what assumption it tests, what observation would change our mind, and what part of the persistent structure it is intended to isolate.
