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

- forgotten hypotheses that later reappeared;
- early distinctions between storage/coding cost and representation cost;
- places where decoder/model cost was ignored or explicitly counted;
- online or adaptive learning mechanisms;
- observations about reusable information;
- early information-leak or boundary concerns;
- evidence that multiple observations were being treated as views of one underlying system;
- any “holy shit, come back to this later” result that was never incorporated into the main trajectory.

The goal is not to reproduce every historical file. The goal is to recover durable relationships between experiments, observations, and changed hypotheses.

## Foundational Historical Branch: E14B / test14b.py

**E14B has now been explicitly archived in the repository as `ROCM-test/test14b.py`.** It should be treated as a foundational implementation, not merely an old script. The source converts tensors from a GGUF model into a tiled latent representation using a convolutional encoder, stores latent grids in a binary file, and records tensor-level metadata and byte offsets in a JSON index.

The core pipeline is:

`GGUF tensor → normalize/reshape → 256×256 tiles → convolutional encoder → latent grid → binary storage + index`

Several details matter historically:

- Large tensors are spatialized into 256×256 tiles rather than treated only as flat parameter vectors.
- The encoder reduces each 256×256 tile to a `16 × 32 × 32` latent grid. At the raw float32 tensor level this is a nominal **16× reduction in stored values per encoded tile**, before accounting for the encoder itself, decoder, metadata, and any subsequent quantization/entropy coding.
- Small and one-dimensional tensors bypass the latent path and are stored uncompressed.
- The JSON index preserves original shape, flattened shape, padded shape, tile count, latent shape, byte offset, and byte length.
- The implementation is therefore already thinking in terms of **structured local pieces plus an addressable persistent index**, rather than one monolithic compressed vector.

### E14B — the important thing we missed

The most important historical insight is not the nominal compression ratio. It is the **choice to turn a large model into a collection of spatially addressable local representations**.

That is an early ancestor of the current local-unit/relational view:

`large system → local units → compact representation → persistent index`

The later E36/E37 work makes the same structural move more explicitly: persistent units become reusable objects, and relations are added between them. E14B therefore appears, in retrospect, to be an early engineering precursor to the idea that a large system may be better represented as **many reusable local pieces with explicit structure between them** rather than as one compressed monolith.

There is also an important methodological warning in E14B. The script demonstrates a representation pipeline, not by itself a validated lossless or behavior-preserving compression result. The printed `LATENT` ratio is a raw tensor-value reduction, not a complete end-to-end compression ratio. Encoder/decoder cost and reconstruction quality must be accounted for before calling it genuine compression.

### E14B archaeological status

**Status: foundational precursor; not a validated compression result.**

**Recovered idea worth preserving:** local compact representations plus explicit addressing/indexing.

**Potentially missed branch:** learn or test relations *between* latent tiles, and evaluate functional reconstruction rather than raw weight reconstruction.

**Current relevance:** high. E14B may be the earliest concrete implementation in the recovered archive of the architectural instinct that later became “local representations + shared relational layer.”

## Discover Law Branch — Scientific Law Discovery

The recovered `Discover_Law_2.py` and `Discover_Law_3.py` establish a separate but highly relevant branch of the project. These experiments are **symbolic regression, not neural-network law discovery**.

Discover Law Two creates noisy observations from a hidden generating relationship, gives the discovery engine only the observed variables and outputs, constructs a predefined mathematical feature library, searches combinations of candidate terms, fits coefficients on training observations, evaluates on held-out observations, and finally tests the selected relationship on fresh data before revealing the hidden law.

Discover Law Three extends this direction with explicit complexity-aware scoring. It searches equations with different term counts, evaluates several complexity penalties, compares prediction error against structural complexity, and evaluates the selected model on fresh noiseless data. The hidden law is not provided to the discovery engine.

The key historical progression is:

`observe data → search explanatory structure → validate on unseen data → prefer simpler explanation → reuse the discovered generator`

This branch is important because it demonstrates that the project's objective is separable from neural networks. A neural network is an implementation candidate, not a requirement of the research question.

### Discover Law — conceptual significance

The important question is not merely “can we fit the data?” It is:

> **Can observations reveal a compact reusable structure that generates or explains those observations?**

This creates a third distinction alongside raw storage and predictive modeling:

`raw storage → remember observations`

`predictive model → remember enough structure to predict observations`

`generative structure → discover a compact rule/relationship that can produce or explain observations`

Discover Law Two/Three primarily test the third category in a restricted mathematical hypothesis space. They do **not** establish arbitrary law discovery from an unrestricted space, and they do not establish that a neural network is necessary.

This branch should remain central rather than being treated as a side experiment that was “to revisit later.” It is a historical bridge between compression, prediction, and explanatory structure.

## Representation Similarity versus Reconstructive Sufficiency

A major conceptual refinement emerged during archaeology and discussion of earlier results:

> **A system may achieve exact reconstruction without maintaining a bitwise- or parameterwise-faithful internal representation. What matters may be whether the distributed representation preserves sufficient constraints to uniquely reconstruct the target.**

A discrete output can be reproduced exactly even when its internal representation does not resemble the output's surface form. A human recalling a phone number is an intuitive example: the reproduced sequence can be exact without implying a literal binary copy stored at one location.

This distinction should be applied when revisiting historical Neural Compression results. Some experiments may have produced unexpectedly good prediction or reconstruction despite poor representation-level similarity metrics. Such cases should not automatically be treated as failures. They may indicate that the representation preserved the distinctions or relationships necessary for the behavior without preserving the target's surface form.

Therefore, explicitly search the historical archive for:

`poor parameter/latent similarity + unexpectedly strong functional prediction or reconstruction`

Such cases may be evidence that a metric measured **representation similarity** rather than **information sufficiency**.

This is a research hypothesis, not a claim about a specific biological mechanism.

### Consequence for compression

Compression does not necessarily require preserving the shape of information. A more general framing is:

> **Compression requires preserving enough information to distinguish and reconstruct what matters.**

This suggests future metrics around **reconstructive sufficiency**, **constraint preservation**, or related information-boundary measures rather than relying only on parameterwise or latentwise similarity.

## Constraint-Based and Bidirectional Reconstruction Hypothesis

A further hypothesis emerged from considering interrupted human thought and missing intermediate states.

Instead of only predicting forward:

`A → B → ? → D`

an uncertain state might be reconstructed by satisfying constraints from both directions:

`A → B → ? ← D`

The candidate state is then the state that makes both neighboring relationships plausible. This is a **bidirectional constraint-propagation hypothesis**, not yet an experimental result.

This may be relevant because it changes the problem from predicting a missing value directly toward finding the smallest state that satisfies the relationships imposed by surrounding observations.

## Generalized Data Discovery Hypothesis

The Discover Law experiments raise a broader question: if symbolic regression can discover a compact generator for data with a known mathematical vocabulary, can the same principle be generalized to arbitrary data types?

The proposed direction is not to assume a neural network. Instead, construct a sufficiently general hypothesis space of transformations and relationships and search for the smallest reusable structure that predicts or reconstructs held-out observations.

For time-series data, candidate structure might include lags, differences, ratios, periodicity, nonlinear transforms, cross-channel relationships, state transitions, piecewise rules, or other data-type-specific operations. For other data types, the hypothesis space would change.

The core question is:

> **Can a generalized system discover the smallest reusable computational structure capable of generating, reconstructing, or predicting observations of a system?**

A possible representation of such a discovered structure is not necessarily one equation. It could be a small program, graph, relational dynamical system, symbolic rule set, neural representation, or hybrid. The research objective should determine the mechanism rather than assuming the mechanism in advance.

This yields a useful architectural principle:

> **Representation-discovery mechanism is not part of the North Star.**

Symbolic methods, neural networks, graphs, programs, dynamical systems, or hybrids should be treated as competing hypothesis-space mechanisms. A neural network is justified only when the structure being sought requires the expressive capacity or search behavior it provides.

## Context, Information Leakage, and the Reconstruction Boundary

A further insight emerged from considering generalized prediction in humans and artificial systems: an observer may have access to far more information than is explicitly labeled as “input.” Surrounding context can strongly constrain an unobserved variable.

Therefore:

> **The apparent information available to an observer may be much larger than the explicitly measured input because surrounding structure can implicitly encode constraints on the target.**

This does not mean a predictor can recover arbitrary missing information. If the hidden variable is genuinely independent of the observed context, there is no structural route to reconstruct it. Prediction can recover information that is sufficiently constrained by the observable structure; it cannot conjure independent information from nothing.

This makes **information boundary** a fundamental research concern rather than merely an implementation detail:

`observable context → implicit constraints → reconstructable information`

versus

`observable context ⟂ hidden variable → no reconstructive path`

A useful future experiment would deliberately hide part of a dataset, progressively remove contextual structure, and measure reconstruction quality. The result would define a **reconstruction frontier**: how much surrounding structure must persist before the hidden information becomes reliably recoverable?

This directly connects to E36/E37 information-boundary concerns and to the original compression objective.

## Minimum Retained Structure / Reconstruction Frontier

A particularly clean formulation emerging from the combined archaeology is:

> **What is the minimum surrounding structure that still uniquely determines the data we care about?**

Conceptually:

`more context retained → better reconstruction → threshold → sufficient persistent structure → reconstruction failure`

The retained structure does not have to contain the original data directly. It may contain relationships, constraints, or a generator that makes the original data recoverable.

This may provide a more general experimental framework than “compression ratio.” Instead of only measuring how many bytes were stored, measure how much persistent structure is required to maintain a specified reconstruction or prediction guarantee.

## Research Process as an Analogy

The project itself has become an example of the problem. The raw experiments are observations; the research log preserves relationships between observations, hypotheses, decisions, and changed interpretations. When a new observation changes the interpretation of an old one, the goal is to update the persistent relational structure rather than discard the old observation.

This is useful as an analogy, not as evidence that the research process literally implements the same mechanism as a biological or artificial observer.

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
Can a compact structure generate or explain observations?
        ↓
What contextual information is implicitly available?
        ↓
What is the minimum retained structure needed for reconstruction?
        ↓
What is the minimum structure required?
```

This is a working reconstruction, not a claim that every historical experiment was consciously designed around these exact questions.

## Architectural Context

A recurring engineering instinct in the project is:

> Build the smallest general mechanism once, then allow many different things to pass through it rather than encoding each case separately.

In the research context this becomes a scientific question: what reusable mechanism can explain a family of observations without separately encoding each observation?

Implementation language is secondary. Python, C#, C++, or another language is an implementation detail, not the conceptual architecture.

More strongly, **the discovery mechanism itself should not be assumed in advance**. Symbolic regression, neural networks, graphs, programs, dynamical systems, or hybrids are candidate mechanisms for searching or representing structure. The research question is the structure that survives, not the technology used to find it.

## Research Hygiene

Do not silently turn a metaphor into a technical definition. Reconstruct the operation or relationship being described before correcting vocabulary.

Do not treat repeated experiments as automatically redundant. A repeated-looking experiment may test a different projection, viewpoint, traversal, representation, observation basis, or invariant.

Do not rerun an experiment merely because it is familiar. State what uncertainty the new version removes.

Do not confuse a good reconstruction score with discovery of the underlying structure. Audit information pathways and ask whether the system could succeed without discovering the intended reusable structure.

Do not confuse representation similarity with information sufficiency. A representation that looks numerically unlike the target may still contain sufficient constraints for exact reconstruction.

When using physical analogies, preserve the operational mapping. Mechanical rotation should not be substituted for a change in observation basis if the latter is the actual research variable.

When evaluating “missing information,” audit contextual leakage first. A hidden value may be inferable because its generating structure is present in the observed context.

## Historical Reconstruction Status

The log is a compressed research memory, not a catalog of every file. The goal is to preserve the smallest useful relational structure of the research history: what we observed, what changed because of it, what explanations were rejected, and why the next experiment was chosen.

The ROCM-test dump is now explicitly part of the reconstruction process. Historical interpretation should proceed from recovered evidence rather than memory of the later project state.

## Current State

**North star:** Find the smallest persistent relational structure capable of explaining multiple observations of the same underlying system.

**Primary concern:** Distinguish structure belonging to the underlying system from structure belonging only to a particular observation, coordinate system, traversal, seed, or implementation.

**Representation principle:** Exact or strong reconstruction does not require a bitwise- or parameterwise-faithful internal representation. The important quantity may be whether persistent structure preserves sufficient constraints for the required reconstruction or prediction.

**Discovery principle:** The system does not need to be neural. Neural networks are one candidate mechanism among symbolic search, programs, graphs, dynamical systems, and hybrids. The smallest sufficient mechanism should be discovered rather than assumed.

**Historical law-discovery branch:** Discover Law Two and Three show that, within a predefined mathematical hypothesis space, observations can support discovery of compact explanatory relationships that generalize to held-out and fresh data. This branch is a central precursor to the question of representing a system by its generator rather than its observations.

**Context principle:** Observable context may contain implicit constraints that make otherwise unobserved information reconstructable. This is not equivalent to arbitrary information recovery; genuinely independent hidden information remains unavailable.

**New research question:** What is the minimum surrounding/persistent structure that still uniquely determines or reliably reconstructs the data we care about?

**Potential experimental framework:** Hide part of a dataset, progressively remove contextual structure, and measure the reconstruction frontier. Compare different discovered structure classes—symbolic rules, relational graphs, programs, neural representations, and hybrids—by persistent cost versus reconstruction/prediction performance.

**Current working model:** A useful compressed representation may consist of local persistent units plus a sparse relational structure that preserves the constraints needed to explain and predict multiple observations. The representation need not resemble the observations or original parameters directly.

**Next decision:** Finish the historical archaeology, especially early cases where poor representation-level similarity coexisted with strong prediction/reconstruction, and then determine whether the generalized reconstruction problem can be expressed in a non-neural symbolic/relational hypothesis space before adding neural machinery.

**Known danger:** Getting absorbed in measurement machinery and losing the original question.

**Living rule:** Every new experiment should state what assumption it tests, what observation would change our mind, and what part of the persistent structure it is intended to isolate.

## GSA-01 — Generation Sampling Audit (Implemented; Awaiting Run)

**Question:** Are the repetitive/collapsed continuations observed in the latent and low-rank checkpoints primarily a greedy-decoding artifact, or do the underlying next-byte distributions themselves show limited useful diversity?

**Hypothesis under test:** Sampling may reveal plausible continuations that greedy argmax decoding suppresses. If sampling increases diversity without improving validity or continuation quality, that will distinguish distributional uncertainty from useful language behavior rather than count as a model-quality improvement.

**Controls and protocol:**
- Reuse the existing dense, low-rank, and latent checkpoints without training or modifying them.
- Reuse the five fixed diagnostic prompts and four held-out prompts from the existing generation audit.
- Keep prompt bytes and generated-byte budgets matched across models.
- Compare one greedy baseline with temperature 0.7, 0.9, and 1.0, and top-p 0.90 and 0.95.
- Use three deterministic seeds per prompt and sampling configuration. Seed derivation is stable and shared across model comparisons.
- Check the custom greedy decoder against the model's existing generate method before the audit.
- For the latent model, enable the existing all-factor inference cache only after confirming bit-identical cached and uncached logits.
- Record byte diversity, repeated trigrams, whitespace/printability, UTF-8 validity, raw and filtered entropy, top-1 probabilities, and cross-seed byte agreement.

**Implementation:**
- generation_sampling_audit.py
- test_generation_sampling_audit.py
- GENERATION_SAMPLING_AUDIT_README.md

**Output paths:** Timestamped JSON and TXT reports under runs/generation_audit/, with the prefix generation_sampling_audit_.

**Status:** IMPLEMENTED AND COMMITTED; awaiting the user's local run. No experimental result has been measured yet.

**What this experiment does not establish:** More varied samples do not necessarily mean more accurate, coherent, or useful language behavior. This remains a small qualitative/diagnostic study on a source-code-heavy corpus, not a general language benchmark or a training/scaling comparison.

**Next action:** Run the focused unit tests, then execute the sampling audit on the local ROCm system and preserve the resulting JSON/TXT artifacts in GitHub. Analyze model and strategy differences before deciding whether decoding, data, or representation is the next bottleneck.



## GSA-01 — Generation Sampling Audit (Run complete; interpretation recorded 2026-11-17)

**Run artifact:** `runs/generation_audit/generation_sampling_audit_20261117_174813_476347.json` and matching `.txt` report. The JSON records creation time `2026-11-17T17:44:42`. The result commit is `7e736d1a62ea82f7b92fc6d7e8ba5addab213787`.

**Execution/validation evidence:** 3 models × 9 prompts × (1 greedy + 5 sampling strategies × 3 seeds) = 432 generation cases, 128 bytes each. The run used device `cuda` and PyTorch `2.13.0+rocm7.1`. All three checkpoints loaded strictly; each custom greedy decoder passed its 8-token parity check; the latent inference factor cache passed bit-identical cached/uncached logits validation. The report is read-only. This is evidence that the audit executed; a separate captured unit-test log is not present in the repository, so unit-test execution remains unverified.

**Aggregate repeated-trigram rate (lower is less repetition):**

| Model | Greedy | T=0.7 | T=0.9 | T=1.0 | Top-p .90 | Top-p .95 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.392 | 0.253 | 0.235 | 0.200 | 0.254 | 0.237 |
| Low-rank | 0.882 | 0.465 | 0.278 | 0.214 | 0.388 | 0.247 |
| Latent | 0.862 | 0.387 | 0.302 | 0.186 | 0.313 | 0.277 |

At temperature 1.0, mean unique-byte fraction was 0.271 dense, 0.261 low-rank, and 0.261 latent; distinct-bigram fraction was 0.669, 0.653, and 0.678 respectively. Temperature 1.0 UTF-8 validity was 24/27 dense, 24/27 low-rank, and 25/27 latent. Top-p .90 and .95 yielded 27/27 UTF-8-valid cases for dense and latent; low-rank had 27/27 at .90 and 26/27 at .95. These are format/diversity diagnostics, not language-quality scores.

**Qualitative held-out inspection:** All four held-out prompts failed to produce a faithful continuation across the three models. Greedy low-rank and latent generations visibly collapsed into repeated fragments or whitespace on several prompts (e.g. low-rank heldout_4 is all spaces; latent heldout_4 is all spaces; latent heldout_3 repeats `null`). Sampling breaks up these loops, but sampled continuations are generally malformed mixtures of code-like tokens, identifiers, syntax fragments, and unrelated material. Dense is less collapsed, but its held-out continuations are also semantically off-target. The held-out reference continuations are drawn from source-code corpus excerpts; no exact continuation recovery or semantic correctness benchmark is implemented here.

**Interpretation:** GSA-01 supports the claim that greedy argmax decoding materially amplifies repetitive collapse in low-rank and latent checkpoints. It does **not** support the stronger claim that useful language behavior was merely hidden by greedy decoding. Sampling exposes a broader, higher-entropy set of next-byte choices, but the qualitative outputs do not show reliable continuation competence. Thus the current evidence points to a combination: a decoding artifact explains part of the visible repetition, while underlying learned distributions remain poorly calibrated/structured for robust continuation on these prompts.

**Important caveats:** (1) This is a small diagnostic set: five fixed prompts plus four held-out excerpts, only 128 generated bytes and three seeds. (2) The corpus is source-code-heavy and mixed-domain; this is not a general language benchmark. (3) The held-out excerpts are not independent external data; they are sampled from the final 10% of the same corpus. (4) Mean raw entropy is measured along each strategy's own generated trajectory, so differences between strategies mix distribution changes with context drift; it is not a fixed-context comparison of model uncertainty. (5) The aggregate means are descriptive, not statistical significance claims.

**Research decision:** Close GSA-01 as a decoding diagnostic. Do not spend another experiment merely sweeping temperature. The next useful question should isolate whether the bottleneck is corpus/domain coherence and next-byte distribution quality versus the latent/shared-constructor representation. Before architecture claims, establish a reproducible held-out evaluation with fixed contexts and continuation-quality criteria, and compare against the same dense/low-rank/latent checkpoints. A follow-up should be designed as one discriminating experiment, not a broad benchmark expansion.
