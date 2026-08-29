# Neural Compression — Context Handoff

**Date:** 2026-08-28  
**Branch:** `main`

## Purpose

This is the durable checkpoint for resuming Neural Compression research in a fresh conversation. **`main` is the authoritative repository state.** Follow `PROJECT_WORKFLOW.md`: establish current state from `main` before continuing, read the relevant research notes/artifacts, discuss before coding, implement only after approval, use the user's real run for analysis, commit completed work to `main`, then close out and update this handoff.

## Current research direction

The research has progressed from isolated compression mechanisms toward a complete trainable relational system and, now, a first language-domain implementation of the relational principles discovered in earlier experiments.

The central hypothesis is:

> A useful model may be built from persistent entities/local representations whose identity remains stable while their meaning is represented through changing relational/contextual structure.

The longer-term question remains:

> What is the smallest persistent relational structure capable of explaining multiple observations of the same underlying system?

The current language work is an engineering test of this hypothesis, **not** a claim about human neuroscience or cognition.

## Historical architectural progression

### R3 / R4 — distributed local representations + shared interaction

R3 moved away from a single global latent bottleneck toward multiple local representations. The architecture represented an observation as local units, passed those representations through shared interaction machinery, and reconstructed from the resulting states.

R4 continued this direction and explicitly tested interaction as a separable mechanism through ablation. The important architectural principle was the separation between distributed representations and a shared mechanism through which those representations can interact.

These experiments motivate the current language architecture but do not by themselves establish semantic representations.

### S5E1 — persistent relational concept memory

S5E1 tested persistent concept identities plus relational context. It demonstrated identity persistence and strong relation retrieval, but closeout identified an important limitation: the evaluation path did not actually apply relational updates to persistent concept states. Therefore it demonstrated the substrate for persistent identity plus relational context, not the full dynamic-memory mechanism.

### S6E2 — integrated persistent relational system

S6E2 combined persistent entities, exact durable observations, mutable contextual states, shared neural interaction, learned relation prediction, checkpoint persistence, observation replay, and an interactive CLI.

The live system demonstrated that adding a relational observation changed contextual state while the underlying entity remained persistent. It successfully learned observed relationships. It did **not** yet demonstrate held-out relational inference or semantic understanding.

## RLM — language-domain implementation

RLM adapts the distributed/persistent representation + shared interaction principle to language.

Conceptual mapping:

| Earlier relational work | RLM |
|---|---|
| observation | token sequence / context |
| local/entity representation | persistent token identity |
| shared interaction | relational field |
| contextual state | context-dependent token state |
| reconstruction | next-token prediction |

RLM should therefore be described as a **language-domain implementation of the relational architectural principle developed through R3/R4 and subsequent persistent-memory experiments**, rather than as a literal copy of R3/R4.

## RLM v1

RLM v1 used byte-level persistent identities. It successfully learned nontrivial next-token statistics, but generated severely degraded text. This established a working language-prediction baseline while showing that byte-level persistent units were not a particularly useful substrate for readable generation.

## RLM v2

RLM v2 changed the persistent unit from bytes to learned vocabulary tokens while retaining the relational-field approach.

### Gorgias-scale run

Reported result:

- vocabulary: **2,996**
- training tokens: **70,678**
- validation tokens: **7,854**
- validation loss: **3.3774792554427164**
- validation perplexity: **29.296829223632812**

Generation from `CALLICLES: Yes, Gorgias, but I must complain` produced recognizable Gorgias-specific vocabulary, character/dialogue names, punctuation, and local dialogue structure, but remained substantially incoherent as prose. This is evidence of learned corpus structure, not semantic understanding.

### Large-corpus stress test

A much larger text corpus was also trained with RLM v2. Reported final checkpoint metadata:

- training tokens: **2,155,783**
- validation tokens: **239,532**
- vocabulary: approximately **38,538** tokens in the reported run
- final validation loss: **4.870605817903878**
- final validation perplexity: **130.39990234375**

Training perplexity reached approximately **20** late in the run while validation remained around **130**, establishing a substantial generalization gap. The large vocabulary also revealed that the persistent identity table dominates model parameter count at this scale; with 128-dimensional identities, the vocabulary embedding matrix alone is roughly 4.95M parameters.

Generation from `Hi friend` produced recognizable English words and corpus vocabulary such as `CHAPTER`, `infection`, `Haversian`, `iodoform`, `prince`, `capital`, `pupil`, `assembled`, and `assiduity`, but not coherent prose. This indicates learned lexical/local distributional structure rather than useful language understanding.

## RLM v2 architectural configuration of interest

The stronger Gorgias experiment used:

- `context = 32`
- `dim = 512`
- `heads = 1`
- `rounds = 2`
- `lr = 3e-4`
- `batch-size = 8`
- `val-fraction = 0.1`

This is notable because it gives persistent entities a relatively large representation while keeping the shared relational interaction mechanism comparatively small/simple. This is conceptually compatible with the earlier research direction of persistent/distributed representations connected through shared interaction.

The parameter count must always be considered alongside vocabulary size; a large vocabulary makes the persistent identity table the dominant component.

## RLM identity-space inspection

A diagnostic tool was added as `rlm_inspect.py` after an initial `inspect.py` name collision with Python's standard-library `inspect` module was discovered and corrected.

For the large-corpus checkpoint, inspecting static identity vectors produced non-obvious nearest neighbors. Examples included:

- `the` → `intrusion`, `bwother`, `Indians`, `donning`, ...
- `infection` → `Cladius`, `importunity`, `590`, `limbs_`, ...
- `prince` → `Nullification`, `aren't`, `autocracy`, `desk`, ...
- `patient` → `assumes`, `Antivenin`, `appreciated`, ...

The strongest similarities were only around 0.30–0.36 and were not obviously semantic. The correct interpretation is **not** that the relational model failed. Static identity-vector cosine geometry is not directly optimized to produce human-readable semantic neighborhoods. The next diagnostic target should therefore be contextual/interaction states and prediction behavior rather than raw nearest-neighbor identity geometry.

## Scientific synthesis

The current evidence supports the following narrower claims:

- Distributed/local representations are a viable alternative to forcing an observation through one global latent bottleneck.
- Shared interaction between distributed representations is a meaningful architectural mechanism worth isolating experimentally.
- Persistent identities can coexist with changing relational/contextual state.
- An integrated persistent relational system can be trained, persisted, inspected, and replayed.
- The relational architecture can learn substantial statistical structure from real language.
- Changing the persistent unit from bytes to learned tokens materially improves the linguistic substrate.
- RLM v2 can train on a corpus containing millions of tokens.
- Static identity-vector geometry does not automatically become human-interpretable semantic geometry.

The following remain unestablished:

- genuine semantic understanding;
- superior generalization versus conventional language-model representations;
- useful long-range reasoning;
- competitive performance against conventional LLMs;
- that the learned structure is the smallest sufficient representation;
- that the architecture reproduces biological brain organization.

## Research progression

`distributed representations → shared interaction → persistent entities → changing relational context → integrated relational system → language-domain RLM → contextual relational analysis → held-out inference / generalization → emergent geometry`

Geometry remains an emergent question. Coordinates must not be supplied as targets and then interpreted as discovered geometry.

## Next research question

The immediate next architectural/diagnostic question is whether persistent token identities and the shared relational field produce **structured contextual states** that explain prediction behavior.

A useful next experiment is to inspect the same persistent entity under different contexts and compare its contextual state and prediction distribution. The test should distinguish persistent identity from changing relational/contextual state.

Only after that should we move toward larger-scale training or direct comparison against conventional language models. If scaling is pursued, vocabulary size, parameter count, corpus composition, context length, and training budget should be controlled so results remain interpretable.

## Important interpretation boundaries

Do not equate low training loss with semantic understanding.

Do not treat static embedding nearest-neighbor results as proof or disproof of relational semantics.

Do not claim that RLM has beaten or reproduced an LLM without a controlled benchmark against a defined conventional baseline.

Do not claim that the architecture reproduces human brain organization.

Do not supply coordinates and then interpret successful coordinate prediction as discovered geometry.

## Repository / workflow rule

**All authoritative research notes, handoffs, and completed experiment artifacts belong on `main`.** When resuming in a new conversation, explicitly start by reading the current `main` state and the relevant docs. Do not rely on stale conversational summaries when repository evidence is available.

The prior `clustered_similar` structural-selector investigation remains paused. Its failures remain historical evidence and should not be silently tuned into success.
