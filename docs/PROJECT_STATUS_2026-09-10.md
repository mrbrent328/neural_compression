# Neural Compression — Project Status Summary

Date: 2026-09-10

This document captures the current working interpretation of the Neural Compression project after a broad review of the repository, Library artifacts, experiment summaries, and historical discussion. It is intentionally a research-status document, not a claim of novelty or a publication-ready result.

## Current high-level conclusion

The project is no longer best described as a collection of unrelated compression experiments. The main line has converged on a coherent question:

> How much of a neural system's useful function can be represented by a much smaller state, when shared machinery interprets that state and the training objective is allowed to care about behavior rather than exact parameter reconstruction?

The strongest recurring result is that exact numerical weight reconstruction is often a poor proxy for preserving useful behavior. Shared constructor cost, relational structure, and the objective used to define equivalence matter at least as much as raw latent dimensionality.

## Major findings so far

### 1. Behavior-aware / activation-aware compression beats plain SVD at the same rank in the tested Q-projection setting

Historical experiment summary:

- Rank 64 (~7x factor storage compression): truncated SVD logit RMSE ~0.06549; activation-aware runs ~0.04162–0.04354.
- Rank 128 (~3.5x): activation-aware runs reached 24/24 exact held-out continuations across 3 runs, with lower logit RMSE than the SVD control; SVD reached 23/24 in that summary.
- Calibration and evaluation prompt sets were separated.

Interpretation: directions used by the model's actual behavior can be more compressible than raw matrix error suggests. This supports carrying functional/activation-aware objectives into future model designs.

### 2. Increasing latent dimensionality often matters less than expected

Across shared-constructor sweeps, moving from L1/L2 to L4/L8 frequently produced little or no reproducible behavioral gain. Tiny latents can retain surprisingly large fractions of behavior, but this should not be interpreted as the latent alone containing the whole model.

Working conclusion: stop treating `How tiny can Z be?` as the central question. The more important questions are how Z is interpreted, whether shared machinery generalizes, and what objective trains the representation.

### 3. Target-specific constructors can hide information

Some spectacular tiny-latent reconstruction results were confounded by the capacity of the constructor/decoder. A tiny latent is not a tiny standalone model when a large decoder is specialized to one target.

This motivated cross-target/shared-constructor tests and remains a core accounting rule:

> Shared machinery must be amortized across many targets/structures before tiny per-instance latents become meaningful compression.

### 4. Simple relational layers were not automatically sufficient

Early pooled and pairwise relational constructor experiments did not reliably improve held-out functional reconstruction. This is a useful negative result: merely adding relational communication does not solve generalization.

### 5. Relational structure can carry predictive information that local units alone do not

E37 (MPRS traversal invariance) is one of the clearest mechanistic results.

The memory system was not given absolute world coordinates, world indices, hidden geometry, position-derived features, or future observations.

Full relational condition:
- mean pass-2 reconstruction error: ~0.1717
- mean pass-2 prediction F1: ~0.5534
- mean pass-2 prediction position error: ~0.3818

No-relations condition:
- reconstruction error remained ~0.1717
- prediction F1 collapsed to 0.0
- prediction position error went to 1.0

A relation-retention sweep showed a dose-response improvement in predictive performance; 64 retained relations crossed the experiment's stated sufficiency thresholds (F1 ~0.5156, position error ~0.3365).

Interpretation: in this synthetic system, local units were sufficient for observation reconstruction, while relations carried information needed for traversal/prediction.

### 6. Persistent relational concept memory is promising but unstable

E41 variants show mixed results.

Some configurations improve genuinely ambiguous masked-homonym prediction with increasing relational context (for example ~0.389 -> ~0.497 in one run; ~0.234 -> ~0.306 in another), while other configurations degrade the same metric.

Alias identity often remains very strong, but robust generalization of relational memory has not yet been established.

Status: `SUPPORTED MECHANISM / UNSTABLE IMPLEMENTATION`.

### 7. Latent geometry exists, but simple "geometry = meaning" stories failed

L2 geometry studies repeatedly found anisotropic local landscapes, principal directions, ridges/valleys, and coordinate-invariant local structure. However, the relationship between curvature and behavior varied by seed and sometimes inverted.

Working conclusion:
- local latent geometry exists: supported
- universal semantic axes: not shown
- more rotation/invariance sanity checks: low priority / largely retired

### 8. The project already contains a functioning latent-native language-model branch

A major correction to prior planning: the project has already built small working language models using the shared-constructor / latent-weight idea. Therefore the next major question is not `Can this be made into a language model?`

It is:

> Does this architecture produce a scaling or efficiency advantage over strong conventional and direct-low-rank controls?

## Compression branch

The causal neural compression work is also more mature than sometimes remembered.

- The system uses a neural probability model with an exact entropy coder.
- Encoder and decoder reproduce the same causal state.
- Exact reconstruction and SHA-256 parity are required in the corrected Test10 lineage.
- Model + payload + metadata cost is explicitly counted.
- Earlier scaling work showed learned shared structure can lose badly at small inputs because model cost dominates, then become competitive or better once the predictor is amortized over enough data.

A definitive final Test10 measurement artifact still needs to be located/classified before making a strong claim about that branch.

## Important methodological correction: the original 24-prompt suite had dead zones

The original behavioral benchmark mixed:

- insensitive floor prompts (still correct even with the target projection zeroed/randomized),
- unsolvable ceiling prompts (not recovered even by high-quality low-rank baselines), and
- genuinely discriminating prompts.

A later active test suite explicitly separated these categories and expanded evaluation toward high-sensitivity prompts. Early exact-count results should therefore be treated as useful but coarse measurements rather than definitive functional scores.

## Current model-building recommendation

Do not make persistent memory, exotic geometry, compressed normalization, or whole-model reconstruction prerequisites for the next serious model.

For the next scaling study, keep most transformer machinery conventional and isolate the experimentally supported idea:

- conventional embeddings / residual stream / normalization / attention,
- tiny per-structure latent state,
- shared constructor,
- generated low-rank MLP structure,
- functional / activation-aware training signal where appropriate.

Compare against at least:

1. conventional dense transformer,
2. direct low-rank parameterized transformer at comparable trainable/storage budget,
3. shared-constructor latent transformer.

Train on enough diverse data that the dense control is not simply overfitting, use multiple seeds and model sizes, and compare loss/perplexity versus parameter count, serialized bytes, training compute, and inference compute.

The decisive open question is now:

> Does shared-constructor latent parameterization have a better scaling curve, parameter efficiency, or storage/function tradeoff than ordinary parameterization?

## Current conceptual map

```text
Can useful information live in a much smaller state?
        |
        +-- Sometimes, surprisingly small.
        |
Does exact weight reconstruction define success?
        |
        +-- Often no; behavior can diverge from weight error.
        |
Can a tiny latent be trusted by itself?
        |
        +-- No; constructor capacity must be counted/amortized.
        |
Does simply making the latent bigger solve it?
        |
        +-- Usually not much.
        |
Does adding generic relational machinery solve it?
        |
        +-- Not automatically.
        |
Can relations carry distinct predictive structure?
        |
        +-- Yes in controlled synthetic experiments (E37).
        |
Can this idea become a language model?
        |
        +-- Yes; small working versions already exist.
        |
NEXT QUESTION
        |
        +-- Does it scale into an advantage?
```

## Research-status language

Safe current statement:

> Useful neural function in these experiments often occupies a much smaller effective representation than raw parameter space suggests. Representation size alone is not the main bottleneck; shared decoder/constructor capacity, relational structure, and behavior-aware objectives matter substantially. The project has already demonstrated small language models built around shared-constructor latent parameterization, but a scaling advantage over strong conventional baselines has not yet been established.

Avoid claiming a breakthrough or unique novelty until the literature comparison and scaled controlled study are complete.

## Project-management note

The GitHub repository and the user's local/Library experiment archive are not perfectly synchronized. Later work such as E37/E41 exists in the Library even when not cleanly represented in the current GitHub tree. A future experiment ledger should be treated as canonical and should link each experiment to its actual script, result artifact, status, and conclusion.
