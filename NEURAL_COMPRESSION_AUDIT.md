# Neural Compression — Canonical Audit Summary

**Audit date:** 2026-09-10  
**Status:** Partial but high-confidence reconstruction of the major experimental lineage. This file is intended to be the canonical project-context handoff and should be updated as the remaining archaeology is completed.

## Core scientific question

Where does the information required for neural behavior actually live, and how small can the variable representation become if shared machinery captures reusable structure?

The project began as neural-weight compression, but the experiments increasingly distinguish raw numerical weight reconstruction from preservation of function/behavior. The recurring hypothesis is that useful behavior may be represented by a much smaller state than the original parameter tensor, provided the representation is interpreted by shared machinery and trained against the right objective.

## Most important correction from the audit

The project has **already built an integrated latent-native language-model prototype**. The next serious question is therefore not merely whether the pieces can be assembled, but whether the architecture produces a scaling advantage against appropriate dense and direct-low-rank controls.

The latent-transformer branch uses tiny per-structure latent variables plus shared constructor machinery to generate low-rank MLP structure inside a functioning decoder language model. A large architecture sweep was run around this design. The next step should treat that as version zero rather than restarting component discovery.

## High-confidence findings

### 1. Weight-space fidelity and behavioral fidelity are not the same objective

Across the project, better numerical reconstruction does not reliably imply better behavioral reconstruction. This motivated activation/behavior-aware objectives.

A particularly strong controlled result is the activation-aware low-rank factorization comparison against ordinary truncated SVD at the same rank/storage budget. At rank 64, activation-aware variants reduced logit RMSE from roughly 0.0655 for SVD to roughly 0.0416–0.0435. At rank 128, all three activation-aware runs recovered 24/24 held-out continuations while the SVD control recovered 23/24. Calibration and evaluation prompts were separated.

**Working conclusion:** If the goal is to preserve what the model does, the states/directions actually used by the model can be a better optimization target than raw matrix error.

### 2. Latent dimensionality is often not the primary bottleneck

Several experiments pushed latent capacity down to one or two dimensions and later swept it upward. Larger latent dimensionality frequently produced little or no reproducible held-out behavioral gain.

**Working conclusion:** Stop treating “how few latent scalars can we use?” as the main research question. The more important variables appear to be the shared constructor/interpreter, relational structure, and training objective.

### 3. Decoder/constructor memorization is a real confound

Early spectacular tiny-latent reconstructions demonstrated that a tiny code can select/reconstruct useful behavior, but a target-specific decoder can absorb a large amount of information. Cross-target/shared-constructor experiments weakened the apparent magic substantially.

**Working conclusion:** A tiny latent is scientifically/economically interesting only when its constructor is genuinely shared and amortized across many targets/structures. Always account for constructor size.

### 4. Generic relational machinery is not automatically sufficient

E15/E16 tested pooled relational and true pairwise-attention variants. They did not automatically solve held-out generalization.

**Working conclusion:** Retire the simple claim “add relational attention and generalization appears.” Relations matter only when the task/representation forces useful relational information to be carried.

### 5. Later persistent-relational experiments show a cleaner relational effect

E37 (MPRS traversal invariance) deliberately withheld absolute world coordinates, world indices, hidden geometry, position-derived features, and future observations from memory.

Full relational condition:
- final units: 160
- final relations: 128
- mean pass-2 reconstruction error: ~0.1717
- mean pass-2 prediction F1: ~0.5534
- mean pass-2 prediction position error: ~0.3818

No-relations ablation:
- same 160 units
- reconstruction error remained ~0.1717
- prediction F1 collapsed to 0
- prediction position error became 1.0

The minimum-structure sweep showed a dose-response relationship. With 64 retained relations, prediction F1 reached ~0.5156 and position error ~0.3365; larger relation sets continued improving prediction.

**Working conclusion:** In this synthetic traversal system, local units can preserve/reconstruct observations while relations carry information required to predict how those observations connect across traversal/time. This is a real ablation result, although it does not by itself establish the same mechanism in language models.

### 6. Persistent relational concept memory is promising but unstable

E41 has multiple configurations/seeds. Some runs improve genuinely ambiguous/homonym queries as relational context increases; others degrade. Identity persistence can remain extremely high while relational neighborhoods change.

**Classification:** SUPPORTED MECHANISM / UNSTABLE IMPLEMENTATION.

Do not cherry-pick the successful E41 runs. The aggregate says there is useful signal, but the formulation is not yet robust enough to call solved.

### 7. Latent geometry is real locally, but semantic interpretation is not simple

The E24–E29 family found anisotropic local latent geometry: preferred and insensitive directions occur. E28 then showed that curvature/ridge direction does not map cleanly or universally onto behavioral importance across seeds. E29 established coordinate/rotation invariance of the mathematical geometry.

**Classification:**
- Local anisotropic geometry: SHOWN.
- Universal meaningful latent axes: NOT SHOWN.
- More coordinate-rotation sanity tests: RETIRED unless a new hypothesis requires them.

### 8. Exact learned compression can amortize model overhead

The Test9 lossless-compression branch eventually beat zlib on sufficiently large source-code inputs while including model overhead. It lost badly at small sizes, approached break-even as the predictor was amortized, and became advantageous at larger inputs.

Test10 tightened methodology: held-out data separation, integer range coding driven by the learned probabilities, identical causal encoder/decoder state, SHA-256 reconstruction checks, and explicit model/metadata cost. A definitive final Test10 measurement artifact has not yet been located in this audit, so Test10 remains OPEN rather than being counted as a win.

**Working conclusion:** Shared learned structure can be initially expensive while per-instance representation becomes cheap enough for amortization to matter. This pattern is conceptually consistent with the shared-constructor neural experiments.

### 9. The original 24-prompt behavioral suite had measurement dead zones

Later auditing found that some prompts were insensitive floors (the model still answered correctly when the target projection was zeroed/randomized), while others were effectively unreachable ceilings (even strong high-rank approximations failed). The active suite was expanded to target sensitive prompts more deliberately.

**Working conclusion:** Early exact-count results such as 12/24 versus 15/24 are useful but crude. Prefer sensitive-prompt and continuous logit/activation metrics for new work.

## Important negative/boundary results

- E18: a simple functional latent objective did not consistently outperform weight-MSE optimization; one seed was materially worse. Retire the blanket claim that a functional objective is automatically better.
- E15/E16: pooled relation features and pairwise attention did not automatically solve held-out reconstruction/generalization.
- E28: latent Hessian/ridge geometry does not universally predict behavioral importance.
- E30: technical failure/incomplete artifact; do not interpret as scientific evidence.
- E41: relational context effects vary by seed/configuration; promising but not robust.

Negative results are part of the project map, not failures to remember.

## Current synthesis

The strongest recurring picture is:

1. Useful function can occupy a much smaller effective representation than raw parameter space suggests.
2. Exact numerical weight reconstruction is often less important than preserving behavior on the states the model actually visits.
3. Making the latent larger frequently matters less than improving the shared machinery that interprets it.
4. Tiny-code claims must include/amortize constructor cost; otherwise information has merely moved into the decoder.
5. Relational structure can carry predictive information that local item representations alone do not, at least in controlled synthetic systems.
6. These ideas can already be assembled into a functioning latent/shared-constructor language model.

## Recommended next major experiment

Do **not** make E41, more geometry work, compressed LayerNorms, whole-model reconstruction, or another exotic latent mechanism prerequisites.

Freeze most of a conventional transformer architecture:
- conventional embeddings
- conventional residual stream
- conventional normalization
- conventional attention

Make the experimental component the part with the strongest accumulated evidence:

**tiny per-structure latent → shared constructor → generated low-rank MLP structure**

Include activation/behavior-aware training where justified by the controlled compression results.

Train three matched families:
1. ordinary dense transformer
2. direct low-rank transformer
3. latent/shared-constructor transformer

Match data, tokenizer, optimization budget, seeds, and as closely as possible compute/parameter accounting. Scale through several model/data sizes. Measure validation loss/perplexity, training throughput, inference throughput, active parameter/state bytes, checkpoint bytes, constructor amortization, memory, and long-context/behavioral tests.

The central question is now:

> Does the latent/shared-constructor representation improve language-learning efficiency per parameter, byte, or unit compute as scale increases?

If yes, characterize the scaling law and failure boundaries. If no, identify where the integrated hypothesis breaks.

## Project-management rule going forward

For every experiment, maintain a canonical ledger entry containing:
- experiment ID/name
- exact script/config/result paths
- hypothesis
- controls
- measured result
- classification: SHOWN / SUPPORTED / OPEN / FAILED / RETIRED
- what the result does *not* establish
- architectural implication
- whether it needs replication

Do not rely on chat memory for project state.

## Remaining archaeology

This audit is intentionally not marked complete. Remaining work includes:
- early numbered Test1–Test8 artifacts and historical versions
- remaining R-series/CIFAR/text branches
- synthetic-latent predecessor runs not yet individually classified
- commit/deleted-version archaeology where local and GitHub histories diverged
- final reconciliation of local/Library-only experiments with GitHub main

The local/Library project contains later work not cleanly represented in GitHub main, including E37/E41-era material. This divergence is itself a project-management problem and should be reconciled.

## One-sentence project state

**We are no longer asking whether tiny latent/shared machinery can do anything; we have enough evidence and a working language-model prototype to justify asking whether the representation scales into a measurable advantage.**
