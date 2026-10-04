# Neural Compression

Neural Compression is an experimental research project investigating how much of the information required for neural behavior can be represented by a smaller, reusable structure rather than by storing every parameter directly.

The project began as an exploration of neural-weight compression. Over time, the research shifted toward a broader question:

> Where does the information required for neural behavior actually live, and how small can the variable representation become if shared machinery captures reusable structure?

The current research direction is especially interested in tiny per-structure latent representations, shared constructors, low-rank generated structure, and persistent relational information.

## What this repository is

This is a research archive and experimental codebase, not a finished neural-network architecture or a claim of a solved compression problem.

The repository preserves experiments, controls, negative results, research notes, prototypes, and the reasoning that led from one experiment to the next. Some experiments are exploratory or incomplete. That is intentional: the history is part of the research record.

The strongest current question is:

> Does a latent/shared-constructor representation improve language-learning efficiency per parameter, byte, or unit compute as scale increases?

The project has already produced an integrated latent-native language-model prototype. The next meaningful test is not whether the individual pieces can work, but whether the combined representation produces a measurable advantage against appropriately matched dense and direct-low-rank controls.

## Current research picture

The experiments provide several useful pieces of evidence, with important limits.

Behavioral fidelity is not always the same thing as weight-space fidelity. Activation-aware low-rank factorization produced lower logit reconstruction error than truncated SVD at the same rank in a controlled comparison, and at one tested rank recovered all held-out continuations in the reported runs.

Very small latent representations can retain useful behavior in some settings. However, the experiments also showed that a target-specific decoder can hide information in the constructor. A tiny latent is therefore only interesting when the machinery interpreting it is genuinely shared and its cost is properly accounted for.

Persistent relational experiments provide evidence that relationships between local units can carry predictive information that the local units alone do not contain in controlled synthetic systems. The persistent relational concept-memory experiments are promising but unstable across seeds and configurations, so they are not presented here as a solved mechanism.

The project also found local anisotropic geometry in latent spaces. Later experiments showed that this geometry does not automatically provide a universal semantic coordinate system or a universal predictor of behavioral importance.

The lossless-compression branch further demonstrated that learned shared predictive structure can become economical when its fixed model cost is amortized over sufficiently large inputs. This is a useful engineering result, but it does not by itself establish that the same mechanism produces a better neural representation.

These distinctions matter. The repository deliberately records both positive and negative results.

## Research lineage

The project developed through several stages:

neural weight compression -> behavioral/activation-aware compression -> tiny latent representations -> shared constructors -> relational representations -> persistent relational structure -> latent/shared-constructor language models.

The research archive documents this progression rather than presenting the final hypothesis as if it appeared fully formed.

See ORIGIN.md for the project's conceptual lineage.

See NEURAL_COMPRESSION_AUDIT.md for the current reconstructed state of the research and its evidence boundaries.

See RESEARCH_LOG.md for the evolving research record.

## Important negative results

The project does not claim that every proposed mechanism worked.

A simple functional latent objective did not consistently outperform weight-MSE optimization.

Generic relational attention did not automatically produce held-out generalization.

Latent curvature or ridge directions did not universally predict behavioral importance.

Some persistent relational memory configurations improved ambiguous queries while others degraded them.

Early behavioral evaluations also contained insensitive prompts, which led to a stronger emphasis on continuous logit and activation metrics.

These results narrowed the research question and changed what experiments were pursued next.

## Reproducing the work

The repository contains many historical experiments with different dependencies, datasets, configurations, and hardware assumptions. There is not yet a single one-command reproduction environment for the entire archive.

Experiment-specific scripts and documentation should be treated as the authoritative source for reproducing individual results.

Where a result has not been fully replicated or an artifact is missing, the repository attempts to say so explicitly rather than presenting it as established.

## Status

This is an experimental research release.

It should be read as a documented body of work and a set of testable hypotheses, not as a production compression library, a proven replacement for conventional transformers, or evidence for claims outside the experiments themselves.

The next major experiment proposed by the current audit is a matched comparison between:

- an ordinary dense transformer,
- a direct low-rank transformer,
- and a latent/shared-constructor transformer using tiny per-structure latents and generated low-rank MLP structure.

The comparison should match data, tokenizer, optimization budget, seeds, and compute accounting as closely as practical, and measure validation loss/perplexity, throughput, active state bytes, checkpoint bytes, constructor amortization, memory, and behavioral performance.

## License

This repository is released under the Apache License 2.0. See LICENSE for the full terms.

The project attribution in NOTICE and CITATION.cff is intended to make the research provenance easy to preserve and reference.

## Attribution

This project was developed by Brent Malone (GitHub: @mrbrent328) with substantial AI-assisted research, analysis, coding, and documentation.

The project history is intentionally preserved in Git. Earlier experiments and research notes remain part of the record.

The work should be understood as a human-AI research collaboration: the repository's chronology, experiment decisions, results, and surviving artifacts provide the most precise record of how the project developed.

If you build on this work, please cite the specific experiment, commit, or release that you use whenever practical.

## Research archive

The repository's Git history is an important part of the archive. Please do not treat the latest files as the entire story.

For questions about a particular result, start with the relevant experiment file, result artifact, research-log entry, and commit history together.

## Disclaimer

This project is exploratory research. Results are subject to the limitations described in the corresponding experiment records and audit documents. No claim in this repository should be interpreted more strongly than the evidence supporting it.
