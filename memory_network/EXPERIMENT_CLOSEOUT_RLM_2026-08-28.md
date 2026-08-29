# Neural Compression — RLM Language-Domain Research Closeout

**Date:** 2026-08-28  
**Branch:** `main`

## Purpose

This note records the first language-domain implementation of the relational architecture developed through the earlier Neural Compression experiments. It consolidates the RLM v1/v2 results and the architectural interpretation connecting them to R3/R4 and the persistent-relational work.

## Research question

Can the architectural principle developed through Neural Compression — persistent/distributed representations connected by shared relational interaction — operate on real language data without simply replacing it with a conventional Transformer-style architecture?

A secondary question is whether language tokens can serve as persistent entities whose identity remains stable while contextual/relational state changes with surrounding tokens.

## Architectural lineage

R3 moved away from a single global latent bottleneck toward distributed local representations and shared interaction. R4 continued this approach and explicitly separated interaction from representation through ablation.

S5E1 introduced persistent concept identities with relational context, while S6E2 integrated persistent entities, mutable contextual state, shared interaction, durable observations, and learned relational scoring.

RLM adapts this principle to language:

`token sequence → persistent token identities → shared relational field → contextual state → next-token prediction`

This is a language-domain implementation of the earlier relational principle, not a literal reproduction of R3/R4.

## RLM v1

V1 used byte-level persistent identities. It learned next-token statistics but produced severely degraded generation, including broken word fragments. This established that the relational language system could train but that byte-level persistent units were a poor substrate for readable language generation.

## RLM v2 — Gorgias run

Configuration of interest:

- context: **32**
- dimension: **512**
- heads: **1**
- relational rounds: **2**
- batch size: **8**
- learning rate: **3e-4**
- validation fraction: **0.1**

Reported result:

- vocabulary size: **2,996**
- training tokens: **70,678**
- validation tokens: **7,854**
- validation loss: **3.3774792554427164**
- validation perplexity: **29.296829223632812**

Generation from:

`CALLICLES: Yes, Gorgias, but I must complain`

produced recognizable Gorgias-specific vocabulary, dialogue names, punctuation, and local dialogue structure. It remained substantially incoherent as prose. The result supports learned corpus/local structure, not semantic understanding.

## RLM v2 — large-corpus stress test

Reported final checkpoint:

- training tokens: **2,155,783**
- validation tokens: **239,532**
- vocabulary size: approximately **38,538**
- validation loss: **4.870605817903878**
- validation perplexity: **130.39990234375**

Training perplexity reached approximately **20** late in the run while validation remained around **130**. This is a substantial generalization gap.

The large vocabulary also changed the parameter accounting substantially. With 128-dimensional persistent identities, the vocabulary embedding matrix alone is approximately 4.95M parameters. Therefore parameter count must always be reported alongside vocabulary size when comparing RLM runs.

Generation from `Hi friend` produced recognizable English words and corpus vocabulary including `CHAPTER`, `infection`, `Haversian`, `iodoform`, `prince`, `capital`, `pupil`, `assembled`, and `assiduity`, but not coherent prose. This is evidence of learned lexical/local distributional structure rather than language understanding.

## Persistent identity inspection

`rlm_inspect.py` was used to inspect learned identity vectors. Static nearest neighbors were not obviously semantic:

- `the` → `intrusion`, `bwother`, `Indians`, `donning`, ...
- `infection` → `Cladius`, `importunity`, `590`, `limbs_`, ...
- `prince` → `Nullification`, `aren't`, `autocracy`, `desk`, ...
- `patient` → `assumes`, `Antivenin`, `appreciated`, ...

Similarities were generally around 0.30–0.36.

### Interpretation

The raw identity space is not directly optimized to make cosine-nearest neighbors human-readable semantic categories. These results therefore neither prove nor disprove relational semantics. The more appropriate next diagnostic is to compare the same persistent identity under different contexts and inspect its contextual state and prediction distribution.

## Findings

### Supported by the experiments

1. Distributed/local representations can replace a single global latent bottleneck as an architectural substrate.
2. Shared interaction is a distinct and meaningful mechanism worth isolating.
3. Persistent identities can coexist with changing contextual/relational states.
4. The integrated relational system is operational and inspectable.
5. The relational architecture can learn substantial statistical structure from real language.
6. Moving from byte identities to learned token identities materially improves the language substrate.
7. RLM v2 can train on a corpus containing millions of tokens.
8. Static identity-vector geometry does not automatically become human-readable semantic geometry.

### Not demonstrated

1. Genuine semantic understanding.
2. Superior generalization compared with conventional language models.
3. Useful long-range reasoning.
4. Competitive LLM-level language performance.
5. That the architecture is the smallest sufficient representation.
6. That the architecture reproduces biological brain organization.

## Conclusion

The RLM experiments establish a useful milestone: the relational architecture developed through Neural Compression can be instantiated as a trainable language system and can learn nontrivial language statistics from both a focused corpus and a multi-million-token corpus.

The strongest conceptual interpretation is that RLM is the language-domain test of the architectural principle developed through R3/R4 and subsequent persistent-relational experiments:

> **persistent entities/local representations + shared relational interaction + changing contextual state**

The experiments do not yet establish that this produces semantic understanding. The next work should therefore inspect contextual relational behavior and controlled generalization before simply scaling the model or claiming comparison with conventional LLMs.

## Next step

Construct a controlled diagnostic that holds a persistent token identity fixed while varying its surrounding context. Measure changes in contextual state, relational interaction, and prediction distribution. This directly tests the intended separation between persistent identity and changing relational meaning.

After that, larger-scale training and controlled conventional-LLM comparisons can be considered.
