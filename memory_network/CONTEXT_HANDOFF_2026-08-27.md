# Neural Compression — Context Handoff

**Date:** 2026-08-28
**Branch:** `main`

## Purpose

This is the durable checkpoint for resuming Neural Compression research in a fresh conversation. The repository is authoritative. Follow `PROJECT_WORKFLOW.md`: one experiment per conversation, discuss before coding, implement only after approval, commit to `main`, use the user's real run for analysis, then close out and update this handoff.

## Current research direction

The project is now pivoting from isolated component experiments toward building a small complete trainable system that combines the useful mechanisms discovered across the experiments.

The central new idea is:

> Can concepts exist as persistent entities whose identity remains stable while their meaning is represented by a changing relational neighborhood?

The longer-term architectural hypothesis is that a useful model may consist of persistent local representations connected by a shared relational/interaction layer, with state evolving over time and potentially admitting a compact geometric representation. The user is particularly interested in whether abstract concepts can be represented more like objects embedded in a structured relational space: for example, `apple` would not be a single flat vector but a persistent identity connected to concepts such as `red`, `fruit`, `sweet`, `food`, `tree`, etc. The meaning of a concept would emerge from its surrounding structure and relationships.

The project should eventually test whether such relational structure can develop useful geometry, potentially in two or three dimensions and later possibly with time as an additional dimension. Coordinates must not be treated as the answer in advance; the first goal is to determine whether the relational architecture itself produces useful structure.

## Completed experiment: Out-of-Distribution Structural Delta Selection

### Question

Can the small nearest-centroid structural selector learned from the original `random`, `clustered`, and `similar` delta families choose efficient lossless representations for genuinely unseen change structures?

### Actual result

The complete test contained 5600 cases. Known cases produced an 83.0952% oracle-choice match rate, 1.0112543 mean ratio, 1.7571-byte mean regret, and 38-byte maximum regret. Novel cases produced an 82.1143% match rate, 1.0558700 mean ratio, 40.1143-byte mean regret, and 821-byte maximum regret. Overall match was 82.4821%, with 4619 zero-regret cases. Adaptive output was better than fixed `absolute` in 3789 cases, equal in 1799, and worse in only 12.

The dominant failure was `clustered_similar`, which had only 28.5714% oracle-choice match and 197.53-byte mean regret, with worst cases reaching 821 bytes. The experiment supports the narrower claim that cheap structural statistics contain transferable information for selecting among lossless delta representations, but does not establish universal generalization.

The earlier handoff proposed investigating whether that failure was caused by missing information or an insufficient nearest-centroid model. That investigation is now intentionally **paused rather than continued as the next research train**. The failure remains useful historical evidence and should not be silently converted into a successful result or retroactively tuned.

## Completed experiment: S5E1 Persistent Relational Concept Memory

### Question

Can a model maintain persistent identities for concepts while their meaning is represented by a changing relational neighborhood?

### Implementation

The experiment used 24 persistent concept states, 8 generic relation states, 16-dimensional embeddings, a shared neural interaction function, and symbolic relational observations. No target embeddings or coordinates were supplied. The model learned relation scores for persistent concept pairs and produced neighborhood signatures.

The experiment was intentionally designed as a precursor to explicit spatial geometry rather than a geometry experiment itself.

### Actual result

The supplied run showed identity persistence at effectively numerical precision, with mean cosine similarity of approximately 1.0 and minimum approximately 0.99999988. Relation retrieval accuracy was approximately 97.9167%, and neighborhood signatures had nonzero mean norm, indicating contextual structure around persistent identities.

These results support the basic architectural premise that persistent concept representations can coexist with relational context. However, an important implementation limitation was identified during closeout: the current evaluation path did **not actually apply relational updates to the persistent concept states**. Therefore the experiment does not establish the stronger claim that a concept's persistent representation dynamically changes in response to new relational observations. It demonstrates the substrate for persistent identity plus relational context, not the full dynamic-memory mechanism.

Repository artifact: `memory_network/relational_concept_experiment.py`.

## Closeout conclusion

The experiments are now considered sufficient to justify moving from isolated mechanism tests toward a first complete trainable system.

The conclusion is not that the individual experiments proved a complete theory of neural representation. Rather, multiple experiments have pointed toward the same architectural question: instead of continuing to optimize isolated selectors or memory components, build a small real system that combines persistent entities, local representations, relational interaction, changing context, and learning, then observe what behavior emerges.

The user's stated goal is straightforward: **build the system and see whether the idea actually works.** It does not need to compete with a large language model. The first meaningful success would be a behavior that emerges naturally from the architecture rather than being explicitly hard-coded.

## Next experiment / active direction

The next experiment is the first integrated model.

The system should begin small and be designed so that every component can be inspected. The initial architecture should include persistent concept/entity states, a shared relational interaction mechanism, a mechanism for incorporating observations over time, and a learning objective that requires the system to preserve identity while using relational context.

The first version should avoid explicit coordinates. Geometry should remain an emergent question. If the relational system demonstrates useful structure, later experiments can test whether that structure can be compressed or organized into a low-dimensional spatial representation, followed by three-dimensional and potentially four-dimensional-with-time variants.

A useful conceptual progression is:

`persistent concepts → relational neighborhoods → changing relationships → integrated learned system → emergent geometry`

The first integrated model should be deliberately small enough to understand, train, and inspect locally. The existing vision-model work is also relevant as evidence that the project can already construct a learned perception component; future architecture work can therefore consider perception as an input subsystem rather than treating it as an unsolved prerequisite.

## Important interpretation boundary

Do not claim that this architecture reproduces how human brains work. The motivating intuition is that humans often experience abstract ideas spatially or relationally, but the experiments test an engineering hypothesis, not a neuroscience claim.

Do not give the model coordinates and then interpret successful coordinate prediction as discovered geometry. If geometry is tested later, the representation must arise from relational constraints or learned structure rather than being supplied as the target.

## Workflow note

The prior `clustered_similar` selector investigation remains paused. The active research thread is the integrated relational model described above. Future handoffs should preserve this distinction so the project does not accidentally resume the older selector-debugging train.

## Repository artifacts

`memory_network/generalization_delta.py` — OOD structural delta selection experiment.

`memory_network/generalization_delta_result_2026-08-28.json` — actual summarized OOD result.

`memory_network/EXPERIMENT_CLOSEOUT_2026-08-28.md` — formal OOD experiment closeout.

`memory_network/relational_concept_experiment.py` — S5E1 persistent relational concept memory experiment.

`PROJECT_WORKFLOW.md` — canonical experiment workflow.
