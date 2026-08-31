# Neural Compression — Context Handoff

**Date:** 2026-08-31  
**Branch:** `main`

## Purpose

Durable checkpoint for resuming Neural Compression research in a fresh conversation. **`main` is the authoritative repository state.** Follow `PROJECT_WORKFLOW.md`: review `main`, discuss before coding, implement only after approval, use the user's real run for analysis, commit completed work to `main`, then close out and update the handoff.

## Central research question

> What is the smallest persistent relational structure capable of explaining multiple observations of the same underlying system?

The research currently investigates whether persistent entities/local representations plus changing relational/contextual structure can provide a compact substrate for explaining multiple observations over time.

The compression branch of this work asks whether those ideas can also yield practical lossless representations, including adaptive combinations of neural and classical coding.

## Research progression

`distributed representations → shared interaction → persistent entities → changing relational context → integrated relational system → language-domain RLM → developmental continuous stream`

The parallel compression line progressed through neural byte prediction, latent/state experiments, local structural selection, Series 4 configuration/adaptive experiments, and now the NCA archive experiments.

## Developmental Stream — completed 2026-08-31

### Question

What persistent structures and behavior emerge when a relational neural system continuously experiences an environment, maintains finite relational memory, and learns what information to retain without predefined semantic concepts?

### Implementation

The system used 16 persistent entities with fixed identity vectors, mutable 32-dimensional contextual states, a shared interaction mechanism, prediction, prediction error, bounded short-term memory, bounded long-term memory, similarity-weighted retrieval, and an online consolidation network. The long-term memory capacity was 12. The environment was numerical and constraint-generated rather than semantic.

The intended loop was:

`observation → prediction → prediction error → contextual update → short-term insertion → consolidation scoring → finite long-term update → retrieval → later prediction`

### Actual result

User-run configuration:

- seed: `314159`
- steps: `5000`
- entities: `16`
- state dimension: `32`
- short-term capacity: `32`
- long-term capacity: `12`

Results:

- **5000 total consolidations**
- **12 final long-term memories**
- mean prediction error: **0.27975707083093004**
- mean prediction loss: **0.2327067244131118**
- mean consolidation loss: **0.591435519683361**
- identity updates during run: **false**
- online learning: **true**
- retrieval affects prediction: **true**

The authoritative run artifact is `memory_network/developmental_stream_result.json`.

### Main conclusion

The continuous developmental loop and finite-capacity memory system operated successfully. Persistent identity remained fixed while contextual state changed; short-term memory remained bounded; long-term memory remained bounded and was replaced; retrieved memories influenced prediction; and online learning occurred.

However, **the experiment did not demonstrate selective memory formation**. Every experience was consolidated because the implementation always selected the highest-scoring candidate and had no explicit retain-nothing/abstain action. Therefore the actual test was candidate ranking under mandatory consolidation, not learned remember-versus-forget behavior.

This is an architectural finding, not a result to be silently corrected after the fact.

### Scientific interpretation

Supported:

- continuous relational state evolution is operational;
- persistent identity and mutable contextual state can coexist;
- finite-capacity relational memory can operate continuously;
- retrieval can participate in prediction;
- online prediction/consolidation learning can run in the loop;
- the architecture is suitable for further adaptive-memory experiments.

Not established:

- selective learned memory formation;
- semantic understanding;
- consciousness or agency;
- biological brain equivalence;
- minimality/optimality of the representation;
- useful novel-condition generalization.

### Important lesson

**Candidate ranking is not the same as deciding whether anything should be remembered.**

Do not add a threshold to the existing run and retroactively interpret it as the original experiment. First analyze whether the learned scorer developed meaningful relative preference among candidates.

## NCA Archive Experiments — 2026-08-31

### Purpose

Test whether the neural compression work can function as part of a practical folder/archive compressor rather than only as a single-file experiment. The NCA architecture uses adaptive local selection so each block can use neural, classical, or store coding as appropriate.

### NCA-01 — baseline

Corpus: `/home/xetal/Desktop/largecalgarycorpus`  
Original: **3,179,847 B**  
Strategy: `adaptive`

Configuration:

- embedding: 32
- hidden: 64
- context: 16 bytes
- states: 4
- optimizer: SGD
- learning rate: 0.04
- seed: 42

Result:

- payload: **1,146,715 B**
- archive: **1,172,168 B**
- reduction: **63.1376%**
- neural blocks: **48**
- classical blocks: **8**
- store blocks: **0**
- elapsed: **2,532.5846 s** (~42m 12.6s)
- SHA-256 manifest: `4c83e773af71310efd20af445df17f96d73b0829930c05fb58c63badc63d348f`

NCA-01 subsequently passed the planned exact round-trip verification and is the locked archive baseline.

### NCA-02 — stronger neural configuration

Question: does the stronger Series-4-derived neural configuration improve the same archive architecture?

Configuration:

- embedding: **48**
- hidden: **300**
- context: **10 bytes**
- states: **3**
- optimizer: SGD
- learning rate: 0.04
- seed: 42
- state selection: deterministic minimum predictive entropy

Result on the same corpus:

- payload: **1,297,872 B**
- archive: **1,323,325 B**
- reduction: **58.3840%**
- neural blocks: **48**
- classical blocks: **8**
- store blocks: **0**
- elapsed: **2,351.8005 s** (~39m 11.8s)
- SHA-256 manifest: `1f3c2c498091020804248d6136cde48057b1d356f343c1452df8d252592fd038`

NCA-02 produced an archive **151,157 B larger** than NCA-01 while completing about **180.8 s faster**. The adaptive allocation was identical: 48 neural, 8 classical, 0 store.

The compression hypothesis therefore was **not supported**: a configuration that performed better in the Series-4 regime did not produce better sequential lossless coding in the archive workload.

NCA-02 exact round-trip verification remains pending and is required before final closeout.

### NCA-02 implementation note

The first NCA-02 attempt failed with a recursive `encode_candidate` dispatch bug. This was an implementation error, not experimental evidence. The dispatcher was corrected and committed to `main`; a small smoke test then successfully produced a 31.29% reduction on a 40.6 KB repository. That smoke test was only a functionality check and is not part of the Calgary corpus experiment.

### Paired interpretation

The clean comparison is:

`same corpus + same adaptive archive machinery + same block allocation → different neural configuration → worse NCA-02 compression`

This suggests the next investigation should focus on **neural predictor/state behavior inside the neural-selected blocks**, rather than immediately changing the adaptive archive selector.

Relevant detailed record: `NCA_RESULTS_2026-08-31.md`.

## Immediate next research decision

Before designing NCA-03, retrospectively analyze the earlier compression experiments—especially Series 4, latent-capacity/state experiments, and adaptive/agreement tests—to determine what property actually correlated with useful compression rather than simply selecting the configuration with the best score in another task.

Specific questions:

1. What did the Series-4 winning configuration optimize?
2. How did context length, model capacity, and state count affect predictive entropy or coded size?
3. Did adaptive/agreement experiments reveal state specialization or selection behavior relevant to NCA?
4. Do the earlier latent/functional-state experiments suggest a better representation or state mechanism than simply increasing model capacity?
5. Can we separate predictor quality from archive overhead and selection effects?

Do **not** automatically interpret the larger NCA-02 model as better simply because it had more capacity. Preserve the negative result and use the earlier evidence to choose the next experiment.

## Repository state

Developmental Stream proposal, implementation design, implementation, real-run result, closeout, and this current handoff are committed to **`main`**. NCA-01/NCA-02 results and NCA-02 implementation are also committed to **`main`**.

Relevant current files:

- `NCA-01_local_adaptive_archive.py`
- `NCA-02_strong_neural_archive.py`
- `NCA_RESULTS_2026-08-31.md`
- `memory_network/DEVELOPMENTAL_STREAM_PROPOSAL.md`
- `memory_network/developmental_stream_design.md`
- `memory_network/developmental_stream_experiment.py`
- `memory_network/developmental_stream_result.json`
- `memory_network/DEVELOPMENTAL_STREAM_CLOSEOUT_2026-08-31.md`
- `memory_network/CONTEXT_HANDOFF_2026-08-31.md`

## Standing interpretation discipline

Do not equate low prediction loss with semantic understanding. Do not interpret internal geometry as human concepts. Do not claim biological equivalence. Preserve surprising negative results as evidence. Keep each new conversation centered on one experiment and commit completed work directly to `main`.
