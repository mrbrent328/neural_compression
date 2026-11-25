# E45 Phase 3C — Results interpretation and E46 design consequences

**Artifact analyzed:** `e45_phase3C_summary(1).json`  
**Experiment:** `E45_persistent_latent_memory_phase3C_sweep`  
**Runs:** 60 runs = 5 training-domain counts × 4 payload widths × 3 seeds.  
**Training budget:** 8,192 records per cell. **Evaluation:** two seen and two unseen stores, 1,024 records per store.

## 1. What the sweep establishes

Across all 20 factorial cells and all three seeds:

- Top-1 record retrieval was 100%.
- Top-1 store retrieval was 100%.
- The address/retrieval path therefore remained reliable under every tested setting.
- The main trade-off is in value reconstruction, not locating the record.
- Wider payloads sharply improve reconstruction on held-out projection domains.
- Increasing training-domain count helps the narrow-payload conditions, but payload width is the dominant factor in this sweep.

### Cell means

| Training domains | Payload | Seen cosine | Unseen cosine | Unseen − seen | Raw/latent ratio | Bytes/record |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 0.9755 | 0.3247 | -0.6508 | 4.592× | 55.75 |
| 2 | 16 | 0.9882 | 0.7137 | -0.2746 | 3.471× | 73.76 |
| 2 | 24 | 0.9957 | 0.8606 | -0.1351 | 2.790× | 91.76 |
| 2 | 32 | 0.9991 | 0.9699 | -0.0293 | 2.332× | 109.78 |
| 4 | 8 | 0.8997 | 0.3596 | -0.5401 | 4.592× | 55.75 |
| 4 | 16 | 0.9494 | 0.6867 | -0.2627 | 3.471× | 73.76 |
| 4 | 24 | 0.9875 | 0.8682 | -0.1192 | 2.790× | 91.76 |
| 4 | 32 | 0.9997 | 0.9967 | -0.0030 | 2.332× | 109.78 |
| 8 | 8 | 0.7421 | 0.4573 | -0.2848 | 4.592× | 55.75 |
| 8 | 16 | 0.8881 | 0.6933 | -0.1947 | 3.471× | 73.76 |
| 8 | 24 | 0.9630 | 0.8645 | -0.0985 | 2.790× | 91.76 |
| 8 | 32 | 0.9996 | 0.9981 | -0.0016 | 2.332× | 109.78 |
| 16 | 8 | 0.6341 | 0.4594 | -0.1746 | 4.592× | 55.75 |
| 16 | 16 | 0.8299 | 0.6787 | -0.1512 | 3.470× | 73.77 |
| 16 | 24 | 0.9361 | 0.8526 | -0.0836 | 2.790× | 91.77 |
| 16 | 32 | 0.9995 | 0.9989 | -0.0006 | 2.332× | 109.78 |
| 32 | 8 | 0.6192 | 0.4854 | -0.1338 | 4.592× | 55.75 |
| 32 | 16 | 0.8062 | 0.6854 | -0.1209 | 3.470× | 73.77 |
| 32 | 24 | 0.9245 | 0.8556 | -0.0689 | 2.790× | 91.77 |
| 32 | 32 | 0.9995 | 0.9992 | -0.0003 | 2.332× | 109.78 |

Values are means over three seeds. The uploaded JSON is the source of record for standard deviations, minima, maxima, MSE, and per-run results.

## 2. Interpretation

### Strongest supported result

The memory can reliably identify the correct record and store after process restart in this synthetic setup. That is distinct from reconstructing the stored value: retrieval is effectively solved for these tested conditions, while reconstruction quality depends strongly on the payload bottleneck.

### Payload width is the dominant lever

At 32 payload dimensions, unseen-domain cosine is 0.9699–0.9992 across the five domain-count settings. At 8 dimensions it is only 0.3247–0.4854. The 32D payload is the full-width learned autoencoder for the 32D value, not an identity/raw-value oracle. Its storage ratio is also the weakest of the four settings: 2.332× rather than 4.592× at 8D.

This is a useful engineering frontier, not proof that the narrowest representation preserves arbitrary meaning. A practical system must choose payload width from a task-quality/storage curve rather than optimize compression ratio alone.

### More training domains help the narrow bottleneck, but do not erase it

For payload 8, unseen cosine rises from 0.3247 at two training domains to 0.4854 at 32. For payload 16, the trend is not monotonic (0.7137 at two domains versus 0.6854 at 32). For payload 24, unseen cosine stays in a relatively narrow 0.8526–0.8682 band. At payload 32, all settings are near 1.0.

Therefore, do not summarize the result as “more domains always improves generalization.” The effect depends on payload width and the seen-domain fit also changes.

### The storage ratio needs careful wording

The reported raw-to-latent ratio includes the shared codec checkpoint, memory files, and metadata under the experiment's accounting. It excludes runtime, filesystem allocation, and indexing overhead beyond listed metadata. It compares synthetic float32 key/value records with this particular latent representation; it is not a comparison against a production vector database, compressed text, or an LLM's native context.

## 3. Guardrails

This remains a synthetic projection-domain ablation. Held-out domains are random projections from the same generator family. There are three seeds, which provide an initial replication estimate rather than a definitive statistical conclusion. Continual writes, deletion, conflicts, interference under updates, forgetting, natural language, and pretrained models are not tested.

Do not claim:
- natural-language semantic generalization;
- arbitrary-domain generalization;
- a pretrained model has learned to use this memory;
- cross-model portability;
- superiority to text or vector-database memory;
- full internal-state restoration.

## 4. E46 decision

**E45 Phase 3C is complete enough to unblock a carefully staged E46.** It does not justify treating the 8D codec as language-ready. The E46 experiment should carry multiple payload widths as explicit conditions:

- **8D:** high-compression / low-fidelity stress condition.
- **16D and 24D:** intermediate trade-off conditions.
- **32D:** high-fidelity reference condition, with the warning that it removes the dimensionality bottleneck and is not a raw-value oracle.

The first E46 objective is not to prove cross-model memory. It is to determine whether a frozen language model can use a persisted memory to answer questions it could not answer from its current prompt alone.

## 5. Proposed E46 stages

### E46-A — Black-box persistent-memory task

Use a local chat-completion API only for the language-model interaction. Build a small, deterministic fact-recall dataset with:
- facts written in session A;
- process termination and restart;
- questions in session B;
- exact-match and semantic/relational questions;
- unanswerable questions that require abstention.

Compare at least:
1. no memory;
2. retrieved raw-text memory;
3. retrieved-and-compressed text baseline, with full byte accounting;
4. shuffled/mismatched memory controls.

This stage establishes that the task and evaluation are sensitive. It does **not** test the E45 latent codec.

### E46-B — Representation-interface feasibility

The OpenAI-compatible chat endpoint is a text interface. It does not, by itself, expose transformer hidden states or permit residual-stream injection. Do not pretend that a successful `/v1/chat/completions` call validates a latent-memory adapter.

Select a local model/runtime that explicitly exposes the required activation hooks. First verify, separately:
- extraction of a chosen layer's contextual representation;
- deterministic save/reload of that representation;
- injection of a retrieved representation at a documented model boundary;
- unchanged base-model weights.

If these hooks are unavailable, remain at E46-A rather than silently substituting text prompting for latent integration.

### E46-C — Same-model latent memory

Freeze one base model. Train only small write/query/read adapters. The intended flow is:

```text
WRITE
fact/context → frozen model → contextual state
             → writer adapter → address + payload
             → persistent store

READ
question → frozen model → query state
         → query adapter → address search
         → retrieved payload → reader adapter
         → documented model-conditioning interface → answer
```

Keep address retrieval and value interpretation as separately measurable stages. The retrieved memory must be the only route by which the held-out fact becomes available. Include no-memory, shuffled, mismatched, and raw-state controls.

### E46-D — Cross-model portability

Only after E46-C works, introduce a second independently started model. Treat each model's query/write/read adapters as model-specific. A shared memory format or shared codec is a hypothesis to test, not an assumption. Test A-writes/B-reads and B-writes/A-reads separately.

## 6. Primary measurements

- Recall accuracy split by exact, paraphrase, relational, and unsupported questions.
- Abstention precision/recall on unanswerable questions.
- Record retrieval accuracy, store retrieval accuracy, and value reconstruction quality.
- Total persistent bytes: records, codec, adapters, indexes, metadata, and any required source material.
- Write/read latency, model inference latency, peak RAM/VRAM, and adapter training cost.
- Same-model restart and cross-model transfer as separate results.

## 7. Immediate next action

Do not run a 20B model or train a new base model yet. First implement E46-A's deterministic task harness and baseline conditions, then confirm the chosen runtime can expose the activation/conditioning boundary needed for E46-C. The user's RX 6600 has 8 GB VRAM, so begin with a small supported model/runtime for interface validation; model size is a later scaling variable.

**Decision rule:** if E46-A shows the task is measurable and E46-B confirms a real activation path, proceed to E46-C. If not, stop and repair the interface/evaluation rather than claiming latent memory integration.
