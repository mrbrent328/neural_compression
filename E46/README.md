# E46 — Pretrained-model memory integration (research plan)

**Status:** E46-A benchmark implemented; E46-B activation-level adapter scaffold committed. No pretrained-model activation/injection result exists yet.  
**Host candidate:** Qwen3-1.7B-Base via a direct Transformers/PyTorch runtime. LM Studio remains a text-only baseline transport, not the activation integration backend.  
**Dependency:** E45 Phase 3C has now been analyzed. It supports a staged E46, but does not establish natural-language or pretrained-model memory.

## Why E46 exists

E45 has tested persistent address/payload memory on synthetic vectors. It has not tested natural-language memories or a pretrained language model. E46 asks whether a frozen pretrained model can use a persistent memory substrate through a small interface, without training a new base language model.

E45 Phase 3C completed its full 60-run factorial sweep (five training-domain counts × four payload widths × three seeds). Record and store retrieval were 100% in every cell. Value reconstruction on unseen synthetic projection domains depended strongly on payload width: mean cosine ranged from 0.3247–0.4854 at 8D, 0.6787–0.7137 at 16D, 0.8526–0.8682 at 24D, and 0.9699–0.9992 at 32D. The 32D condition is a full-width learned autoencoder, not an identity/raw-value oracle; it gives the best fidelity but the weakest storage ratio (2.332×). See [E45_PHASE3C_ANALYSIS.md](E45_PHASE3C_ANALYSIS.md) for the full matrix, interpretation, and guardrails.

## Staged plan

### Gate 0 — E45 value-path diagnosis (complete)

The full 60-run sweep is now complete. The result resolves the immediate payload-width question sufficiently to proceed with a cautious integration study. The 12-cell diagnostic subset below is retained only as a reproducibility option, not as the next required action:

```bash
python E45/e45_latent_memory_phase3c.py --device cpu \
  --seeds 45 46 47 \
  --domain-counts 2 32 \
  --payload-dims 8 32 \
  --output runs/E45_phase3C_diag12
```

This crosses low/high training-domain diversity with narrow/full-width learned payloads over three seeds. It is a diagnostic subset, not a replacement for the registered 60-cell factorial. It writes to a separate output directory so it cannot overwrite a prior Phase 3C run. If the pattern is ambiguous, run the complete sweep:

```bash
python E45/e45_latent_memory_phase3c.py --device cpu
```

Decision criteria:
- Confirm record/store retrieval remains reliable.
- Compare seen/unseen cosine and MSE.
- Determine whether wider payloads and/or more training domains close the gap.
- Judge quality together with complete storage cost.
- Preserve partial results and do not claim arbitrary-domain generalization.

### Gate 1 — LM Studio connection smoke test (optional transport check)

LM Studio documents an OpenAI-compatible `/v1/models` endpoint and `/v1/chat/completions` endpoint. The default local base URL is `http://localhost:1234/v1`.

Start the local server in LM Studio's Developer tab. Load the intended chat model, then from the repository root run:

```bash
python E46/lmstudio_smoke.py
```

The script lists models, prefers an available `gpt-oss-20b` model ID, otherwise selects the first listed model, sends one deterministic request, reports latency and usage, and saves a JSON record under `runs/E46/lmstudio_smoke.json`. To select a model explicitly:

```bash
python E46/lmstudio_smoke.py --model "MODEL_ID_FROM_LM_STUDIO"
```

If server authentication is enabled, pass `--api-key "$LM_API_TOKEN"`. To use a different host or port, pass `--base-url`.

This is an integration check only. It does not test latent memory, hidden-state access, or learning. The script uses only Python's standard library.

### Gate 2 — E46-A: establish a black-box language-memory baseline

Before claiming that E45's latent codec works for language, create a small, reproducible task with:
- facts presented in one session;
- a genuine process restart;
- questions asked in a later session;
- a no-memory baseline;
- a conventional text-memory retrieval baseline;
- a shuffled or mismatched-association control;
- exact accounting of stored text, embeddings, codec weights, adapters, and metadata.

The task must distinguish verbatim recall, paraphrase, relational questions, and unsupported questions. The model must be told when evidence is absent; guessing is scored as an error.

The text-memory baseline is a control, not evidence that latent compression works.

### Gate 3 — E46-B/C: verify activation access, then connect the learned latent substrate

Only after Gates 0–2:
- Freeze the base language model.
- Train only small interface components.
- Keep the shared memory substrate separate from model-specific adapters.
- Separate address retrieval from payload interpretation/conditioning.
- Compare write/read across independently started model sessions.
- Add a second model only after the first-model interface works.

LM Studio's OpenAI-compatible chat API is a black-box text interface. Do not assume it exposes transformer hidden states, attention activations, or trainable internal hooks. If the chosen integration requires those, use an explicitly supported model/runtime path for activation access; do not pretend the chat endpoint provides it.

## Required controls and measurements

- No-memory baseline.
- Conventional text retrieval baseline.
- Latent-memory condition.
- Shuffled records and mismatched key/payload associations.
- Process restart and persisted-file reload.
- Recall accuracy by task type; unsupported-answer rate.
- Storage: raw content, latent records, codec, adapters, metadata, indexes.
- Runtime: write/read latency, inference latency, peak RAM/VRAM, and compute.
- Adapter parameter count and training budget.
- Cross-model transfer is a separate result, not inferred from same-model persistence.

## Scientific boundaries

A successful LM Studio connection is not a memory result. A successful text-memory task is not proof of latent compression. A successful single-model test is not cross-model portability. A smaller latent file is not a system-level storage win unless codec, adapters, metadata, indexes, and required source material are included.

No base-model retraining is planned for the initial stages. Small interface training may still be required later.

## Current next action

E46-A is implemented as a black-box text-memory control. E46-B now has an initial activation-level adapter scaffold in [E46B_MEMORY_ADAPTER.md](E46B_MEMORY_ADAPTER.md) and [e46b_memory_adapter.py](e46b_memory_adapter.py), with CPU unit tests under `tests/`.

Next: validate the adapter against the locally downloaded Qwen3-1.7B-Base checkpoint using a direct Transformers/PyTorch runtime. Confirm the exact decoder-block path, capture shape, hook output contract, persistence across a fresh process, and unchanged base-model weights. Only after this integration boundary works should E46-C train the writer/query/reader interface. The current adapter is an untrained scaffold: a zero-initialized gate intentionally makes it behavior-neutral, and no memory-utility or compression result is claimed.

## References

- LM Studio OpenAI-compatible endpoints: https://lmstudio.ai/docs/developer/openai-compat
- LM Studio API overview: https://lmstudio.ai/docs/developer/rest
\n\n## E45 Phase 3C decision\n\nThe full sweep supports carrying 8D/16D/24D/32D payload widths as explicit quality/storage conditions into later work. Use 32D as the high-fidelity reference and 8D as the high-compression stress condition; do not assume a narrow payload generalizes. E46-A remains a text-interface task and must not be conflated with the latent path. E46-C requires a runtime with actual activation access and a documented conditioning interface.\n

## E46-A — black-box persistent text-memory benchmark

The first language-facing gate is implemented in `E46/e46a_text_memory.py`. It creates eight fictional fact records and 32 questions split evenly across exact recall, paraphrase, relational lookup, and unsupported questions.

Run the preparation command, terminate that process, then run evaluation from a fresh process:

```bash
python E46/e46a_text_memory.py prepare
# End the process. Start a fresh shell/process:
python E46/e46a_text_memory.py evaluate
```

The evaluator uses the OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints. Defaults to `http://localhost:1234/v1`; pass `--model MODEL_ID` if automatic model selection is not appropriate. Optional authentication uses `LM_API_TOKEN`.

Conditions:
- no memory;
- raw-text memory;
- shuffled/mismatched key-to-payload memory.

It records per-category exact-match accuracy, response latency, raw outputs, and basic text-store byte counts in `runs/E46A/evaluation.json`.

**Interpretation boundary:** this is a deterministic synthetic task to validate the persistence/evaluation harness and model behavior. It does not test latent compression, hidden-state access, semantic generalization, or cross-model portability. One batched call per condition is a smoke-level measurement; repeat across models/seeds before making broad claims.
