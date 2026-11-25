# E46 — Pretrained-model memory integration (research plan)

**Status:** PLANNED; no pretrained-model integration result exists yet.  
**Backend selected:** LM Studio (local OpenAI-compatible server).  
**Dependency:** E45 Phase 3C must be analyzed before treating the current latent value codec as ready for language-model integration.

## Why E46 exists

E45 has tested persistent address/payload memory on synthetic vectors. It has not tested natural-language memories or a pretrained language model. E46 asks whether a frozen pretrained model can use a persistent memory substrate through a small interface, without training a new base language model.

The repository's current E45 Phase 3B result is mixed: pooled record and store retrieval were 100%, while mean cosine reconstruction was 0.7348 on seen projection domains and 0.4502 on unseen domains. Phase 3C was added to isolate training-domain diversity and payload-width effects. Its smoke test passed, but the full factorial sweep remains unrun. E46 should not quietly treat that unresolved value-path weakness as solved.

## Staged plan

### Gate 0 — Finish the E45 value-path diagnosis

Recommended first run: a 12-cell diagnostic subset of the existing Phase 3C design:

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

### Gate 1 — LM Studio connection smoke test

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

### Gate 2 — Establish a language-memory baseline

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

### Gate 3 — Connect the learned latent substrate

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

Run the 12-cell E45 Phase 3C diagnostic subset, inspect the aggregate, then run the LM Studio connection harness. Keep the full 60-cell run available if the diagnostic does not resolve the value-path question.

## References

- LM Studio OpenAI-compatible endpoints: https://lmstudio.ai/docs/developer/openai-compat
- LM Studio API overview: https://lmstudio.ai/docs/developer/rest
