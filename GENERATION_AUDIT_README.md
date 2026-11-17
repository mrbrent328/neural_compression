# Neural Compression — Generation Audit

This is a read-only inference experiment for the three existing `latent_transformer_v0` checkpoints.
It does **not** train, modify, or overwrite any model checkpoint.

## 1. Put the script in the repository

Copy `generation_audit.py` into the root of `neural_compression/`, beside `corpus.txt` and the `latent_transformer_v0/` directory.

Expected checkpoint layout:

```text
neural_compression/
├── corpus.txt
├── generation_audit.py
├── latent_transformer_v0/
│   └── latent_transformer_v0.py
└── runs/
    └── latent_transformer_v0/
        ├── dense_s1337/model.pt
        ├── lowrank_s1337/model.pt
        └── latent_s1337/model.pt
```

## 2. Run

On the AMD ROCm setup used for the training runs:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python generation_audit.py --device auto
```

If the checkpoints are in a different location, use `--run-root` or explicit `--dense`, `--lowrank`, and `--latent` paths.

## 3. What it tests

- Five fixed prompts: two documentation-like English prompts, one technical requirement prompt, one C# prompt, and one Python prompt.
- Four prompts sampled from the final 10% held-out tail of `corpus.txt`.
- Identical prompt bytes and generation length for all three models.
- Greedy decoding, matching the existing model's built-in generation method.
- The actual held-out continuation is included in the report for side-by-side human inspection; it is never shown to the model.
- Checkpoint path, SHA-256, file size, model configuration, and parameter counts are recorded.

The script writes timestamped JSON and TXT reports under `runs/generation_audit/`.

## 4. How to judge it

For each model, compare:

1. **Validity:** are the bytes readable text, or mostly replacement characters / fragments?
2. **Local structure:** do identifiers, punctuation, syntax, and word boundaries make sense?
3. **Continuation:** does it continue the prompt in a plausible way?
4. **Consistency:** does it behave similarly across all prompts, or collapse into repetition?
5. **Held-out comparison:** does it resemble the actual continuation in style and structure without simply copying the prompt?

This is a qualitative generation audit, not a benchmark of general intelligence or broad English ability. The training corpus is source-code-heavy, so code/documentation continuation is the fairest interpretation.

## 5. Suggested two-hour session

- 10 min: place the script and check the three checkpoint paths.
- 5–15 min: run the audit.
- 30–45 min: inspect outputs side by side and mark obvious failures / strengths.
- 20 min: compare the held-out references and identify whether failures are shared or latent-specific.
- Remaining time: decide the next experiment from the evidence (prompt/data mismatch, constructor capacity, or inference optimization).

Keep the generated JSON/TXT reports with the experiment artifacts. Do not overwrite the original checkpoints or training reports.
