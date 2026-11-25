# E46-B4 — Entity-disjoint generalization and memory scaling

## Research question

Does the learned address space transfer to facts about entity identifiers that were never used in training, and how does retrieval change as the candidate memory bank grows?

## Dataset

The deterministic generator creates 256 records across eight categories (32 entities each):
- 192 training entities (24/category), each with two training query phrasings and one validation phrasing.
- 64 unseen entities (8/category), each with two final test phrasings.
- Every record has one separate capture prompt.

The unseen entity IDs and their test queries are excluded from gradient updates and checkpoint selection. Validation queries are only used for checkpoint selection. The final unseen-entity test is not used for model selection.

## Run

Generate the specification:

```bash
python E46/e46b_b4_generalization.py generate-spec
```

Capture Qwen activations (CPU default; about 960 model prompts):

```bash
python E46/e46b_b4_generalization.py collect \
  --model-path "/home/xetal/Models/Qwen3-1.7B-Base"
```

Train the address writer and query encoder, then build a 256-record persistent bank:

```bash
python E46/e46b_b4_generalization.py train
```

The report is written to `runs/E46B/B4/adapter/e46b4_report.json`.

## Metrics

- Validation top-1 on the 192 training entities.
- Final test top-1, mean reciprocal rank, mean positive rank and positive-vs-runner-up cosine margin against the complete 256-record bank.
- Unseen-only bank evaluation (64 records).
- Candidate bank scale sweep at 8, 32, 128 and 256 records. Each query's true memory is guaranteed to be present; distractors are sampled deterministically.

The scale sweep is a controlled sampled-candidate test, not a single nested bank-prefix test. Each size is sampled independently per query.

## Guardrails

- Qwen is frozen; only address_writer and query_encoder are trained.
- Payload writer/reader are not optimized. This phase tests addressing only.
- Entity-disjoint evaluation is stronger than paraphrasing known entities, but this is still synthetic, templated data. It does not establish open-domain semantic generalization.
- The 256 records are small enough for a first CPU experiment. A larger and less templated corpus should follow if this gate succeeds.
