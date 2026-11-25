# E46-B3 — Supervised activation-space addressing

**Purpose:** train the memory address writer and query encoder to map matching stored-fact activations and later-question activations into a shared retrieval space, while keeping Qwen3-1.7B-Base frozen.

This is a retrieval-addressing experiment only. The payload writer/reader are not trained here, and success on the small synthetic fact set is not evidence of broad semantic memory or answer reconstruction.

## Dataset design

`e46b_b3_dataset.json` defines eight independent facts. Each has:
- one memory/capture prompt,
- two training query phrasings,
- two held-out query phrasings.

The held-out prompts are never used for gradient updates. This tests wording variation within a small controlled vocabulary, not open-domain generalization.

## Run from repository root

1. Capture activation vectors (one Qwen load; CPU is supported):

```bash
python E46/e46b_b3_address_training.py collect \
  --model-path "/home/xetal/Models/Qwen3-1.7B-Base"
```

2. Train only the address writer and query encoder:

```bash
python E46/e46b_b3_address_training.py train
```

The script reports held-out top-1 retrieval accuracy before and after training and retains the checkpoint with the best held-out accuracy.

3. Build a fresh persistent bank using the trained address writer:

```bash
python E46/e46b_b3_address_training.py build-bank
```

4. Test a held-out paraphrase using the existing Qwen injection runner:

```bash
python E46/e46b_qwen_host_check.py inject \
  --model-path "/home/xetal/Models/Qwen3-1.7B-Base" \
  --adapter-dir "runs/E46B/B3/adapter" \
  --gate 0.25 \
  --prompt "Tell me the recorded calibration temperature belonging to ORBIT-17."
```

Repeat for the other held-out questions. Inspect `retrieval_ranking_all_memories`; the expected top record is identified by `capture_id`.

## Interpretation guardrails

- Base model weights are used only to collect activations; they are never optimized.
- Only `address_writer` and `query_encoder` are optimized with cross-entropy over in-batch memory candidates.
- The capture activation is the positive memory target; other facts in the batch are negatives.
- The best checkpoint is selected using held-out query top-1 accuracy.
- Payload encoding remains random/untrained. This experiment evaluates address selection, not whether recalled payloads preserve the fact or improve generated answers.
- With eight facts, results are a proof-of-mechanism only. A later study needs many more facts, multiple random seeds, domain-separated splits, and comparisons against lexical/embedding baselines.
