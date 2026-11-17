# Persistent Structure — Milestone B: strict bounded ingestion

This milestone turns the metadata audit into a bounded sample builder. It does
not build the full 1–5 GB corpus and does not claim legal clearance.

## What it does

- Resolves the requested Hugging Face dataset revision to an immutable commit SHA.
- Streams records and applies a deterministic bounded shuffle.
- Accepts only exact strings in the explicit allowlist.
- Rejects composite license strings, aliases, missing values, and unknown values.
- Requires identifier, title, and creator for exact `CC-BY-4.0` records.
- Conservatively normalizes Unicode to NFC and line endings to LF.
- Preserves indentation, tabs, and other whitespace (important for code).
- Rejects NUL-containing, empty, non-string, and oversized text.
- Deduplicates using SHA-256 of normalized text.
- Writes provenance-preserving `corpus.jsonl` and a hash-bearing `manifest.json`.
- Records source revision, sampling parameters, license distribution, rejection
  reasons, collection/language distributions, and output hashes.

## Exact initial allowlist

```text
CC0-1.0
CC-BY-4.0
MIT
Apache-2.0
BSD-2-Clause
BSD-3-Clause
```

This list deliberately does NOT treat `CC0` as `CC0-1.0`, `CC-By` as
`CC-BY-4.0`, or `Public Domain` as a blanket approval. Composite strings such
as `MIT, BSD-3-Clause` are rejected because the metadata does not explain
whether the terms are alternatives, cumulative obligations, or file-level
scanner output. The allowlist is a project policy gate, not a legal opinion.

## Run

From the directory containing `corpus_ingest.py`:

```bash
python -m pip install datasets huggingface_hub

python corpus_ingest.py \
  --dataset PleIAs/common_corpus \
  --split train \
  --revision main \
  --max-records 100000 \
  --shuffle-buffer 5000 \
  --seed 1337 \
  --target-mb 20 \
  --max-document-mb 1 \
  --output-dir persistent_structure_corpus_sample
```

The script resolves `main` to a commit SHA before streaming and records the
resolved SHA in every output record and the manifest. For strict reruns, pass
that recorded SHA to `--revision`.

The scan is bounded by `--max-records`; the output is bounded by `--target-mb`.
If the next JSONL record would exceed the byte target, ingestion stops rather
than writing a partial record.

## Tests

```bash
python -m unittest -v
python -m py_compile corpus_ingest.py test_corpus_ingest.py
```

## Important limitations

- A matching metadata string is not legal clearance.
- The source's underlying license notices and rights may need independent
  verification, especially for software snippets and public-domain claims.
- The dataset's bounded audit sample is not an exhaustive license inventory.
- Public Domain, CC0 aliases, ambiguous labels, composite expressions, and
  unknown values remain rejected unless separately reviewed and deliberately
  added to policy.
- CC-BY rows require minimum attribution metadata; this does not itself prove
  that every attribution obligation has been satisfied.
- This creates a small research sample only. Do not treat it as the final
  training corpus.
