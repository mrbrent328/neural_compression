# Persistent Structure — Milestone A metadata audit

This is the first bounded implementation step for the strict corpus workflow.
It inspects dataset metadata only. It does **not** download the full corpus,
write training text, approve licenses, or claim legal clearance.

## Run

```bash
python -m pip install datasets
python corpus_audit.py \
  --dataset PleIAs/common_corpus \
  --split train \
  --revision main \
  --max-records 1000 \
  --shuffle-buffer 5000 \
  --seed 1337 \
  --output corpus_audit_report.json
```

`--shuffle-buffer 5000` performs a deterministic bounded shuffle of the stream.
The report records the requested revision and sampling parameters. For a
reproducible ingestion run, replace `main` with a resolved dataset commit hash.

For an offline unit-test run:

```bash
python -m unittest -v
python -m py_compile corpus_audit.py test_corpus_audit.py
```

## Safety properties

- Streaming access; only up to `--max-records` are inspected.
- No document text is written to the report or printed to the terminal.
- Exact license strings are counted; no fuzzy matching or license equivalence.
- Missing and blank license metadata are visible and remain unapproved.
- The report explicitly marks itself inspection-only.
- This sample is not an exhaustive license inventory; it cannot authorize a full build.

## Next gate

Review the resulting report and inspect the dataset's exact license strings and
provenance. Only then define a small, explicit, reviewed allowlist. Any value not
explicitly approved must remain rejected by default.
