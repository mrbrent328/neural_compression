# NORTHSTAR

**Northstar** is the permanent working name for the active research effort in this repository. Use this name in conversation and project handoffs. Do not make the user navigate internal experiment IDs, filenames, or branch labels.

## Mission

Investigate whether useful neural behavior can be represented with compact learned state interpreted by shared machinery, and whether that representation can learn language effectively at meaningful scale.

The current lead is the shared-constructor language-model line already represented in the repository. This is a research direction, not a claim of success. Existing versions and controls remain distinct and must never be blended in reporting.

## Canonical project map

- `README.md`: repository-level overview.
- `NORTHSTAR/README.md`: this operating map and collaboration contract.
- `NORTHSTAR/EXPERIMENT_LEDGER.md`: canonical index of hypotheses, exact source/config/result locations, execution status, evidence, and next action.
- `NORTHSTAR/DECISIONS.md`: dated decisions and approved scope changes.
- Existing experiment folders and files: preserved as historical source of truth. Do not rename, move, overwrite, or delete them merely to make the tree look tidy.
- `runs/`: generated run outputs only. Every run must have a unique immutable run directory containing its config, logs, metrics, checkpoint(s), and a source revision reference. Never mix outputs from different configurations.
- `data/`: prepared datasets and manifests. Keep source/revision, tokenizer, split hashes, preprocessing configuration, and licensing/attribution with each dataset. Do not commit large raw datasets or secrets.

## Naming rules

1. Conversation codename: **Northstar**.
2. Human-readable experiment names describe the question, not just a serial number.
3. Every execution gets a unique run ID, for example `YYYYMMDD-HHMMSS_<short-purpose>_<seed>`.
4. Never reuse a run directory. A changed configuration means a new run ID.
5. Preserve historical filenames and IDs in source references so every result remains traceable.

## Experiment contract — required before implementation

Before changing code for a substantive experiment, write down:
- question and falsifiable hypothesis;
- exact model/algorithm and what is held constant;
- dataset, tokenizer, splits, preprocessing, and hashes;
- parameterization, dimensions, optimizer, seed, budget, and hardware;
- controls and comparison method;
- primary metrics and decision threshold;
- what would count as failure or an inconclusive result.

If any of these materially change, stop and tell the user before implementing. Ask for approval when the change alters the agreed research question or comparison. Do not silently replace a planned experiment with a smaller, easier, or different one.

## Evidence and reporting rules

Every update must distinguish:
- **Inspected:** source/config/results were read.
- **Written:** code or documentation was created or changed.
- **Tested:** an exact command was executed; include environment and outcome.
- **Run:** training/evaluation actually executed; include run ID and output paths.
- **Measured:** values came from recorded outputs.
- **Estimated / inferred:** derived from formulas or reasoning, never presented as measurements.

Never say a test passed unless it was executed. Never describe a configuration-derived parameter estimate as a trained model result. Report failures, skipped work, warnings, and limitations alongside successes. Keep negative results.

## Version and artifact integrity

- Record repository commit SHA for every run.
- Save the complete resolved configuration with each run.
- Save data-manifest and tokenizer hashes.
- Preserve original checkpoints and logs; do not overwrite the only copy.
- Make code changes in focused commits with descriptive messages.
- Keep generated artifacts out of source directories.
- Do not commit credentials, private data, or large datasets.

## Collaboration contract

The assistant is responsible for tracking internal labels and project state. The user should not have to remember filenames or correct experiment-number confusion.

When the user's wording is ambiguous, ask one clear question rather than guessing. Use **Northstar** as the shared reference. Explain technical choices plainly, challenge weak assumptions respectfully, and separate evidence from interpretation.

Do not claim this document or a chat memory guarantees perfect continuity. At the start of each Northstar work session, re-read this file and the experiment ledger, inspect the current GitHub default-branch head, and verify the actual source and run artifacts before making claims.
