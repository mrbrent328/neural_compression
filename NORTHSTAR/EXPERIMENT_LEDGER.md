# Northstar — Experiment Ledger

This is the canonical working index. Update it whenever an experiment is proposed, changed, executed, evaluated, or retired. Keep entries concise but traceable.

## Active research lead

**Working description:** Shared-constructor language-model approach.

**Question:** Can compact per-structure learned state, interpreted by shared constructor machinery, support useful next-token learning while reducing persistent parameter storage without unacceptable quality, compute, or temporary-memory costs?

**Current source trail (historical labels retained only for traceability):**
- `latent_transformer_v0/`: integrated prototype, checkpoints, reports, samples, and inference diagnostics.
- `experiments/E49_LLM_SCALE.md`: later matched dense / direct-low-rank / shared-constructor comparison plan and implementation notes.
- `NEURAL_COMPRESSION_AUDIT.md`: broader research synthesis and rationale.

**Status:** Selected as the current lead for Northstar. This is a prioritization decision, not evidence that the approach works. The existence of scripts, checkpoints, or documentation does not prove that a particular run was executed or validated. Re-inspect exact artifacts before reporting status.

**Next action:** Audit the current implementation and all available run outputs. Establish exactly what exists, what ran, what is reproducible, and the most meaningful next experiment. Do not change the scientific question or silently substitute a different scale/configuration.

## Entry template

Copy this block for each distinct experiment or run family:

### [Human-readable experiment name]
- **Status:** PROPOSED / APPROVED / WRITTEN / TESTED / RUNNING / COMPLETED / FAILED / INCONCLUSIVE / RETIRED
- **Question / hypothesis:**
- **Exact source files:**
- **Code revision (commit SHA):**
- **Resolved configuration:**
- **Dataset / tokenizer / manifest hashes:**
- **Controls:**
- **Run IDs and output paths:**
- **Commands actually executed:**
- **Tests actually executed and outcomes:**
- **Measured results:**
- **Estimates or inferences (clearly separated):**
- **Limitations / deviations:**
- **Decision and rationale:**
- **Next action:**

## Evidence vocabulary

- **SHOWN:** directly demonstrated by the specified executed test/run.
- **SUPPORTED:** evidence exists but is limited, conditional, or not fully replicated.
- **OPEN:** not yet tested or not resolved.
- **FAILED:** the specified criterion was not met.
- **INCONCLUSIVE:** execution or evidence cannot distinguish the hypotheses.
- **RETIRED:** intentionally no longer pursued; preserve the record and reason.

Never upgrade a claim's status based only on code being written.
