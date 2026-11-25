# Scaling Revalidation V1 — Paragraph-Held-Out Medium Comparison

**Status:** Implemented; not yet run.  
**Purpose:** Re-test the medium-scale dense / direct-low-rank / latent comparison with a separately supplied validation corpus and multiple-seed support.

## Why this experiment exists

The previous 5,000-step report used a 204,955-byte corpus assembled from repeated project material. The current committed corpus.txt is 204,559 bytes, so the exact source artifact used for the earlier run is not presently identical to the current repository file. The old training code used a sequential 90/10 byte split.

A sequential split can put related material on both sides of the boundary. A quick audit of the current corpus found 10 normalized paragraphs in the validation tail that also appeared in the training prefix. This is a concrete overlap warning, not proof that the earlier validation scores were invalid. The old run's original 204,955-byte input should be recovered if exact reproduction is required.

The current corpus does not preserve source-document IDs. Therefore this experiment performs exact normalized paragraph deduplication followed by a deterministic shuffled paragraph-level holdout. It does **not** claim document-level independence: paragraphs from the same original source may still land in both partitions.

## Changes

- Adds --val-data to latent_transformer_v0.py. When supplied, training and validation files are loaded independently; the old sequential split remains the default for historical experiments.
- Adds revalidate_scaling.py, which removes exact normalized duplicate paragraphs, creates deterministic train/validation files, records a manifest, and runs the matched medium-scale comparison.
- Defaults to one seed (1337) as a practical first pass. Additional seeds can be supplied for replication.
- Uses the previous medium configuration: 6 layers, width 384, feed-forward width 1536, 6 heads, rank 48, latent dimension 16, context 256, batch 16, learning rate 3e-4.
- Defaults to 5,000 steps and evaluation every 100 steps. Every run writes its own console log and model report. progress.json is updated after each completed run.
- Stops on the first failed run and refuses to overwrite existing reports unless --force is supplied.

## First-run command

From the repository root:

    python3 revalidate_scaling.py --steps 5000 --seeds 1337 --device auto

On the AMD ROCm setup used previously, the script sets HSA_OVERRIDE_GFX_VERSION=10.3.0 if it is not already set. It prints the selected device and dtype at the start of each model run. Confirm that it reports the intended GPU before allowing the full run to continue; if it selects CPU, stop with Ctrl+C and rerun using the correct PyTorch/ROCm environment.

## Recommended staged use

1. First check only corpus preparation:

       python3 revalidate_scaling.py --prepare-only

2. Start the one-seed, three-mode 5,000-step comparison:

       python3 revalidate_scaling.py --steps 5000 --seeds 1337 --device auto

3. If the run starts correctly and resource use is acceptable, later replicate with additional seeds in a new output directory:

       python3 revalidate_scaling.py --steps 5000 --seeds 2027 31415 --device auto --output-dir runs/scaling_revalidation_v1_replication

The first run is a pilot/revalidation, not a statistically conclusive study. Additional seeds are required before treating the result as robust.

## Outputs

runs/scaling_revalidation_v1/ contains:

- corpus_split/train.txt
- corpus_split/validation.txt
- corpus_split/manifest.json
- seed_1337/{dense,lowrank,latent}/.../report.json
- seed_1337/{dense,lowrank,latent}/console.log
- progress.json

Compare best validation loss, final validation loss, train/validation divergence, parameter count, checkpoint bytes, throughput, and peak memory where available. Do not compare only final-step loss; retain the best checkpoint and its step in the analysis.

## Interpretation limits

This improves separation over a sequential byte split and removes exact normalized duplicate paragraphs. It does not solve source-document leakage, near-duplicate leakage, topic/domain imbalance, or the limited diversity of the existing synthetic/repeated project corpus. The results remain a small byte-level language-model experiment, not a general language benchmark.

The experiment does not assume that latent/shared-constructor will win. Dense, low-rank, and latent are all required controls.
