# Scaling Comparison — 5,000-Step Results

**Experiment:** Dense vs direct low-rank vs latent/shared-constructor transformer  
**Run count:** 9/9 completed successfully  
**Training budget:** 5,000 optimizer steps per run  
**Corpus:** 204,955 bytes (synthetic/repeated project corpus; not a diverse natural-language benchmark)  
**Tokenizer:** byte-level, vocabulary 256  
**Context:** 256  
**Batch size:** 16  
**Seed:** 1337  
**Optimizer:** AdamW (see experiment runner for implementation); learning rate 3e-4 with cosine decay and 100-step warmup  
**Evaluation cadence:** every 100 steps, plus initial step  
**Device/dtype:** CPU / float32 for all nine runs; the Radeon GPU was not used  
**Runner:** `scaling_comparison.py` invoking `latent_transformer_v0/latent_transformer_v0.py`  
**Source archive:** user-supplied `Archive.zip`; per-run reports contained 51 history points (step 1 and every 100 steps through 5,000).

## Model configurations

| Scale | Layers | d_model | d_ff | Heads | Rank | Latent dim |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | 2 | 128 | 512 | 4 | 32 | 8 |
| Small | 4 | 256 | 1,024 | 4 | 48 | 16 |
| Medium | 6 | 384 | 1,536 | 6 | 48 | 16 |

All modes used the same scale dimensions, corpus, tokenizer, context, batch size, seed, and training budget. Dense does not use the rank/latent values; they are retained in the shared configuration.

## Results

Validation loss is lower-is-better. “Best” is the minimum recorded validation loss across the 51 evaluations; “final” is the step-5,000 value.

| Scale | Mode | Parameters | Checkpoint bytes | Best val loss (step) | Final val loss | Final train loss | Final tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Tiny | Dense | 590,464 | 2,373,570 | **1.505116 (4200)** | 1.6267917 | 0.8437374 | 44,433.56 |
| Tiny | Low-rank | 320,128 | 1,293,732 | 1.963467 (4200) | 2.0218817 | 1.8673145 | 46,856.51 |
| Tiny | Latent | 249,382 | 1,014,883 | 2.022574 (4200) | 2.0722180 | 1.9919375 | 35,474.83 |
| Small | Dense | 4,327,680 | 17,326,780 | **1.572457 (1000)** | 2.3777089 | 0.0647355 | 8,743.58 |
| Small | Low-rank | 1,919,232 | 7,695,616 | **1.572341 (4200)** | 1.7250017 | 0.9118215 | 12,012.46 |
| Small | Latent | 1,240,412 | 4,984,511 | 1.667390 (4200) | 1.8112800 | 1.1857375 | 9,333.96 |
| Medium | Dense | 14,357,376 | 57,449,590 | 1.675159 (600) | 2.4476570 | 0.0547321 | 3,243.19 |
| Medium | Low-rank | 5,399,424 | 21,621,788 | **1.566881 (1600)** | 2.4428249 | 0.2268038 | 5,242.33 |
| Medium | Latent | 3,799,106 | 15,224,859 | 1.610339 (1600) | 2.2734089 | 0.3423449 | 4,099.03 |

## Latent parameter breakdown

| Scale | Constructor | Latents (z) | Attention | Other | Total |
|---|---:|---:|---:|---:|---:|
| Tiny | 52,080 | 54 | 131,072 | 66,176 | 249,382 |
| Small | 58,256 | 204 | 1,048,576 | 133,376 | 1,240,412 |
| Medium | 58,256 | 306 | 3,538,944 | 201,600 | 3,799,106 |

## Findings supported by this run

1. **Medium latent vs low-rank:** latent reaches 1.610339 best validation loss versus 1.566881 for low-rank (a 0.043458 gap), with 3,799,106 versus 5,399,424 parameters. The latent model has about 70.4% of low-rank's parameter count.
2. **Small latent vs low-rank:** latent best validation loss is 1.667390 versus 1.572341, with 1,240,412 versus 1,919,232 parameters (about 64.6%).
3. **Tiny latent vs low-rank:** latent best validation loss is 2.022574 versus 1.963467, with 249,382 versus 320,128 parameters (about 77.9%).
4. **Dense remains strongest at tiny scale.** At small and medium scales, low-rank has the lowest best validation loss. The latent model is competitive but does not win best validation loss at any scale in this 5,000-step run.
5. **Storage/parameter reduction:** medium latent uses about 26.5% of dense's parameters and its saved state-dictionary checkpoint is about 15.2 MB versus 57.4 MB for dense.
6. **Throughput:** low-rank is fastest in all three scale groups in these CPU measurements. Latent constructor work does not yield a throughput win in this run.

## Training dynamics / caution

The gap between best and final validation loss is substantial for several runs, especially small and medium dense and medium low-rank. Their training losses continue to fall while validation losses rise. This is consistent with overfitting or a train/validation distribution issue, but these logs alone do not establish the cause. The next investigation should inspect the corpus split, data repetition/leakage, training curves, and checkpoint selection before changing architecture.

The validation-best checkpoints occur at different steps (600–4,200), so final-step comparisons alone would be misleading. Use best-validation checkpoints for the reported best-loss comparison, and preserve final metrics as a separate measure.

## Reproducibility and artifact notes

- Each `report.json` recorded 51 evaluations, seed 1337, device `cpu`, dtype `torch.float32`, and corpus size 204,955 bytes.
- Each run processed 20,480,000 tokens by step 5,000.
- The full per-evaluation metrics for all nine runs are preserved in `scaling_comparison_5000_history.csv` (459 rows plus header): train loss, validation loss, perplexity, elapsed seconds, throughput, tokens seen, and peak device memory field.
- Checkpoint byte counts are saved model state files; optimizer parameter elements are reported separately. They are not optimizer-state-inclusive checkpoint sizes.
- The uploaded archive also contained console logs, generated samples, and nine binary model checkpoints. The archive is approximately 119.7 MB, larger than GitHub's 100 MB per-file limit. Binary checkpoints are therefore not included in this text-data commit; retain the original archive as the source of those binary artifacts. No claim is made that those checkpoint binaries have been uploaded to this repository.
- The earlier 500-step study remains documented separately in `SCALING_RESULTS.md`; this report supplements rather than replaces it.

## Conclusion

This run supports a parameter/storage efficiency result for the latent/shared-constructor model, particularly at medium scale, but does **not** show a validation-loss or throughput win over direct low-rank. The results are also affected by pronounced late-training validation deterioration. Investigate training/validation behavior and repeatability before drawing broader architectural conclusions.
