# Latent Transformer v0 vs v1 Structural Atoms — First Matched Comparison

**Date:** 2026-11-24  
**Status:** Two completed runs; one matched seed (1337). Exploratory result, not a multi-seed conclusion.

## Question

Does a shared sparse dictionary of structural atoms improve the learned latent MLP constructor compared with the v0 coordinate-conditioned constructor?

## Matched protocol

Both runs used the same:

- Corpus: 204,955 bytes; byte-level vocabulary (256).
- Seed: 1337.
- Device/dtype: CPU / float32.
- Model: 6 layers, width 384, 6 attention heads, feed-forward width 1,536, context 256.
- Factor rank 64; per-matrix latent dimension 32.
- Constructor hidden width 128, type embedding width 16, and 8 Fourier bands.
- 5,000 optimizer steps; 20,480,000 tokens processed.
- Training schedule, optimizer, batch size, evaluation cadence, and inherited contiguous 90/10 train/validation split.

The only intended architectural change was the MLP constructor. v0 uses its shared coordinate constructor. v1 uses a shared sparse structural-atom dictionary with 32 atoms, atom dimension 16, top-k 4, and temperature 1.0.

## Results

| Measurement | v0 latent | v1 structural atoms |
|---|---:|---:|
| Total parameters | 3,807,618 | 3,805,090 |
| Constructor parameters | 66,480 | 63,952 |
| Matrix-specific latent parameters | 594 | 594 |
| Checkpoint file | 15,255,899 B | 15,246,732 B |
| Best validation loss | 1.655740 | **1.653539** |
| Best validation step | 2,250 | 2,250 |
| Perplexity at best checkpoint | 5.236956 | **5.225439** |
| Final training loss | **0.349649** | 0.350240 |
| Final validation loss | 2.201734 | **2.201038** |
| Final perplexity | 9.040678 | 9.034385 |
| Overall throughput | **3,719.93 tok/s** | 3,222.39 tok/s |
| Elapsed training time | **5,505.48 s** | 6,355.53 s |

### Direct differences

- Structural atoms used **2,528 fewer parameters**: 0.066% fewer total parameters and 3.80% fewer constructor parameters.
- Its checkpoint was **9,167 bytes smaller** (about 0.060%).
- Best validation loss was lower by **0.002202** (about 0.13%) at the same step. Final validation loss was lower by only **0.000696**.
- Structural atoms ran at **13.37% lower overall throughput** and took **850.05 seconds (14 minutes 10 seconds) longer**.
- Final training losses were nearly identical.

## Structural atom usage

Across 18 matrix-specific latents, top-k 4 produced 72 selections. Only **7 of 32 atoms** were selected at least once. Three atoms each received 18 selections (54 of 72 total selections). Selection entropy was 1.666 nats against a maximum of 3.466 nats for a uniform 32-atom distribution.

This records selection concentration, not semantic specialization or proof that unused atoms are unnecessary. It is an observation from one trained model.

## Interpretation

1. **Quality is effectively tied in this run.** The atom model's small validation advantage is too small to establish a meaningful improvement from one seed.
2. **The parameter reduction is real but marginal at whole-model scale.** The constructor has 3.80% fewer parameters, but total parameters fall by only 0.066%, because attention and other components dominate this model.
3. **The atom implementation is slower in this CPU run.** Its sparse top-k composition did not translate into faster training; measured throughput was 13.37% lower.
4. **Atom selection is concentrated.** Seven atoms were used, with three accounting for 75% of selections. Further work can test whether this reflects useful specialization, redundant dictionary capacity, or selector collapse.
5. **Both runs show the same validation pattern.** Best validation occurs at step 2,250, followed by deterioration while training loss continues to fall. The inherited split is a contiguous 90/10 byte split, so this should be described as validation deterioration—not attributed solely to overfitting without a split/generalization control.

## Conclusion

This first matched comparison does **not** demonstrate a meaningful quality, storage, or compute advantage for the structural-atom constructor. It achieves nearly identical language-model metrics, with a very small reduction in constructor parameters and checkpoint size, but slower CPU training.

The result does not reject the atom idea. It narrows the next question: can atom selection and composition be made more efficient or useful, and does the pattern replicate across seeds? A multi-seed comparison and a better-controlled validation split are required before claiming an architectural advantage.

## Limitations

- One seed and one matched run per architecture.
- CPU float32 only; no GPU or inference-latency benchmark.
- Small byte-level corpus and random initialization; not a pretrained model.
- Contiguous 90/10 split; validation deterioration may include corpus-order distribution effects.
- Peak device memory was not reported.
- The report's generated samples are short qualitative artifacts, not a meaningful text-quality evaluation.

## Artifacts

- [v0 latent baseline report](results/latent_baseline_seed1337.json)
- [v1 structural atom report](results/structural_atoms_seed1337.json)
- [v1 structural atom implementation](latent_transformer_v1_structural_atoms.py)
- [Structural atom experiment design](STRUCTURAL_ATOMS_V1.md)
