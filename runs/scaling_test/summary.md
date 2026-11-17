# Scaling Comparison — Summary

Three transformer modes compared at three scales.
Same corpus, tokenizer, steps, batch size, LR schedule, and seed for every run.

## Results

| Scale | Mode | Total params | Ckpt MB | Best val loss | Final val loss | Tokens/s |
|-------|------|-------------|---------|--------------|----------------|---------|
| tiny | dense | 590,464 | 2.4 | 5.2344 | 5.2344 | 13,547 |

## Parameter breakdown (latent mode)

| Scale | Constructor | Latents (z) | Attention | Other | Total |
|-------|------------|-------------|-----------|-------|-------|

## Notes

* `rank` and `latent_dim` are identical across lowrank and latent at each scale.
* Dense has no rank/latent parameters — its MLP size is `d_model × d_ff × 2 + d_ff × d_model`.
* Constructor parameter count is the same regardless of model depth or width.
* Checkpoint bytes include optimizer state only for the saved model weights.
