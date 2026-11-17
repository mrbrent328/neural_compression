# Scaling Comparison — Summary

Three transformer modes compared at three scales.
Same corpus, tokenizer, steps, batch size, LR schedule, and seed for every run.

## Results

| Scale | Mode | Total params | Ckpt MB | Best val loss | Final val loss | Tokens/s |
|-------|------|-------------|---------|--------------|----------------|---------|
| tiny | dense | 590,464 | 2.4 | 2.4425 | 2.4425 | 22,580 |
| tiny | lowrank | 320,128 | 1.3 | 2.8069 | 2.8069 | 23,701 |
| tiny | latent | 249,382 | 1.0 | 2.5891 | 2.5891 | 19,084 |
| small | dense | 4,327,680 | 17.3 | 1.8854 | 1.8854 | 2,924 |
| small | lowrank | 1,919,232 | 7.7 | 2.4092 | 2.4092 | 4,141 |
| small | latent | 1,240,412 | 5.0 | 2.4081 | 2.4081 | 4,851 |
| medium | dense | 14,357,376 | 57.4 | 1.2955 | 1.2955 | 1,478 |
| medium | lowrank | 5,399,424 | 21.6 | 2.3247 | 2.3247 | 2,960 |
| medium | latent | 3,799,106 | 15.2 | 2.3308 | 2.3308 | 2,440 |

## Parameter breakdown (latent mode)

| Scale | Constructor | Latents (z) | Attention | Other | Total |
|-------|------------|-------------|-----------|-------|-------|
| tiny | 52,080 | 54 | 131,072 | 66,176 | 249,382 |
| small | 58,256 | 204 | 1,048,576 | 133,376 | 1,240,412 |
| medium | 58,256 | 306 | 3,538,944 | 201,600 | 3,799,106 |

## Notes

* `rank` and `latent_dim` are identical across lowrank and latent at each scale.
* Dense has no rank/latent parameters — its MLP size is `d_model × d_ff × 2 + d_ff × d_model`.
* Constructor parameter count is the same regardless of model depth or width.
* Checkpoint bytes include optimizer state only for the saved model weights.
