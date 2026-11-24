# R5 — R3 Local Latent Diffusion

R5 is the first generative follow-up to R3. It freezes the trained R3
LocalInteractionAutoencoder, encodes CIFAR-10 images into its sixteen local
32-dimensional states, and trains a small transformer denoiser to model the
distribution of those states. Sampling starts from Gaussian noise; deterministic
DDIM steps produce local states that the frozen R3 decoder converts to images.

This is a prototype, not a claim of image quality or speed yet. The first
evaluation is deliberately unconditional and uses CIFAR-10 at 32x32.

## Prerequisites

- The R3 source file in this repository.
- A locally saved R3 checkpoint. The checkpoint is not committed to GitHub:
  `artifacts/real_cifar10_local_interaction/r3_local_interaction_40000.pt`
  is the default path, but pass `--checkpoint` if yours is elsewhere.
- Either the CIFAR-10 Parquet data used by R3, or a saved R3 local-state array shaped `[N,16,32]`.
- PyTorch, NumPy, pandas, Pillow, and a PyArrow Parquet engine.

ROCm PyTorch exposes AMD GPUs through `torch.cuda`; the script prints the
detected device and ROCm version. It intentionally uses ordinary PyTorch
operations rather than CUDA-specific extensions.

## Run

From the repository root:

```bash
python artifacts/R5_latent_diffusion.py \
  --data-root /home/xetal/Desktop/latent_model_test/cifar10 \
  --checkpoint artifacts/real_cifar10_local_interaction/r3_local_interaction_40000.pt
```

To leave it training after closing the terminal:

```bash
mkdir -p runs/r5
nohup python -u artifacts/R5_latent_diffusion.py \
  --data-root /home/xetal/Desktop/latent_model_test/cifar10 \
  --checkpoint artifacts/real_cifar10_local_interaction/r3_local_interaction_40000.pt \
  > runs/r5/train.log 2>&1 & echo $! > runs/r5/pid
```

Watch progress with `tail -f runs/r5/train.log`.

## Reuse saved R3 local states

If you already have an R3 export such as `local_states_40000.npy`, you can
train R5 without the original Parquet images or re-encoding:

```bash
python artifacts/R5_latent_diffusion.py \\
  --latents /path/to/local_states_40000.npy \\
  --checkpoint /path/to/r3_local_interaction_40000.pt
```

The file must contain a finite float-compatible array shaped `[N,16,32]`.
R5 validates this shape and records the latent source and number of examples
in `r5_results.json`. When using a saved evaluation export, describe the run
as training on those saved evaluation representations—not as training on
the number of images in the R3 checkpoint filename.

## Outputs

By default, outputs go to `artifacts/real_cifar10_latent_diffusion/`:

- `r3_latents_40000_12345.pt`: cached frozen-R3 encodings.
- `r5_latent_diffusion.pt`: denoiser checkpoint, updated every five epochs.
- `samples_epoch_010.png`, etc.: generated image grids every ten epochs.
- `r5_results.json`: final configuration, timing, and noise-loss history.

When using `--data-root`, the first run encodes images and later runs reuse the cache unless
`--reencode` is passed. With `--latents`, R5 loads the supplied saved states directly. Default training is 100 epochs, 500 diffusion steps,
and 50 DDIM sampling steps. Reduce `--epochs` for a quick smoke run.

## Interpretation boundary

R3's reconstruction metrics do not predict generative quality. R5 must be
judged by whether samples are recognizable, varied, non-duplicated, and
produced at useful speed. Compare generated grids across epochs and inspect
sampling time separately from training time. The first run is unconditional;
class conditioning and larger image sizes are later experiments.
