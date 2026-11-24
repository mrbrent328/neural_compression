#!/usr/bin/env python3
"""R5 post-training diagnostics: compare genuine R3 latents with R5 samples.

This loads the saved R3 and R5 checkpoints. It does NOT train or update either model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from R3_local_interaction_cifar10 import LocalInteractionAutoencoder
from R5_latent_diffusion import LatentDenoiser, alpha_bar, save_grid


def summary(values: torch.Tensor) -> dict:
    a = values.detach().float().cpu().numpy()
    flat = a.reshape(-1)
    by_dim = a.reshape(-1, a.shape[-1])
    return {
        "shape": list(a.shape),
        "overall": {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "q01": float(np.quantile(flat, 0.01)),
            "q05": float(np.quantile(flat, 0.05)),
            "median": float(np.median(flat)),
            "q95": float(np.quantile(flat, 0.95)),
            "q99": float(np.quantile(flat, 0.99)),
        },
        "per_dimension_mean": by_dim.mean(axis=0).tolist(),
        "per_dimension_std": by_dim.std(axis=0).tolist(),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents", type=Path, required=True,
                   help="Saved R3 local states (.npy), normally [10000,16,32]")
    p.add_argument("--r3-checkpoint", type=Path, required=True)
    p.add_argument("--r5-checkpoint", type=Path,
                   default=HERE / "real_cifar10_latent_diffusion" / "r5_latent_diffusion.pt")
    p.add_argument("--out", type=Path,
                   default=HERE / "real_cifar10_latent_diffusion" / "diagnostics")
    p.add_argument("--sample-count", type=int, default=256)
    p.add_argument("--sample-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--grid-scale", type=int, default=8)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"R5 diagnostics | device={device} | torch={torch.__version__} | ROCm={getattr(torch.version, 'hip', None)}", flush=True)

    z = torch.from_numpy(np.load(args.latents.expanduser(), allow_pickle=False)).float()
    if z.ndim != 3 or tuple(z.shape[1:]) != (16, 32):
        raise ValueError(f"Expected [N,16,32] R3 latents, got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise ValueError("R3 latent file contains NaN or infinite values")

    r3 = torch.load(args.r3_checkpoint.expanduser(), map_location="cpu", weights_only=False)
    r5 = torch.load(args.r5_checkpoint.expanduser(), map_location="cpu", weights_only=False)
    arch = r3.get("architecture", {})
    decoder = LocalInteractionAutoencoder(
        local_dim=int(arch.get("local_dim", 32)),
        interaction_layers=int(arch.get("interaction_layers", 1)),
    ).to(device)
    decoder.load_state_dict(r3["model_state_dict"])
    decoder.eval()

    denoiser = LatentDenoiser(
        tokens=int(r5.get("tokens", z.shape[1])),
        dim=int(r5.get("local_dim", z.shape[2])),
    ).to(device)
    denoiser.load_state_dict(r5["denoiser_state_dict"])
    denoiser.eval()

    mean = r5["latent_mean"].to(device)
    std = r5["latent_std"].to(device)
    steps = int(r5["diffusion_steps"])
    sample_steps = int(args.sample_steps or r5.get("sample_steps", 50))
    count = args.sample_count

    # Reproduce the saved R5 deterministic DDIM sampling procedure.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(count, denoiser.tokens, denoiser.dim, device=device, generator=gen)
    ab = alpha_bar(steps, device)
    ts = torch.linspace(steps - 1, 0, sample_steps, device=device).long()
    for i, ti in enumerate(ts):
        t = torch.full((count,), int(ti.item()), device=device, dtype=torch.long)
        eps = denoiser(x, t)
        a = ab[ti]
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
        if i == len(ts) - 1:
            x = x0
        else:
            ap = ab[ts[i + 1]]
            x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps

    generated = x * std + mean
    real = z.to(device)
    real_count = min(count, real.shape[0])

    real_images = decoder.decode_local(real[:real_count])
    generated_images = decoder.decode_local(generated)
    save_grid(real_images, args.out / "real_latent_reconstructions.png",
              scale=max(1, args.grid_scale))
    save_grid(generated_images, args.out / "generated_from_saved_r5_checkpoint.png",
              scale=max(1, args.grid_scale))

    real_normalized = ((z - mean.cpu()) / std.cpu()).clamp(-8, 8)
    generated_normalized = x.detach().cpu()
    real_flat = real_normalized.reshape(-1)
    gen_flat = generated_normalized.reshape(-1)
    q = torch.quantile(real_flat, torch.tensor([0.01, 0.99]))
    outside = ((gen_flat < q[0]) | (gen_flat > q[1])).float().mean()

    report = {
        "experiment": "R5_real_vs_generated_latent_diagnostics",
        "device": str(device),
        "rocm_version": getattr(torch.version, "hip", None),
        "real_latent_source": str(args.latents),
        "r3_checkpoint": str(args.r3_checkpoint),
        "r5_checkpoint": str(args.r5_checkpoint),
        "real_latents": summary(z),
        "generated_latents": summary(generated),
        "real_normalized_latents": summary(real_normalized),
        "generated_normalized_latents": summary(generated_normalized),
        "real_normalized_q01_q99": [float(q[0]), float(q[1])],
        "generated_normalized_fraction_outside_real_q01_q99": float(outside),
        "sample_count": count,
        "sample_steps": sample_steps,
        "diffusion_steps": steps,
        "note": "Real statistics use the complete saved latent array; generated statistics use the requested sample count."
    }
    (args.out / "latent_distribution_diagnostics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        args.out / "latent_distribution_diagnostics.npz",
        real_raw=z.numpy(),
        generated_raw=generated.detach().cpu().numpy(),
        real_normalized=real_normalized.numpy(),
        generated_normalized=generated_normalized.numpy(),
    )
    print(json.dumps({
        "real_raw_overall": report["real_latents"]["overall"],
        "generated_raw_overall": report["generated_latents"]["overall"],
        "real_normalized_overall": report["real_normalized_latents"]["overall"],
        "generated_normalized_overall": report["generated_normalized_latents"]["overall"],
        "generated_fraction_outside_real_q01_q99": float(outside),
        "outputs": str(args.out),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
