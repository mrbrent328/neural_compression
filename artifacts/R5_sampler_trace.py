#!/usr/bin/env python3
"""Trace R5 reverse diffusion without retraining.

Runs the saved denoiser twice from the same initial Gaussian noise:
  1) the original 50-step deterministic DDIM schedule
  2) all 500 diffusion timesteps (500-step deterministic DDIM)

Records normalized latent statistics at every reverse step, compares the
final states with the saved R3 latent distribution, and decodes contact sheets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from R3_local_interaction_cifar10 import LocalInteractionAutoencoder
from R5_latent_diffusion import LatentDenoiser, alpha_bar, save_grid


def tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().float()
    return {
        "mean": float(x.mean().cpu()),
        "std": float(x.std(unbiased=False).cpu()),
        "min": float(x.min().cpu()),
        "max": float(x.max().cpu()),
        "rms": float(x.square().mean().sqrt().cpu()),
        "finite_fraction": float(torch.isfinite(x).float().mean().cpu()),
    }


@torch.no_grad()
def trace_sampler(model, initial, diffusion_steps, sample_steps, device):
    model.eval()
    x = initial.clone()
    ab = alpha_bar(diffusion_steps, device)
    ts = torch.linspace(diffusion_steps - 1, 0, sample_steps, device=device).long()
    rows = []

    for i, ti in enumerate(ts):
        t = torch.full((x.shape[0],), int(ti.item()), device=device, dtype=torch.long)
        eps = model(x, t)
        a = ab[ti]
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()

        row = {
            "iteration": i,
            "timestep": int(ti.item()),
            "alpha_bar": float(a.cpu()),
            **{f"x_{k}": v for k, v in tensor_stats(x).items()},
            **{f"eps_{k}": v for k, v in tensor_stats(eps).items()},
            **{f"x0_{k}": v for k, v in tensor_stats(x0).items()},
        }
        rows.append(row)

        if i == len(ts) - 1:
            x = x0
        else:
            ap = ab[ts[i + 1]]
            x = ap.sqrt() * x0 + (1 - ap).sqrt() * eps

    return x, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents", type=Path, required=True,
                   help="Original R3 local-state .npy file, normally [10000,16,32]")
    p.add_argument("--r3-checkpoint", type=Path, required=True)
    p.add_argument("--r5-checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/real_cifar10_latent_diffusion_r3pos/sampler_trace"))
    p.add_argument("--sample-count", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261124)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"R5 sampler trace | device={device} | torch={torch.__version__} | ROCm={getattr(torch.version, 'hip', None)}", flush=True)

    z = torch.from_numpy(np.load(args.latents.expanduser(), allow_pickle=False)).float()
    if z.ndim != 3 or tuple(z.shape[1:]) != (16, 32):
        raise ValueError(f"Expected R3 latents [N,16,32], got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise ValueError("R3 latent file contains NaN or infinity")

    r3 = torch.load(args.r3_checkpoint.expanduser(), map_location="cpu", weights_only=False)
    r5 = torch.load(args.r5_checkpoint.expanduser(), map_location="cpu", weights_only=False)

    arch = r3.get("architecture", {})
    decoder = LocalInteractionAutoencoder(
        local_dim=int(arch.get("local_dim", 32)),
        interaction_layers=int(arch.get("interaction_layers", 1)),
    ).to(device)
    decoder.load_state_dict(r3["model_state_dict"])
    decoder.eval()

    tokens = int(r5.get("tokens", z.shape[1]))
    dim = int(r5.get("local_dim", z.shape[2]))
    denoiser = LatentDenoiser(tokens=tokens, dim=dim).to(device)
    denoiser.load_state_dict(r5["denoiser_state_dict"])
    denoiser.eval()

    mean = r5["latent_mean"].to(device)
    std = r5["latent_std"].to(device)
    diffusion_steps = int(r5["diffusion_steps"])
    count = min(args.sample_count, 256)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = torch.randn(count, tokens, dim, device=device, generator=generator)

    # Same starting noise makes the 50-step and 500-step paths comparable.
    results = {}
    for steps in (int(r5.get("sample_steps", 50)), diffusion_steps):
        final, rows = trace_sampler(denoiser, initial, diffusion_steps, steps, device)
        raw = final * std + mean
        tag = f"{steps}_step"
        results[tag] = {
            "sample_steps": steps,
            "normalized_final": tensor_stats(final),
            "raw_final": tensor_stats(raw),
            "trace": rows,
        }
        with (args.out / f"trace_{tag}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        images = decoder.decode_local(raw)
        save_grid(images, args.out / f"samples_{tag}.png", scale=8)
        print(f"{tag}: normalized std={results[tag]['normalized_final']['std']:.4f}, "
              f"raw std={results[tag]['raw_final']['std']:.4f}", flush=True)

    real_norm = ((z - mean.cpu()) / std.cpu())
    real_stats = tensor_stats(real_norm)
    report = {
        "experiment": "R5_reverse_sampler_step_count_comparison",
        "device": str(device),
        "rocm_version": getattr(torch.version, "hip", None),
        "r3_latent_file": str(args.latents),
        "r3_checkpoint": str(args.r3_checkpoint),
        "r5_checkpoint": str(args.r5_checkpoint),
        "real_normalized_latent_stats": real_stats,
        "same_initial_noise_seed": args.seed,
        "diffusion_steps": diffusion_steps,
        "comparisons": results,
        "interpretation_hint": "Inspect trace CSVs for the first timestep where x or predicted x0 statistics diverge sharply.",
    }
    (args.out / "sampler_trace_results.json").write_text(json.dumps(report, indent=2))
    print(f"Saved sampler traces and image grids to: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
