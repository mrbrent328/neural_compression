#!/usr/bin/env python3
"""
V6 — Behavioral Coupling Test

Tests whether local latent geometry predicts model behavior.

Pipeline:

known factor perturbation
        |
        v
V5 exact generator
        |
        v
standardized 64x64 input
        |
        v
trained V1 autoencoder
        |
        +--> latent movement
        |
        +--> reconstruction movement

No latent-space assumptions are made.
"""

from pathlib import Path
import json
import csv
import numpy as np
import torch

# This script is intended to live beside the V1/V5 scripts.
from known_factor_synthetic_latent_v5_vector_field import (
    ROWS,
    COLS,
    DELTA,
    make_patterns,
    generate_from_factors,
    generate_exact_v1_dataset,
    TinyAutoencoder,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best_model.pt"
OUT = SCRIPT_DIR / "synthetic_latent_v6_behavior"


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    model = TinyAutoencoder(ROWS * COLS, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def standardize(x, mean, std):
    return ((x - mean) / std).flatten(1)


@torch.no_grad()
def run_model(model, x):
    z = model.encode(x)
    y = model.decode(z)
    return z.cpu().numpy(), y.cpu().numpy()


def make_input(factors, patterns, mean, std):
    A, B, C, D = patterns

    raw = generate_from_factors(
        factors[:, 0],
        factors[:, 1],
        factors[:, 2],
        factors[:, 3],
        A, B, C, D
    )

    return standardize(raw, mean, std)


def main():
    OUT.mkdir(exist_ok=True)

    model = load_model()

    _, factors, patterns, mean, std = generate_exact_v1_dataset()

    # held-out split matches V5
    factors = factors[int(len(factors) * 0.8):]

    base_x = make_input(factors, patterns, mean, std)
    base_z, base_y = run_model(model, base_x)

    results = []

    names = ["A", "B", "C", "interaction"]

    for idx, name in enumerate(names):
        plus = factors.clone()
        minus = factors.clone()

        plus[:, idx] += DELTA
        minus[:, idx] -= DELTA

        zp, yp = run_model(
            model,
            make_input(plus, patterns, mean, std)
        )

        zm, ym = run_model(
            model,
            make_input(minus, patterns, mean, std)
        )

        dz = (zp - zm) / (2 * DELTA)
        dy = (yp - ym) / (2 * DELTA)

        latent_mag = np.linalg.norm(dz, axis=1)
        output_mag = np.linalg.norm(dy, axis=1)

        for i in range(len(factors)):
            results.append({
                "probe": i,
                "factor": name,
                "latent_magnitude": float(latent_mag[i]),
                "output_magnitude": float(output_mag[i]),
                "output_per_latent": float(
                    output_mag[i] / max(latent_mag[i], 1e-12)
                ),
            })

    with open(OUT / "v6_behavior_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {}

    for name in names:
        rows = [
            r for r in results
            if r["factor"] == name
        ]

        latent = np.array(
            [r["latent_magnitude"] for r in rows]
        )
        output = np.array(
            [r["output_magnitude"] for r in rows]
        )

        summary[name] = {
            "latent_mean": float(latent.mean()),
            "output_mean": float(output.mean()),
            "correlation": float(
                np.corrcoef(latent, output)[0, 1]
            ),
        }

    with open(OUT / "v6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("V6 COMPLETE")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
