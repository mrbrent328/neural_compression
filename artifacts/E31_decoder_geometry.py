#!/usr/bin/env python3
"""
E31 — Local Decoder Geometry

Measures how latent-space movement expands/contracts after decoding.

For each latent point z:
    latent direction perturbation
            |
            v
        decoder(z + dz)
            |
            v
     output-space movement

This estimates a local decoder metric:

    G = J^T J

where J is the decoder Jacobian.

For a 2D latent space this gives:
- local scaling
- principal output-sensitive directions
- anisotropy
- determinant / volume change

Uses the real V1 autoencoder.
"""

from pathlib import Path
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from known_factor_synthetic_latent_v1 import TinyAutoencoder, ROWS, COLS
from known_factor_synthetic_latent_v5_vector_field import (
    generate_exact_v1_dataset,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.pt"
OUT = ROOT / "synthetic_latent_e31_decoder_geometry"

DELTA = 0.01


def load_model():
    ckpt = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = TinyAutoencoder(ROWS * COLS, 2)
    model.load_state_dict(
        ckpt["model_state_dict"]
    )
    model.eval()

    return model


def standardize(x, mean, std):
    return ((x - mean) / std).flatten(1)


@torch.no_grad()
def encode(model, x):
    return model.encode(x)


@torch.no_grad()
def decode(model, z):
    return model.decode(z)


def decoder_geometry(model, z):
    """
    Finite difference decoder Jacobian.

    latent dimension = 2
    output dimension = 4096
    """

    z = z.clone().detach()

    cols = []

    for axis in range(2):
        zp = z.clone()
        zm = z.clone()

        zp[:, axis] += DELTA
        zm[:, axis] -= DELTA

        yp = decode(model, zp)
        ym = decode(model, zm)

        dz = (
            yp - ym
        ) / (2 * DELTA)

        cols.append(
            dz.flatten().cpu().numpy()
        )

    J = np.stack(cols, axis=1)

    G = J.T @ J

    eigvals = np.linalg.eigvalsh(G)

    return {
        "J_norm": float(np.linalg.norm(J)),
        "metric_trace": float(np.trace(G)),
        "metric_det": float(np.linalg.det(G)),
        "eig_low": float(eigvals[0]),
        "eig_high": float(eigvals[1]),
        "anisotropy": float(
            eigvals[1] /
            max(eigvals[0], 1e-12)
        ),
    }


def main():
    OUT.mkdir(exist_ok=True)

    model = load_model()

    _, factors, patterns, mean, std = generate_exact_v1_dataset()

    # same held-out region used in V5
    factors = factors[int(len(factors)*0.8):]

    # reconstruct inputs
    from known_factor_synthetic_latent_v5_vector_field import generate_from_factors

    A, B, C, D = patterns

    raw = generate_from_factors(
        factors[:,0],
        factors[:,1],
        factors[:,2],
        factors[:,3],
        A, B, C, D,
    )

    x = standardize(
        raw,
        mean,
        std,
    )

    with torch.no_grad():
        z = encode(model, x)

    rows = []

    for i in range(len(z)):
        result = {
            "probe": i,
            "z0": float(z[i,0]),
            "z1": float(z[i,1]),
        }

        geo = decoder_geometry(
            model,
            z[i:i+1],
        )

        result.update(geo)
        rows.append(result)

    with open(
        OUT / "e31_decoder_geometry.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        key: {
            "mean": float(
                np.mean(
                    [r[key] for r in rows]
                )
            ),
            "median": float(
                np.median(
                    [r[key] for r in rows]
                )
            ),
            "max": float(
                np.max(
                    [r[key] for r in rows]
                )
            ),
        }
        for key in [
            "J_norm",
            "metric_trace",
            "metric_det",
            "eig_low",
            "eig_high",
            "anisotropy",
        ]
    }

    with open(
        OUT / "e31_summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    anis = np.array(
        [r["anisotropy"] for r in rows]
    )

    plt.figure(figsize=(8,6))
    plt.hist(anis, bins=40)
    plt.xlabel("Decoder anisotropy")
    plt.ylabel("count")
    plt.title("E31 Local Decoder Geometry")
    plt.tight_layout()
    plt.savefig(
        OUT / "e31_anisotropy_hist.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8,6))
    plt.scatter(
        [r["z0"] for r in rows],
        [r["z1"] for r in rows],
        c=anis,
        s=18,
    )
    plt.xlabel("latent z0")
    plt.ylabel("latent z1")
    plt.title("Decoder anisotropy across latent space")
    plt.colorbar(label="anisotropy")
    plt.tight_layout()
    plt.savefig(
        OUT / "e31_latent_anisotropy_map.png",
        dpi=160,
    )
    plt.close()

    print("="*80)
    print("E31 COMPLETE")
    print("="*80)
    print(json.dumps(summary, indent=2))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
