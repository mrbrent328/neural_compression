#!/usr/bin/env python3
"""
Known-Factor Synthetic Latent Experiment v5
LOCAL LATENT VECTOR FIELD

Question
--------
V4 showed that latent sensitivity magnitude is highly non-uniform across
known factor space. V5 asks the complementary question:

    When A/B/C changes, WHICH DIRECTION does the latent move?

For every held-out probe we estimate:

    d z / d A
    d z / d B
    d z / d C
    d z / d interaction

using symmetric finite differences.

We then measure:
- vector magnitude
- vector angle
- pairwise cosine similarity between factor response vectors
- how stable each factor's direction is across input space
- whether factor directions rotate with A/B/C
- whether high-sensitivity regions also have unusual directional geometry

No training occurs.
The model, generator, seed, normalization, and held-out split match v1/v3/v4.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


SEED = 42
N_SAMPLES = 2500
ROWS = 64
COLS = 64
LATENT_DIM = 2
TRAIN_FRACTION = 0.8
NOISE_STD = 0.03

DELTA = 0.01
TOP_FRACTION = 0.10

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "synthetic_latent_v5"


class TinyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_patterns(rows, cols, device="cpu"):
    y = torch.linspace(-1, 1, rows, device=device).view(-1, 1)
    x = torch.linspace(-1, 1, cols, device=device).view(1, -1)

    A = torch.exp(-((x + 0.35) ** 2 + (y - 0.25) ** 2) / 0.30)
    B = torch.sin(2.5 * x) * torch.cos(1.7 * y)
    C = torch.sin(7 * x + 1.2 * y) * torch.exp(-(x * x + y * y) / 0.8)
    D = A * B

    patterns = torch.stack([A, B, C, D])
    patterns = patterns / patterns.flatten(1).std(dim=1).view(-1, 1, 1)
    return patterns


def generate_exact_v1_dataset():
    seed_everything(SEED)
    patterns = make_patterns(ROWS, COLS, "cpu")

    factors = torch.empty(N_SAMPLES, 4)
    factors.uniform_(-1, 1)

    a = factors[:, 0].view(-1, 1, 1)
    b = factors[:, 1].view(-1, 1, 1)
    c = factors[:, 2].view(-1, 1, 1)
    d = factors[:, 3].view(-1, 1, 1)

    raw = (
        a * patterns[0]
        + b * patterns[1]
        + c * patterns[2]
        + d * a * b * patterns[3]
    )

    noisy = raw + torch.randn_like(raw) * NOISE_STD

    mean = noisy.mean()
    std = noisy.std()
    standardized = (noisy - mean) / std

    return standardized, factors, patterns, mean, std


def generate_from_factors(a, b, c, d, A, B, C, D):
    return (
        a.view(-1, 1, 1) * A
        + b.view(-1, 1, 1) * B
        + c.view(-1, 1, 1) * C
        + d.view(-1, 1, 1)
        * a.view(-1, 1, 1)
        * b.view(-1, 1, 1)
        * D
    )


def standardize(raw, mean, std):
    return ((raw - mean) / std).flatten(1)


def load_model():
    model_path = SCRIPT_DIR / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Put this script beside best_model.pt."
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    model = TinyAutoencoder(ROWS * COLS, LATENT_DIM)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, device, checkpoint


@torch.no_grad()
def encode_factors(model, device, factors, patterns, norm_mean, norm_std):
    A, B, C, D = patterns

    raw = generate_from_factors(
        factors[:, 0],
        factors[:, 1],
        factors[:, 2],
        factors[:, 3],
        A, B, C, D,
    )

    x = standardize(raw, norm_mean, norm_std)
    return model.encode(x.to(device)).cpu().numpy()


def cosine_rows(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros(len(a), dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return np.clip(out, -1.0, 1.0)


def angle_rows(a, b):
    return np.degrees(np.arccos(cosine_rows(a, b)))


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0


def rank_corr(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def save_heat_scatter(x, y, values, xlabel, ylabel, title, path):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        x,
        y,
        c=values,
        s=25,
        alpha=0.75,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(sc, label="value")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_vector_field(factors, vectors, title, path, max_points=500):
    # A/B projection is used for the spatial field. Vector is the actual
    # 2-D latent response, not an A/B-space derivative.
    x = factors[:, 0]
    y = factors[:, 1]

    mag = np.linalg.norm(vectors, axis=1)
    keep = np.argsort(mag)[-min(max_points, len(mag)):]

    plt.figure(figsize=(9, 8))
    plt.quiver(
        x[keep],
        y[keep],
        vectors[keep, 0],
        vectors[keep, 1],
        mag[keep],
        angles="xy",
        scale_units="xy",
        scale=None,
        alpha=0.75,
    )
    plt.xlabel("factor A")
    plt.ylabel("factor B")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def circular_mean_deviation(vectors, reference):
    """Angular deviation from a reference 2-D latent direction."""
    ref_norm = np.linalg.norm(reference)
    vec_norm = np.linalg.norm(vectors, axis=1)
    denom = vec_norm * ref_norm

    cos = np.zeros(len(vectors))
    valid = denom > 1e-12
    cos[valid] = (
        vectors[valid] @ reference
    ) / denom[valid]

    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    print("=" * 80)
    print("KNOWN-FACTOR SYNTHETIC LATENT EXPERIMENT v5")
    print("LOCAL LATENT VECTOR FIELD")
    print("=" * 80)

    model, device, checkpoint = load_model()

    regenerated, regen_factors, patterns, norm_mean, norm_std = (
        generate_exact_v1_dataset()
    )

    saved = torch.load(
        SCRIPT_DIR / "dataset.pt",
        map_location="cpu",
        weights_only=False,
    )

    tensor_error = float(
        (regenerated - saved["tensors"].float()).abs().max()
    )
    factor_error = float(
        (regen_factors - saved["factors"].float()).abs().max()
    )

    split = int(N_SAMPLES * TRAIN_FRACTION)
    factors = regen_factors[split:]
    n = len(factors)

    print(f"Device: {device}")
    print(f"Best v1 epoch: {checkpoint.get('best_epoch', '?')}")
    print(f"Best v1 test MSE: {checkpoint.get('best_test_mse', '?')}")
    print(f"Exact tensor error: {tensor_error:.8g}")
    print(f"Exact factor error: {factor_error:.8g}")
    print(f"Held-out probes: {n}")
    print(f"Finite-difference delta: {DELTA}")
    print()

    names = ["A", "B", "C", "interaction"]
    vectors = {}

    # ---------------------------------------------------------------
    # Estimate local Jacobian columns.
    # ---------------------------------------------------------------
    for idx, name in enumerate(names):
        plus = factors.clone()
        minus = factors.clone()

        plus[:, idx] += DELTA
        minus[:, idx] -= DELTA

        z_plus = encode_factors(
            model, device, plus, patterns, norm_mean, norm_std
        )
        z_minus = encode_factors(
            model, device, minus, patterns, norm_mean, norm_std
        )

        vectors[name] = (z_plus - z_minus) / (2.0 * DELTA)

    magnitudes = {
        name: np.linalg.norm(vectors[name], axis=1)
        for name in names
    }

    total_sensitivity = np.sqrt(
        sum(magnitudes[name] ** 2 for name in names)
    )

    print("Local derivative magnitudes:")
    for name in names:
        m = magnitudes[name]
        print(
            f"{name:12s} "
            f"median={np.median(m):.6f} "
            f"mean={np.mean(m):.6f} "
            f"p95={np.percentile(m, 95):.6f} "
            f"max={np.max(m):.6f}"
        )
    print()

    # ---------------------------------------------------------------
    # Pairwise geometry.
    # ---------------------------------------------------------------
    pair_names = [
        ("A", "B"),
        ("A", "C"),
        ("A", "interaction"),
        ("B", "C"),
        ("B", "interaction"),
        ("C", "interaction"),
    ]

    pair_cos = {}
    pair_angles = {}

    print("Pairwise response directions:")
    for a, b in pair_names:
        c = cosine_rows(vectors[a], vectors[b])
        ang = np.degrees(np.arccos(c))

        key = f"{a}_vs_{b}"
        pair_cos[key] = c
        pair_angles[key] = ang

        print(
            f"{key:22s} "
            f"cos median={np.median(c):+.5f} "
            f"mean={np.mean(c):+.5f} "
            f"angle median={np.median(ang):.2f}° "
            f"angle p95={np.percentile(ang,95):.2f}°"
        )
    print()

    # ---------------------------------------------------------------
    # Directional consistency.
    #
    # Reference direction = normalized mean response vector.
    # This measures whether a factor behaves like a global latent axis.
    # ---------------------------------------------------------------
    direction_deviation = {}
    reference_vectors = {}

    print("Directional consistency:")
    for name in names:
        v = vectors[name]
        ref = v.mean(axis=0)
        ref_norm = np.linalg.norm(ref)

        if ref_norm < 1e-12:
            ref = np.array([1.0, 0.0])
        else:
            ref = ref / ref_norm

        reference_vectors[name] = ref
        deviation = circular_mean_deviation(v, ref)
        direction_deviation[name] = deviation

        print(
            f"{name:12s} "
            f"median deviation={np.median(deviation):.2f}° "
            f"mean={np.mean(deviation):.2f}° "
            f"p95={np.percentile(deviation,95):.2f}° "
            f"max={np.max(deviation):.2f}°"
        )
    print()

    # ---------------------------------------------------------------
    # Does direction depend on factor coordinates?
    # ---------------------------------------------------------------
    factor_np = factors.numpy()

    direction_correlations = {}

    for name in names:
        # Angle relative to the global mean direction.
        deviation = direction_deviation[name]

        direction_correlations[name] = {}

        for j, factor_name in enumerate(names):
            f = factor_np[:, j]
            direction_correlations[name][factor_name] = {
                "pearson": pearson(f, deviation),
                "spearman": rank_corr(f, deviation),
            }

    print("Direction-vs-factor correlations (Spearman):")
    for name in names:
        vals = direction_correlations[name]
        pretty = " ".join(
            f"{k}={v['spearman']:+.3f}" for k, v in vals.items()
        )
        print(f"  {name:12s}: {pretty}")
    print()

    # ---------------------------------------------------------------
    # High-sensitivity subset.
    # ---------------------------------------------------------------
    top_n = max(1, int(math.ceil(n * TOP_FRACTION)))
    order = np.argsort(total_sensitivity)[::-1]
    top = order[:top_n]
    rest = order[top_n:]

    print("=" * 80)
    print("HIGH-SENSITIVITY DIRECTIONAL TEST")
    print("=" * 80)
    print(f"Top {TOP_FRACTION:.0%}: {top_n} probes")

    for name in names:
        dev = direction_deviation[name]

        print(
            f"{name:12s} "
            f"top median deviation={np.median(dev[top]):.2f}° | "
            f"rest median={np.median(dev[rest]):.2f}°"
        )

    print("\nPairwise angle: top vs rest")
    for a, b in pair_names:
        key = f"{a}_vs_{b}"
        ang = pair_angles[key]
        print(
            f"{key:22s} "
            f"top median={np.median(ang[top]):.2f}° | "
            f"rest median={np.median(ang[rest]):.2f}°"
        )
    print()

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------
    csv_path = OUTPUT_DIR / "probe_vector_geometry.csv"

    top_set = set(top.tolist())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "probe",
            "A",
            "B",
            "C",
            "interaction",
            "total_sensitivity",
        ]

        for name in names:
            header += [
                f"d{name}_z0",
                f"d{name}_z1",
                f"mag_{name}",
                f"angle_from_global_{name}",
            ]

        for a, b in pair_names:
            key = f"{a}_vs_{b}"
            header += [
                f"cos_{key}",
                f"angle_{key}",
            ]

        header.append("top_10pct")
        writer.writerow(header)

        for i in range(n):
            row = [
                i,
                *factor_np[i].tolist(),
                float(total_sensitivity[i]),
            ]

            for name in names:
                row += [
                    float(vectors[name][i, 0]),
                    float(vectors[name][i, 1]),
                    float(magnitudes[name][i]),
                    float(direction_deviation[name][i]),
                ]

            for a, b in pair_names:
                key = f"{a}_vs_{b}"
                row += [
                    float(pair_cos[key][i]),
                    float(pair_angles[key][i]),
                ]

            row.append(i in top_set)
            writer.writerow(row)

    # ---------------------------------------------------------------
    # JSON summary
    # ---------------------------------------------------------------
    summary = {
        "experiment": "known_factor_synthetic_latent_v5_vector_field",
        "seed": SEED,
        "n_probes": n,
        "delta": DELTA,
        "exact_regeneration": {
            "max_tensor_error": tensor_error,
            "max_factor_error": factor_error,
        },
        "best_epoch": checkpoint.get("best_epoch"),
        "best_test_mse": checkpoint.get("best_test_mse"),
        "reference_vectors": {
            k: v.tolist() for k, v in reference_vectors.items()
        },
        "magnitude_summary": {
            name: {
                "median": float(np.median(magnitudes[name])),
                "mean": float(np.mean(magnitudes[name])),
                "p95": float(np.percentile(magnitudes[name], 95)),
                "max": float(np.max(magnitudes[name])),
            }
            for name in names
        },
        "direction_deviation_summary": {
            name: {
                "median": float(np.median(direction_deviation[name])),
                "mean": float(np.mean(direction_deviation[name])),
                "p95": float(np.percentile(direction_deviation[name], 95)),
                "max": float(np.max(direction_deviation[name])),
            }
            for name in names
        },
        "pairwise_angle_summary": {
            key: {
                "cos_median": float(np.median(pair_cos[key])),
                "cos_mean": float(np.mean(pair_cos[key])),
                "angle_median": float(np.median(pair_angles[key])),
                "angle_p95": float(np.percentile(pair_angles[key], 95)),
            }
            for key in pair_angles
        },
        "direction_factor_correlations": direction_correlations,
        "high_sensitivity": {
            "top_fraction": TOP_FRACTION,
            "top_count": top_n,
            "direction_deviation_top_medians": {
                name: float(np.median(direction_deviation[name][top]))
                for name in names
            },
            "direction_deviation_rest_medians": {
                name: float(np.median(direction_deviation[name][rest]))
                for name in names
            },
            "pairwise_angle_top_medians": {
                key: float(np.median(pair_angles[key][top]))
                for key in pair_angles
            },
            "pairwise_angle_rest_medians": {
                key: float(np.median(pair_angles[key][rest]))
                for key in pair_angles
            },
        },
    }

    with open(OUTPUT_DIR / "v5_vector_geometry_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------

    # Magnitude maps
    for name in names:
        save_heat_scatter(
            factor_np[:, 0],
            factor_np[:, 1],
            magnitudes[name],
            "factor A",
            "factor B",
            f"{name} response magnitude across A/B space",
            OUTPUT_DIR / f"magnitude_{name}_AB.png",
        )

    # Direction angle maps
    for name in names:
        save_heat_scatter(
            factor_np[:, 0],
            factor_np[:, 1],
            direction_deviation[name],
            "factor A",
            "factor B",
            f"{name} direction deviation across A/B space",
            OUTPUT_DIR / f"direction_deviation_{name}_AB.png",
        )

    # Pairwise angle maps
    for a, b in pair_names[:3]:
        key = f"{a}_vs_{b}"
        save_heat_scatter(
            factor_np[:, 0],
            factor_np[:, 1],
            pair_angles[key],
            "factor A",
            "factor B",
            f"Angle between {a} and {b} responses",
            OUTPUT_DIR / f"angle_{key}_AB.png",
        )

    # Actual latent response vector fields over A/B space
    for name in ["A", "B", "C"]:
        save_vector_field(
            factor_np,
            vectors[name],
            f"Latent response vector field for {name}",
            OUTPUT_DIR / f"vector_field_{name}.png",
        )

    # Distribution of directional deviations
    plt.figure(figsize=(9, 6))
    for name in ["A", "B", "C"]:
        plt.hist(
            direction_deviation[name],
            bins=35,
            alpha=0.45,
            label=name,
        )
    plt.xlabel("angle from factor's global mean latent direction (degrees)")
    plt.ylabel("count")
    plt.title("Do factor directions stay globally consistent?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "direction_deviation_distributions.png",
        dpi=150,
    )
    plt.close()

    # Pairwise angle distributions
    plt.figure(figsize=(10, 6))
    for a, b in pair_names[:3]:
        key = f"{a}_vs_{b}"
        plt.hist(
            pair_angles[key],
            bins=35,
            alpha=0.4,
            label=key,
        )
    plt.xlabel("angle between response vectors (degrees)")
    plt.ylabel("count")
    plt.title("Relative geometry between factor responses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "pairwise_angle_distributions.png",
        dpi=150,
    )
    plt.close()

    # ---------------------------------------------------------------
    # Top probes
    # ---------------------------------------------------------------
    print("=" * 80)
    print("TOP 15 SENSITIVITY PROBES")
    print("=" * 80)

    for i in top[:15]:
        vals = factor_np[i]

        print(
            f"probe {i:3d} | "
            f"A={vals[0]:+.3f} "
            f"B={vals[1]:+.3f} "
            f"C={vals[2]:+.3f} "
            f"D={vals[3]:+.3f} | "
            f"total={total_sensitivity[i]:.3f}"
        )

        for name in ["A", "B", "C"]:
            v = vectors[name][i]
            print(
                f"    {name}: "
                f"dz=({v[0]:+.3f},{v[1]:+.3f}) "
                f"mag={magnitudes[name][i]:.3f} "
                f"angle_from_global={direction_deviation[name][i]:.1f}°"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results: {OUTPUT_DIR.resolve()}")
    print(f"CSV:     {csv_path}")


if __name__ == "__main__":
    main()
