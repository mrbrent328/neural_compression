#!/usr/bin/env python3
"""
Known-Factor Synthetic Latent Experiment v4
LOCAL SENSITIVITY / OUTLIER LOCALIZATION

Purpose
-------
V3 showed rare, very large latent displacements. V4 asks:

    Are those high-sensitivity points random,
    or do they occur at particular known A/B/C/interactions?

We keep the V3 setup exactly:
- same seed and synthetic generator
- same v1 normalization
- same held-out test split
- same trained v1 autoencoder
- clean counterfactual inputs (no added noise)
- latent dimension = 2

Instead of asking only for average displacement, we estimate the local
Jacobian magnitude at every probe using symmetric finite differences.

For each factor f:
    dz/df ~= [z(f + delta) - z(f - delta)] / (2 delta)

We then:
1. rank probes by local sensitivity
2. map sensitivity against A/B/C/interaction factors
3. make pairwise A/B/A/C/B/C maps
4. compare the top 10% against the remaining probes
5. save all probe-level data to JSON/CSV

This is intentionally an analysis experiment: NO TRAINING.
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

# Small symmetric steps. We use two so we can see whether the result
# itself is stable with respect to finite-difference scale.
DELTAS = [0.01, 0.05]
TOP_FRACTION = 0.10

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "synthetic_latent_v4"


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

    noise = torch.randn_like(raw) * NOISE_STD
    noisy = raw + noise

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
    model_path = INPUT_DIR / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Put this script beside v1's best_model.pt."
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


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def rank_corr(x, y):
    # Simple Spearman correlation without scipy.
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def save_scatter(x, y, c, xlabel, ylabel, title, path, log_color=True):
    plt.figure(figsize=(8, 6))
    if log_color:
        color_values = np.log10(np.asarray(c) + 1e-8)
        label = "log10 local sensitivity"
    else:
        color_values = c
        label = "local sensitivity"

    sc = plt.scatter(
        x,
        y,
        c=color_values,
        s=24,
        alpha=0.75,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(sc, label=label)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    print("=" * 80)
    print("KNOWN-FACTOR SYNTHETIC LATENT EXPERIMENT v4")
    print("LOCAL SENSITIVITY / OUTLIER LOCALIZATION")
    print("=" * 80)

    model, device, checkpoint = load_model()

    (
        regenerated,
        regen_factors,
        patterns,
        norm_mean,
        norm_std,
    ) = generate_exact_v1_dataset()

    dataset_path = INPUT_DIR / "dataset.pt"
    saved = torch.load(
        dataset_path,
        map_location="cpu",
        weights_only=False,
    )

    tensor_error = float(
        (regenerated - saved["tensors"].float()).abs().max()
    )
    factor_error = float(
        (regen_factors - saved["factors"].float()).abs().max()
    )

    print(f"Device: {device}")
    print(f"Best v1 epoch: {checkpoint.get('best_epoch', '?')}")
    print(f"Best v1 test MSE: {checkpoint.get('best_test_mse', '?')}")
    print(f"Exact regeneration tensor error: {tensor_error:.8g}")
    print(f"Exact regeneration factor error: {factor_error:.8g}")
    print(f"Normalization mean: {norm_mean.item():.10f}")
    print(f"Normalization std:  {norm_std.item():.10f}")
    print()

    split = int(N_SAMPLES * TRAIN_FRACTION)
    test_factors = regen_factors[split:]

    # Use every held-out probe. This is only 500 points and requires no training.
    base_factors = test_factors.clone()
    n = len(base_factors)

    print(f"Held-out probes: {n}")
    print(f"Finite-difference deltas: {DELTAS}")
    print()

    factor_names = ["A", "B", "C", "interaction"]
    sensitivity_by_delta = {}
    derivative_vectors_by_delta = {}

    for delta in DELTAS:
        print(f"--- delta = {delta:.2f} ---")

        derivs = {}
        magnitudes = {}

        for idx, name in enumerate(factor_names):
            plus = base_factors.clone()
            minus = base_factors.clone()
            plus[:, idx] += delta
            minus[:, idx] -= delta

            z_plus = encode_factors(
                model, device, plus, patterns, norm_mean, norm_std
            )
            z_minus = encode_factors(
                model, device, minus, patterns, norm_mean, norm_std
            )

            derivative = (z_plus - z_minus) / (2.0 * delta)
            magnitude = np.linalg.norm(derivative, axis=1)

            derivs[name] = derivative
            magnitudes[name] = magnitude

            print(
                f"{name:12s} median={np.median(magnitude):.6f} "
                f"mean={np.mean(magnitude):.6f} "
                f"p95={np.percentile(magnitude, 95):.6f} "
                f"max={np.max(magnitude):.6f}"
            )

        # Total Jacobian/Frobenius-like sensitivity across all four factors.
        total = np.sqrt(
            sum(magnitudes[name] ** 2 for name in factor_names)
        )

        sensitivity_by_delta[f"{delta:.2f}"] = {
            name: magnitudes[name] for name in factor_names
        }
        derivative_vectors_by_delta[f"{delta:.2f}"] = derivs
        sensitivity_by_delta[f"{delta:.2f}"]["total"] = total

        print(
            f"{'TOTAL':12s} median={np.median(total):.6f} "
            f"mean={np.mean(total):.6f} "
            f"p95={np.percentile(total, 95):.6f} "
            f"max={np.max(total):.6f}"
        )
        print()

    # ------------------------------------------------------------------
    # Stable sensitivity: compare small and larger finite differences.
    # We use delta=.01 as the primary local estimate.
    # ------------------------------------------------------------------
    primary = sensitivity_by_delta["0.01"]
    larger = sensitivity_by_delta["0.05"]

    total_small = primary["total"]
    total_large = larger["total"]

    # Ratio near 1 means the local derivative estimate is stable across scale.
    stability_ratio = total_large / np.maximum(total_small, 1e-8)

    # Rank probes by local total sensitivity.
    order = np.argsort(total_small)[::-1]
    top_n = max(1, int(math.ceil(n * TOP_FRACTION)))
    top_idx = order[:top_n]
    rest_idx = order[top_n:]

    print("=" * 80)
    print("TOP-SENSITIVITY LOCALIZATION")
    print("=" * 80)
    print(f"Top fraction: {TOP_FRACTION:.0%}")
    print(f"Top probes:   {top_n}")
    print()

    def summarize_group(indices, label):
        vals = base_factors.numpy()[indices]
        print(label)
        print(
            f"  A           mean={vals[:,0].mean():+.4f} "
            f"median={np.median(vals[:,0]):+.4f}"
        )
        print(
            f"  B           mean={vals[:,1].mean():+.4f} "
            f"median={np.median(vals[:,1]):+.4f}"
        )
        print(
            f"  C           mean={vals[:,2].mean():+.4f} "
            f"median={np.median(vals[:,2]):+.4f}"
        )
        print(
            f"  interaction mean={vals[:,3].mean():+.4f} "
            f"median={np.median(vals[:,3]):+.4f}"
        )
        print(
            f"  total sens  mean={total_small[indices].mean():.4f} "
            f"median={np.median(total_small[indices]):.4f}"
        )

    summarize_group(top_idx, "TOP 10%")
    summarize_group(rest_idx, "REMAINING 90%")
    print()

    # Correlation of local sensitivity with each known factor.
    correlations = {}
    for name, factor_idx in zip(factor_names, range(4)):
        f = base_factors[:, factor_idx].numpy()
        correlations[name] = {
            "pearson_total_sensitivity": pearson(f, total_small),
            "spearman_total_sensitivity": rank_corr(f, total_small),
            "pearson_log_total_sensitivity": pearson(
                f, np.log10(total_small + 1e-8)
            ),
        }

    print("Sensitivity correlation with known factors:")
    for name, vals in correlations.items():
        print(
            f"  {name:12s} "
            f"Pearson={vals['pearson_total_sensitivity']:+.4f} "
            f"Spearman={vals['spearman_total_sensitivity']:+.4f}"
        )
    print()

    # ------------------------------------------------------------------
    # Save probe-level CSV.
    # ------------------------------------------------------------------
    csv_path = OUTPUT_DIR / "probe_sensitivity.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "probe",
            "A",
            "B",
            "C",
            "interaction",
            "sens_A",
            "sens_B",
            "sens_C",
            "sens_interaction",
            "total_sensitivity",
            "stability_ratio",
            "top_10pct",
        ])

        factors_np = base_factors.numpy()

        for i in range(n):
            writer.writerow([
                i,
                *factors_np[i].tolist(),
                float(primary["A"][i]),
                float(primary["B"][i]),
                float(primary["C"][i]),
                float(primary["interaction"][i]),
                float(total_small[i]),
                float(stability_ratio[i]),
                bool(i in set(top_idx)),
            ])

    # ------------------------------------------------------------------
    # JSON summary.
    # ------------------------------------------------------------------
    top_probe_records = []
    for i in top_idx[:25]:
        top_probe_records.append({
            "probe": int(i),
            "A": float(base_factors[i, 0]),
            "B": float(base_factors[i, 1]),
            "C": float(base_factors[i, 2]),
            "interaction": float(base_factors[i, 3]),
            "sens_A": float(primary["A"][i]),
            "sens_B": float(primary["B"][i]),
            "sens_C": float(primary["C"][i]),
            "sens_interaction": float(primary["interaction"][i]),
            "total_sensitivity": float(total_small[i]),
            "stability_ratio_005_001": float(stability_ratio[i]),
        })

    summary = {
        "experiment": "known_factor_synthetic_latent_v4_local_sensitivity",
        "source_experiment": "known_factor_synthetic_latent_v3_controlled",
        "seed": SEED,
        "n_probes": n,
        "deltas": DELTAS,
        "best_epoch": checkpoint.get("best_epoch"),
        "best_test_mse": checkpoint.get("best_test_mse"),
        "exact_regeneration": {
            "max_tensor_error": tensor_error,
            "max_factor_error": factor_error,
        },
        "primary_delta": 0.01,
        "top_fraction": TOP_FRACTION,
        "group_summary": {
            "top_10pct_factor_mean": base_factors.numpy()[top_idx].mean(axis=0).tolist(),
            "top_10pct_factor_median": np.median(
                base_factors.numpy()[top_idx], axis=0
            ).tolist(),
            "remaining_90pct_factor_mean": base_factors.numpy()[rest_idx].mean(axis=0).tolist(),
            "remaining_90pct_factor_median": np.median(
                base_factors.numpy()[rest_idx], axis=0
            ).tolist(),
        },
        "sensitivity_correlations": correlations,
        "sensitivity_summary": {
            delta: {
                name: {
                    "median": float(np.median(vals)),
                    "mean": float(np.mean(vals)),
                    "p95": float(np.percentile(vals, 95)),
                    "max": float(np.max(vals)),
                }
                for name, vals in data.items()
            }
            for delta, data in sensitivity_by_delta.items()
        },
        "top_25_probes": top_probe_records,
    }

    with open(OUTPUT_DIR / "v4_sensitivity_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Plots: factor vs total local sensitivity.
    # ------------------------------------------------------------------
    factors_np = base_factors.numpy()

    save_scatter(
        factors_np[:, 0],
        factors_np[:, 1],
        total_small,
        "factor A",
        "factor B",
        "Local latent sensitivity across A/B space",
        OUTPUT_DIR / "sensitivity_AB.png",
    )

    save_scatter(
        factors_np[:, 0],
        factors_np[:, 2],
        total_small,
        "factor A",
        "factor C",
        "Local latent sensitivity across A/C space",
        OUTPUT_DIR / "sensitivity_AC.png",
    )

    save_scatter(
        factors_np[:, 1],
        factors_np[:, 2],
        total_small,
        "factor B",
        "factor C",
        "Local latent sensitivity across B/C space",
        OUTPUT_DIR / "sensitivity_BC.png",
    )

    # Interaction space.
    save_scatter(
        factors_np[:, 0],
        factors_np[:, 1],
        primary["interaction"],
        "factor A",
        "factor B",
        "Interaction-channel sensitivity across A/B space",
        OUTPUT_DIR / "interaction_sensitivity_AB.png",
    )

    # Histograms: normal population vs top sensitivity tail.
    plt.figure(figsize=(9, 6))
    plt.hist(
        total_small,
        bins=40,
        alpha=0.7,
        label="all probes",
    )
    plt.axvline(
        np.percentile(total_small, 90),
        linewidth=2,
        label="90th percentile",
    )
    plt.xlabel("total local latent sensitivity")
    plt.ylabel("count")
    plt.title("Distribution of local latent sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sensitivity_distribution.png",
        dpi=150,
    )
    plt.close()

    # Stability plot: small vs larger finite-difference estimate.
    plt.figure(figsize=(8, 7))
    plt.scatter(
        total_small,
        total_large,
        s=18,
        alpha=0.65,
    )
    lim = max(
        float(np.max(total_small)),
        float(np.max(total_large)),
    )
    plt.plot([0, lim], [0, lim], linewidth=1)
    plt.xlabel("sensitivity, delta=0.01")
    plt.ylabel("sensitivity, delta=0.05")
    plt.title("Finite-difference sensitivity stability")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sensitivity_stability.png",
        dpi=150,
    )
    plt.close()

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results: {OUTPUT_DIR.resolve()}")
    print(f"CSV:     {csv_path}")
    print()
    print("Top 10 highest-sensitivity probes:")
    for i in top_idx[:10]:
        vals = factors_np[i]
        print(
            f"  probe {i:3d} | "
            f"A={vals[0]:+.3f} B={vals[1]:+.3f} "
            f"C={vals[2]:+.3f} D={vals[3]:+.3f} | "
            f"sens={total_small[i]:.4f} | "
            f"stability={stability_ratio[i]:.3f}"
        )


if __name__ == "__main__":
    main()
