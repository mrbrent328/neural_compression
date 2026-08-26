#!/usr/bin/env python3
"""
Known-Factor Synthetic Latent Experiment v3
EXACT-PREPROCESSING + ISOLATED INTERVENTIONS

This is the corrected follow-up to v2.

Fixes:
1. Reproduces v1's exact synthetic dataset generation and normalization
   (seed=42), so counterfactual inputs live in the same standardized
   distribution the autoencoder was trained on.
2. Separates DIRECT A/B interventions from TOTAL A/B interventions.
   The generator is X = a*A + b*B + c*C + d*a*b*D.
3. Uses multiple intervention sizes to estimate local sensitivity.
4. Measures latent displacement and reconstruction response.
5. Reports the interaction-mediated component:
       total_A - direct_A
       total_B - direct_B

The key question:
Does the interesting latent geometry survive when the experiment is
made causally clean and distribution-matched?
"""

import json
import math
import random
import time
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

DELTAS = [0.01, 0.05, 0.10, 0.25, 0.50]
N_PROBES = 500

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "synthetic_latent_v3"


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

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


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
    """
    Reproduce v1's generator exactly, including the random factor/noise
    sequence. This gives us the exact affine normalization used by v1.
    """
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

    return standardized, factors, patterns, mean, std, raw, noise


def standardize_with_v1_stats(raw_tensor, mean, std):
    return ((raw_tensor - mean) / std).flatten(1)


def generate_from_factors(a, b, c, d, A, B, C, D):
    x = (
        a.view(-1, 1, 1) * A
        + b.view(-1, 1, 1) * B
        + c.view(-1, 1, 1) * C
        + d.view(-1, 1, 1)
        * a.view(-1, 1, 1)
        * b.view(-1, 1, 1)
        * D
    )
    return x


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def cosine_abs(v, w):
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < 1e-12 or nw < 1e-12:
        return 0.0
    return abs(float(np.dot(v, w) / (nv * nw)))


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KNOWN-FACTOR SYNTHETIC LATENT EXPERIMENT v3")
    print("EXACT PREPROCESSING + ISOLATED INTERVENTIONS")
    print("=" * 80)

    model, device, checkpoint = load_model()

    (
        regenerated,
        regen_factors,
        patterns,
        norm_mean,
        norm_std,
        raw,
        noise,
    ) = generate_exact_v1_dataset()

    # Compare regenerated data against the saved v1 dataset.
    dataset_path = INPUT_DIR / "dataset.pt"
    saved = torch.load(
        dataset_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_tensors = saved["tensors"].float()
    saved_factors = saved["factors"].float()

    tensor_error = float((regenerated - saved_tensors).abs().max())
    factor_error = float((regen_factors - saved_factors).abs().max())

    print(f"Device: {device}")
    print(f"Best v1 epoch: {checkpoint.get('best_epoch', '?')}")
    print(f"v1 best test MSE: {checkpoint.get('best_test_mse', '?')}")
    print(f"Exact regeneration max tensor error: {tensor_error:.8g}")
    print(f"Exact regeneration max factor error: {factor_error:.8g}")
    print(f"v1 normalization mean: {norm_mean.item():.10f}")
    print(f"v1 normalization std:  {norm_std.item():.10f}")
    print()

    if tensor_error > 1e-5 or factor_error > 1e-6:
        print("WARNING: regenerated data does not exactly match saved v1 data.")
        print("The intervention results will still run, but the exactness check failed.")

    split = int(N_SAMPLES * TRAIN_FRACTION)

    # Use held-out samples, exactly matching v1's test split.
    test_factors = regen_factors[split:]

    n = min(N_PROBES, len(test_factors))
    indices = np.linspace(0, len(test_factors) - 1, n, dtype=int)

    base_factors = test_factors[indices].clone()

    a = base_factors[:, 0]
    b = base_factors[:, 1]
    c = base_factors[:, 2]
    d = base_factors[:, 3]

    A, B, C, D = patterns

    # Baseline is regenerated cleanly, then standardized with the exact
    # v1 normalization constants. This intentionally removes noise so
    # the causal intervention itself is isolated.
    baseline_raw = generate_from_factors(a, b, c, d, A, B, C, D)
    baseline = standardize_with_v1_stats(
        baseline_raw, norm_mean, norm_std
    )

    with torch.no_grad():
        z_base = model.encode(baseline.to(device))
        recon_base = model.decode(z_base)

    z_base_np = z_base.cpu().numpy()
    recon_base_np = recon_base.cpu().numpy()

    all_results = {}
    vector_fields = {}

    for delta in DELTAS:
        print(f"\n--- delta = {delta:.2f} ---")

        # DIRECT A: change A's coefficient, freeze interaction term.
        # This is NOT generated by changing 'a' in the full generator.
        # We explicitly construct:
        #   (a+delta)A + bB + cC + d*a*bD
        direct_A_raw = (
            (a + delta).view(-1, 1, 1) * A
            + b.view(-1, 1, 1) * B
            + c.view(-1, 1, 1) * C
            + d.view(-1, 1, 1)
            * a.view(-1, 1, 1)
            * b.view(-1, 1, 1)
            * D
        )

        # TOTAL A: change a everywhere it occurs.
        total_A_raw = generate_from_factors(
            a + delta, b, c, d, A, B, C, D
        )

        # DIRECT B: freeze interaction term.
        direct_B_raw = (
            a.view(-1, 1, 1) * A
            + (b + delta).view(-1, 1, 1) * B
            + c.view(-1, 1, 1) * C
            + d.view(-1, 1, 1)
            * a.view(-1, 1, 1)
            * b.view(-1, 1, 1)
            * D
        )

        # TOTAL B: change b everywhere it occurs.
        total_B_raw = generate_from_factors(
            a, b + delta, c, d, A, B, C, D
        )

        # C has no interaction coupling.
        C_changed_raw = generate_from_factors(
            a, b, c + delta, d, A, B, C, D
        )

        # Pure interaction-strength intervention.
        D_changed_raw = generate_from_factors(
            a, b, c, d + delta, A, B, C, D
        )

        candidates = {
            "direct_A": standardize_with_v1_stats(
                direct_A_raw, norm_mean, norm_std
            ),
            "total_A": standardize_with_v1_stats(
                total_A_raw, norm_mean, norm_std
            ),
            "direct_B": standardize_with_v1_stats(
                direct_B_raw, norm_mean, norm_std
            ),
            "total_B": standardize_with_v1_stats(
                total_B_raw, norm_mean, norm_std
            ),
            "C": standardize_with_v1_stats(
                C_changed_raw, norm_mean, norm_std
            ),
            "interaction": standardize_with_v1_stats(
                D_changed_raw, norm_mean, norm_std
            ),
        }

        encoded = {}
        reconstructed = {}

        with torch.no_grad():
            for name, x in candidates.items():
                z = model.encode(x.to(device))
                r = model.decode(z)
                encoded[name] = z.cpu().numpy()
                reconstructed[name] = r.cpu().numpy()

        disp = {
            name: encoded[name] - z_base_np
            for name in encoded
        }

        # Interaction-mediated component of changing A/B.
        disp["A_interaction_mediated"] = (
            disp["total_A"] - disp["direct_A"]
        )
        disp["B_interaction_mediated"] = (
            disp["total_B"] - disp["direct_B"]
        )

        # Normalize by delta to approximate a local derivative.
        derivative = {
            name: vectors / delta
            for name, vectors in disp.items()
        }

        delta_result = {}

        for name, vectors in disp.items():
            norms = np.linalg.norm(vectors, axis=1)

            mean_vec = vectors.mean(axis=0)

            # For derivative-like quantities, report the median as well;
            # it is less sensitive to the occasional large excursions.
            delta_result[name] = {
                "mean_vector": mean_vec.tolist(),
                "mean_magnitude": float(norms.mean()),
                "median_magnitude": float(np.median(norms)),
                "max_magnitude": float(norms.max()),
                "mean_abs_cosine_to_mean": float(
                    np.mean([
                        cosine_abs(v, mean_vec)
                        for v in vectors
                    ])
                ),
                "derivative_mean_vector": (
                    derivative[name].mean(axis=0).tolist()
                ),
                "derivative_median_magnitude": float(
                    np.median(np.linalg.norm(
                        derivative[name], axis=1
                    ))
                ),
            }

        # Reconstruction response against the actual counterfactual input.
        recon_metrics = {}

        for name, x in candidates.items():
            x_np = x.numpy()
            err = np.mean(
                (reconstructed[name] - x_np) ** 2,
                axis=1,
            )
            recon_shift = np.linalg.norm(
                reconstructed[name] - recon_base_np,
                axis=1,
            )

            recon_metrics[name] = {
                "mean_mse_to_counterfactual": float(err.mean()),
                "median_mse_to_counterfactual": float(
                    np.median(err)
                ),
                "mean_reconstruction_shift": float(
                    recon_shift.mean()
                ),
            }

        all_results[f"{delta:.2f}"] = {
            "displacement": delta_result,
            "reconstruction": recon_metrics,
        }

        vector_fields[f"{delta:.2f}"] = {
            name: vectors.tolist()
            for name, vectors in disp.items()
        }

    # ------------------------------------------------------------------
    # Plot 1: sensitivity vs delta
    # ------------------------------------------------------------------
    names = [
        "direct_A",
        "total_A",
        "direct_B",
        "total_B",
        "C",
        "interaction",
    ]

    plt.figure(figsize=(10, 6))

    for name in names:
        medians = [
            all_results[f"{d:.2f}"]["displacement"][name][
                "median_magnitude"
            ]
            for d in DELTAS
        ]
        plt.plot(DELTAS, medians, marker="o", label=name)

    plt.xlabel("Intervention delta")
    plt.ylabel("Median latent displacement")
    plt.title("Latent response as intervention size changes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sensitivity_vs_delta.png",
        dpi=150,
    )
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: derivative magnitude
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for name in names:
        derivative_medians = [
            all_results[f"{d:.2f}"]["displacement"][name][
                "derivative_median_magnitude"
            ]
            for d in DELTAS
        ]
        plt.plot(
            DELTAS,
            derivative_medians,
            marker="o",
            label=name,
        )

    plt.xlabel("Intervention delta")
    plt.ylabel("Median |Δz / Δfactor|")
    plt.title("Local latent sensitivity across intervention scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "derivative_vs_delta.png",
        dpi=150,
    )
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: direct vs total interaction decomposition at delta=.50
    # ------------------------------------------------------------------
    chosen = all_results["0.50"]["displacement"]

    labels = [
        "direct A",
        "A interaction-mediated",
        "total A",
        "direct B",
        "B interaction-mediated",
        "total B",
        "C",
        "interaction",
    ]

    keys = [
        "direct_A",
        "A_interaction_mediated",
        "total_A",
        "direct_B",
        "B_interaction_mediated",
        "total_B",
        "C",
        "interaction",
    ]

    values = [
        chosen[k]["median_magnitude"]
        for k in keys
    ]

    plt.figure(figsize=(11, 6))
    plt.bar(labels, values)
    plt.ylabel("Median latent displacement")
    plt.title("Clean intervention decomposition at delta = 0.50")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "intervention_decomposition.png",
        dpi=150,
    )
    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: vector field for the smallest delta
    # ------------------------------------------------------------------
    smallest = vector_fields[f"{DELTAS[0]:.2f}"]

    plt.figure(figsize=(9, 8))

    plot_names = [
        ("direct_A", "direct A"),
        ("direct_B", "direct B"),
        ("C", "C"),
        ("interaction", "interaction"),
    ]

    for key, label in plot_names:
        vectors = np.asarray(smallest[key])
        # Plot a subset for readability.
        step = max(1, len(vectors) // 150)
        v = vectors[::step]
        plt.scatter(
            v[:, 0],
            v[:, 1],
            s=14,
            alpha=0.35,
            label=label,
        )

    plt.axhline(0, linewidth=0.7)
    plt.axvline(0, linewidth=0.7)
    plt.xlabel("Δz1")
    plt.ylabel("Δz2")
    plt.title(
        f"Local latent response vectors (delta={DELTAS[0]:.2f})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "local_response_vectors.png",
        dpi=150,
    )
    plt.close()

    results = {
        "experiment": "known_factor_synthetic_latent_v3_controlled",
        "source_experiment": "known_factor_synthetic_latent_v1",
        "seed": SEED,
        "n_probes": n,
        "deltas": DELTAS,
        "noise_std": NOISE_STD,
        "normalization": {
            "mean": float(norm_mean),
            "std": float(norm_std),
        },
        "exact_regeneration": {
            "max_tensor_error": tensor_error,
            "max_factor_error": factor_error,
        },
        "best_epoch": checkpoint.get("best_epoch"),
        "best_test_mse": checkpoint.get("best_test_mse"),
        "results_by_delta": all_results,
        "interpretation": {
            "direct_A": "changes A coefficient while freezing the A*B interaction contribution",
            "total_A": "changes A everywhere, including its effect on the interaction term",
            "direct_B": "changes B coefficient while freezing the A*B interaction contribution",
            "total_B": "changes B everywhere, including its effect on the interaction term",
            "A_interaction_mediated": "total_A latent displacement minus direct_A displacement",
            "B_interaction_mediated": "total_B latent displacement minus direct_B displacement",
        },
    }

    with open(OUTPUT_DIR / "v3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results: {OUTPUT_DIR.resolve()}")
    print()
    print("At delta=0.50:")
    for key in keys:
        value = chosen[key]["median_magnitude"]
        print(f"  {key:28s} median |Δz| = {value:.6f}")


if __name__ == "__main__":
    main()
