#!/usr/bin/env python3
"""
Known-Factor Synthetic Latent Experiment v2
CONTROLLED INTERVENTION GEOMETRY

Loads the v1 dataset/model, then directly probes the learned latent geometry.

For each test sample we create controlled counterfactuals:

    baseline
    A changed only
    B changed only
    C changed only
    interaction changed only

We measure the latent displacement caused by each intervention.

The important questions:

1. Does changing A move the latent in a consistent direction?
2. Same for B and C?
3. Does changing interaction strength produce a consistent direction?
4. Does the interaction displacement depend on the current A/B state?
5. How much reconstruction changes under each intervention?

Outputs:
    synthetic_latent_v2/
        intervention_results.json
        intervention_displacements.png
        intervention_angles.png
        interaction_dependence.png
        sample_trajectories.png
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROWS = 64
COLS = 64

class TinyAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "synthetic_latent_v2"

N_PROBES = 500
INTERVENTION_DELTA = 0.5


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x*x) * np.sum(y*y))
    if denom < 1e-12:
        return 0.0

    return float(np.sum(x*y) / denom)


def angle_between(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    cosine = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("KNOWN-FACTOR LATENT INTERVENTION GEOMETRY v2")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    dataset_path = INPUT_DIR / "dataset.pt"
    model_path = INPUT_DIR / "best_model.pt"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing {dataset_path}. Run v1 first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run v1 first."
        )

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    tensors = data["tensors"].float()
    factors = data["factors"].float()

    model = TinyAutoencoder(
        ROWS * COLS,
        latent_dim=2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Use held-out portion, exactly like v1.
    split = int(len(tensors) * 0.8)

    test_tensors = tensors[split:]
    test_factors = factors[split:]

    n = min(N_PROBES, len(test_tensors))

    # Select evenly distributed samples rather than only first N.
    indices = np.linspace(
        0,
        len(test_tensors) - 1,
        n,
        dtype=int,
    )

    base = test_tensors[indices].clone()
    base_factors = test_factors[indices].clone()

    # ------------------------------------------------------------------
    # Important:
    #
    # We reconstruct counterfactual tensors from the ORIGINAL known
    # generator patterns, rather than modifying arbitrary tensor pixels.
    #
    # This means an intervention corresponds to an actual change in
    # the hidden generative variable.
    # ------------------------------------------------------------------

    patterns = data["patterns"].float()

    A = patterns[0]
    B = patterns[1]
    C = patterns[2]
    D = patterns[3]

    a = base_factors[:, 0]
    b = base_factors[:, 1]
    c = base_factors[:, 2]
    d = base_factors[:, 3]

    def generate(a_, b_, c_, d_):
        x = (
            a_.view(-1, 1, 1) * A
            + b_.view(-1, 1, 1) * B
            + c_.view(-1, 1, 1) * C
            + d_.view(-1, 1, 1)
            * a_.view(-1, 1, 1)
            * b_.view(-1, 1, 1)
            * D
        )
        return x.flatten(1)

    baseline = generate(a, b, c, d)

    # Interventions.
    A_changed = generate(a + INTERVENTION_DELTA, b, c, d)
    B_changed = generate(a, b + INTERVENTION_DELTA, c, d)
    C_changed = generate(a, b, c + INTERVENTION_DELTA, d)
    D_changed = generate(a, b, c, d + INTERVENTION_DELTA)

    with torch.no_grad():
        z_base = model.encode(baseline.to(device)).cpu().numpy()
        z_A = model.encode(A_changed.to(device)).cpu().numpy()
        z_B = model.encode(B_changed.to(device)).cpu().numpy()
        z_C = model.encode(C_changed.to(device)).cpu().numpy()
        z_D = model.encode(D_changed.to(device)).cpu().numpy()

    disp = {
        "A": z_A - z_base,
        "B": z_B - z_base,
        "C": z_C - z_base,
        "interaction": z_D - z_base,
    }

    # ------------------------------------------------------------------
    # Displacement statistics
    # ------------------------------------------------------------------

    stats = {}

    for name, vectors in disp.items():
        norms = np.linalg.norm(vectors, axis=1)

        mean_vector = vectors.mean(axis=0)

        centered = vectors - mean_vector
        covariance = np.cov(centered.T)

        stats[name] = {
            "mean_vector": mean_vector.tolist(),
            "mean_magnitude": float(norms.mean()),
            "median_magnitude": float(np.median(norms)),
            "std_magnitude": float(norms.std()),
            "min_magnitude": float(norms.min()),
            "max_magnitude": float(norms.max()),
            "covariance": covariance.tolist(),
        }

    # ------------------------------------------------------------------
    # Direction consistency
    #
    # Compare every displacement to the mean displacement.
    # If interventions have a stable latent direction, angles should
    # cluster near zero.
    # ------------------------------------------------------------------

    angle_stats = {}

    for name, vectors in disp.items():
        mean_vector = vectors.mean(axis=0)

        angles = np.array([
            angle_between(v, mean_vector)
            for v in vectors
        ])

        # Because changing a factor can legitimately point in opposite
        # directions depending on whether the delta crosses zero, also
        # report the absolute cosine alignment.
        alignments = []

        norm_mean = np.linalg.norm(mean_vector)

        for v in vectors:
            nv = np.linalg.norm(v)

            if nv < 1e-12 or norm_mean < 1e-12:
                alignments.append(0.0)
            else:
                alignments.append(
                    abs(
                        float(
                            np.dot(v, mean_vector)
                            / (nv * norm_mean)
                        )
                    )
                )

        angle_stats[name] = {
            "mean_angle_deg": float(angles.mean()),
            "median_angle_deg": float(np.median(angles)),
            "std_angle_deg": float(angles.std()),
            "mean_absolute_cosine": float(np.mean(alignments)),
        }

    # ------------------------------------------------------------------
    # Interaction dependence
    #
    # Does interaction displacement depend on the current A/B state?
    # ------------------------------------------------------------------

    interaction = disp["interaction"]

    interaction_norm = np.linalg.norm(interaction, axis=1)

    interaction_dep = {
        "magnitude_vs_A": pearson(interaction_norm, a.numpy()),
        "magnitude_vs_B": pearson(interaction_norm, b.numpy()),
        "z1_vs_A": pearson(interaction[:, 0], a.numpy()),
        "z1_vs_B": pearson(interaction[:, 0], b.numpy()),
        "z2_vs_A": pearson(interaction[:, 1], a.numpy()),
        "z2_vs_B": pearson(interaction[:, 1], b.numpy()),
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    results = {
        "experiment": "known_factor_latent_intervention_geometry_v2",
        "source_experiment": "known_factor_synthetic_latent_v1",
        "n_probes": n,
        "intervention_delta": INTERVENTION_DELTA,
        "device": str(device),
        "displacement_statistics": stats,
        "direction_consistency": angle_stats,
        "interaction_dependence": interaction_dep,
    }

    with open(OUTPUT_DIR / "intervention_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Plot 1: displacement vectors
    # ------------------------------------------------------------------

    plt.figure(figsize=(9, 8))

    labels = {
        "A": "A intervention",
        "B": "B intervention",
        "C": "C intervention",
        "interaction": "Interaction intervention",
    }

    for name, vectors in disp.items():
        mean = vectors.mean(axis=0)

        plt.scatter(
            vectors[:, 0],
            vectors[:, 1],
            s=10,
            alpha=0.15,
            label=labels[name],
        )

        plt.arrow(
            0,
            0,
            mean[0],
            mean[1],
            width=0.002,
            head_width=0.03,
            length_includes_head=True,
        )

    plt.axhline(0, linewidth=0.7)
    plt.axvline(0, linewidth=0.7)
    plt.xlabel("Δz1")
    plt.ylabel("Δz2")
    plt.title("Latent displacement caused by controlled interventions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "intervention_displacements.png",
        dpi=150,
    )
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: direction consistency
    # ------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for name, vectors in disp.items():
        mean_vector = vectors.mean(axis=0)

        angles = np.array([
            angle_between(v, mean_vector)
            for v in vectors
        ])

        plt.hist(
            angles,
            bins=30,
            alpha=0.5,
            label=labels[name],
        )

    plt.xlabel("Angle from mean intervention direction (degrees)")
    plt.ylabel("Count")
    plt.title("Consistency of latent intervention directions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "intervention_angles.png",
        dpi=150,
    )
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: interaction displacement vs A/B
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc1 = axes[0].scatter(
        a.numpy(),
        interaction_norm,
        c=b.numpy(),
        s=12,
        alpha=0.6,
    )
    axes[0].set_xlabel("factor A")
    axes[0].set_ylabel("|Δz interaction|")
    axes[0].set_title("Interaction displacement vs A")
    fig.colorbar(sc1, ax=axes[0], label="factor B")

    sc2 = axes[1].scatter(
        b.numpy(),
        interaction_norm,
        c=a.numpy(),
        s=12,
        alpha=0.6,
    )
    axes[1].set_xlabel("factor B")
    axes[1].set_ylabel("|Δz interaction|")
    axes[1].set_title("Interaction displacement vs B")
    fig.colorbar(sc2, ax=axes[1], label="factor A")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "interaction_dependence.png",
        dpi=150,
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 4: a handful of individual intervention trajectories
    # ------------------------------------------------------------------

    plt.figure(figsize=(9, 8))

    for i in range(min(50, n)):
        points = np.array([
            z_base[i],
            z_A[i],
            z_B[i],
            z_C[i],
            z_D[i],
        ])

        plt.plot(
            points[:, 0],
            points[:, 1],
            alpha=0.25,
            linewidth=0.8,
        )

        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=8,
            alpha=0.5,
        )

    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Individual latent intervention trajectories")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sample_trajectories.png",
        dpi=150,
    )
    plt.close()

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print("\nMEAN LATENT DISPLACEMENTS")
    print("-" * 80)

    for name, s in stats.items():
        print(
            f"{name:12s} "
            f"Δz=[{s['mean_vector'][0]:+.6f}, "
            f"{s['mean_vector'][1]:+.6f}] "
            f"| magnitude={s['mean_magnitude']:.6f}"
        )

    print("\nDIRECTION CONSISTENCY")
    print("-" * 80)

    for name, s in angle_stats.items():
        print(
            f"{name:12s} "
            f"mean angle={s['mean_angle_deg']:.2f}° "
            f"| median={s['median_angle_deg']:.2f}° "
            f"| |cos|={s['mean_absolute_cosine']:.4f}"
        )

    print("\nINTERACTION DEPENDENCE")
    print("-" * 80)

    for key, value in interaction_dep.items():
        print(f"{key:16s}: {value:+.5f}")

    print()
    print(f"Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
