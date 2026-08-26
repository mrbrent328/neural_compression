#!/usr/bin/env python3
"""
R2 SIMPLE — SAFE STRUCTURE CHECK

A deliberately small, memory-safe follow-up to R1.

It does NOT retrain anything.

It only uses the saved R1 latent_test_<size>.npy files and asks four
straightforward questions:

1. How does latent geometry change with dataset size?
2. How much do the SAME evaluation images move between models?
3. Do CIFAR-10 classes naturally separate in the latent space?
4. Is the 40k diagonal strongly one-dimensional?

No all-pairs image-distance matrix.
No giant tensors.
No expensive reconstruction inference.
No 10,000 x 10,000 distance calculation.

This is intentionally a "don't melt the computer" version.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_R1 = Path(
    "/home/xetal/Desktop/latent_model_test/"
    "synthetic_latent_v1/real_cifar10_scaling_parquet"
)

DEFAULT_OUT = DEFAULT_R1 / "simple_structure_check"

SIZES = [500, 1000, 5000, 10000, 25000, 40000]
SEED = 12345
TEST_SIZE = 10000


def load_latents(r1_out, sizes):
    result = {}

    for size in sizes:
        path = r1_out / f"latent_test_{size}.npy"

        if not path.exists():
            print(f"Skipping missing: {path}")
            continue

        z = np.load(path, mmap_mode="r")

        if z.ndim != 2 or z.shape[1] != 2:
            raise RuntimeError(
                f"Unexpected latent shape for {size}: {z.shape}"
            )

        result[size] = z

    return result


def geometry(z):
    # Work on a small 2-column array only.
    x = np.asarray(z, dtype=np.float64)

    mean = x.mean(axis=0)
    centered = x - mean

    cov = np.cov(centered.T)
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0)

    anisotropy = eig[1] / max(eig[0], 1e-12)

    participation = (
        eig.sum() ** 2 /
        max((eig ** 2).sum(), 1e-12)
    )

    corr = np.corrcoef(x.T)[0, 1]

    return {
        "mean_z0": float(mean[0]),
        "mean_z1": float(mean[1]),
        "std_z0": float(x[:, 0].std()),
        "std_z1": float(x[:, 1].std()),
        "eig_low": float(eig[0]),
        "eig_high": float(eig[1]),
        "anisotropy": float(anisotropy),
        "participation_ratio": float(participation),
        "corr_z0_z1": float(corr),
        "mean_radius": float(
            np.linalg.norm(centered, axis=1).mean()
        ),
    }


def align_to_reference(reference, target):
    """
    Align target to reference using 2D orthogonal Procrustes.

    Only the 2D latent coordinates are copied into RAM.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)

    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)

    scale_a = np.linalg.norm(a)
    scale_b = np.linalg.norm(b)

    if scale_a < 1e-12 or scale_b < 1e-12:
        return b

    aa = a / scale_a
    bb = b / scale_b

    u, _, vt = np.linalg.svd(bb.T @ aa)
    r = u @ vt

    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    aligned = b @ r
    aligned *= scale_a / scale_b

    return aligned


def trajectory_stats(reference, target):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)

    # Alignment is done globally, then displacement is calculated per image.
    b_aligned = align_to_reference(a, b)

    displacement = np.linalg.norm(b_aligned - a, axis=1)

    return {
        "mean": float(displacement.mean()),
        "median": float(np.median(displacement)),
        "p95": float(np.percentile(displacement, 95)),
        "max": float(displacement.max()),
    }, b_aligned


def class_separation(z, labels):
    """
    Very simple label-only diagnostic.

    Labels are NEVER used for training. They are only used here to ask
    whether semantic classes happen to occupy different regions.
    """
    z = np.asarray(z, dtype=np.float64)
    overall = z.mean(axis=0)

    between = 0.0
    within = 0.0

    class_centers = []

    for cls in range(10):
        mask = labels == cls

        if not np.any(mask):
            continue

        points = z[mask]
        center = points.mean(axis=0)

        between += len(points) * np.sum(
            (center - overall) ** 2
        )

        within += np.sum(
            (points - center) ** 2
        )

        class_centers.append(
            [cls, float(center[0]), float(center[1])]
        )

    return {
        "between_within_ratio": float(
            between / max(within, 1e-12)
        ),
        "class_centers": class_centers,
    }


def load_labels(data_root, seed):
    """
    Reconstruct exactly the same fixed 10k evaluation split used by R1.
    Only labels are loaded here, not images.
    """
    import pandas as pd

    parquet = (
        data_root /
        "plain_text" /
        "train-00000-of-00001.parquet"
    )

    df = pd.read_parquet(
        parquet,
        columns=["label"],
    )

    rng = np.random.default_rng(seed)

    indices = np.sort(
        rng.choice(
            len(df),
            size=TEST_SIZE,
            replace=False,
        )
    )

    return df.iloc[indices]["label"].to_numpy(
        dtype=np.int64
    )


def plot_geometry(rows, out):
    sizes = [r["training_size"] for r in rows]

    metrics = [
        ("anisotropy", "Latent anisotropy"),
        ("participation_ratio", "Effective dimensionality"),
        ("mean_radius", "Mean centered radius"),
    ]

    for key, title in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(
            sizes,
            [r[key] for r in rows],
            marker="o",
        )
        plt.xscale("log")
        plt.xlabel("Training examples")
        plt.ylabel(title)
        plt.title(title + " vs dataset size")
        plt.tight_layout()
        plt.savefig(
            out / f"{key}_scaling.png",
            dpi=160,
        )
        plt.close()


def plot_latent(z, size, out):
    # Plot a maximum of 5,000 points to keep rendering lightweight.
    z = np.asarray(z)

    if len(z) > 5000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(z), 5000, replace=False)
        z = z[np.sort(idx)]

    plt.figure(figsize=(7, 7))
    plt.scatter(
        z[:, 0],
        z[:, 1],
        s=3,
        alpha=0.25,
    )
    plt.xlabel("latent z0")
    plt.ylabel("latent z1")
    plt.title(f"Latent space — {size:,} training examples")
    plt.tight_layout()
    plt.savefig(
        out / f"latent_{size}.png",
        dpi=160,
    )
    plt.close()


def plot_trajectories(aligned, sizes, out):
    # Only show 300 trajectories.
    n = len(aligned[sizes[0]])
    rng = np.random.default_rng(SEED)
    sample = np.sort(
        rng.choice(
            n,
            size=min(300, n),
            replace=False,
        )
    )

    plt.figure(figsize=(9, 7))

    for idx in sample:
        xy = np.array(
            [aligned[size][idx] for size in sizes]
        )
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            alpha=0.12,
            linewidth=0.6,
        )

    for size in sizes:
        xy = aligned[size][sample]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=8,
            alpha=0.35,
        )

    plt.xlabel("aligned z0")
    plt.ylabel("aligned z1")
    plt.title("Same-image latent movement across training sizes")
    plt.tight_layout()
    plt.savefig(
        out / "same_image_trajectories.png",
        dpi=160,
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--r1-out",
        type=Path,
        default=DEFAULT_R1,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/home/xetal/Desktop/latent_model_test/cifar10"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=SIZES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("R2 SIMPLE — SAFE STRUCTURE CHECK")
    print("=" * 82)
    print("No retraining.")
    print("No giant distance matrices.")
    print("No image tensor loaded into memory.")
    print()

    latents = load_latents(
        args.r1_out,
        args.sizes,
    )

    sizes = [
        s for s in args.sizes
        if s in latents
    ]

    if len(sizes) < 2:
        raise RuntimeError(
            "Need at least two saved R1 latent files."
        )

    print("Found latent runs:")
    for s in sizes:
        print(
            f"  {s:>6,}: "
            f"{latents[s].shape}"
        )

    # ------------------------------------------------------------------
    # 1. Basic geometry.
    # ------------------------------------------------------------------
    print()
    print("GEOMETRY")
    print("-" * 82)

    rows = []

    for size in sizes:
        g = geometry(latents[size])
        g["training_size"] = size
        rows.append(g)

        print(
            f"{size:>6,} | "
            f"anisotropy={g['anisotropy']:.4f} | "
            f"PR={g['participation_ratio']:.4f} | "
            f"corr={g['corr_z0_z1']:.6f} | "
            f"radius={g['mean_radius']:.4f}"
        )

    plot_geometry(rows, args.out)

    # ------------------------------------------------------------------
    # 2. Same-image trajectories.
    # ------------------------------------------------------------------
    print()
    print("SAME-IMAGE LATENT MOVEMENT")
    print("-" * 82)

    reference = sizes[0]
    aligned = {
        reference: np.asarray(
            latents[reference],
            dtype=np.float64,
        )
    }

    trajectory_report = {}

    for size in sizes:
        if size == reference:
            trajectory_report[str(size)] = {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "max": 0.0,
            }
            continue

        stats, aligned_target = trajectory_stats(
            latents[reference],
            latents[size],
        )

        aligned[size] = aligned_target
        trajectory_report[str(size)] = stats

        print(
            f"{size:>6,} vs {reference:,} | "
            f"mean={stats['mean']:.4f} | "
            f"median={stats['median']:.4f} | "
            f"p95={stats['p95']:.4f}"
        )

    plot_trajectories(
        aligned,
        sizes,
        args.out,
    )

    # ------------------------------------------------------------------
    # 3. Class structure.
    # ------------------------------------------------------------------
    print()
    print("CLASS STRUCTURE")
    print("-" * 82)

    labels = load_labels(
        args.data_root,
        args.seed,
    )

    class_report = {}

    for size in sizes:
        report = class_separation(
            latents[size],
            labels,
        )

        class_report[str(size)] = report

        print(
            f"{size:>6,} | "
            f"between/within={report['between_within_ratio']:.6f}"
        )

        # Simple class-center plot. No giant arrays.
        centers = np.asarray(
            report["class_centers"],
            dtype=float,
        )

        plt.figure(figsize=(7, 6))

        plt.scatter(
            centers[:, 1],
            centers[:, 2],
            s=60,
        )

        for cls, x, y in centers:
            plt.annotate(
                str(int(cls)),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.xlabel("latent z0")
        plt.ylabel("latent z1")
        plt.title(
            f"Class centers — {size:,} training examples"
        )
        plt.tight_layout()
        plt.savefig(
            args.out / f"class_centers_{size}.png",
            dpi=160,
        )
        plt.close()

    # ------------------------------------------------------------------
    # 4. 40k / high-anisotropy diagnostic.
    # ------------------------------------------------------------------
    collapse_report = {}

    for size in sizes:
        z = np.asarray(
            latents[size],
            dtype=np.float64,
        )

        g = geometry(z)

        collapse_report[str(size)] = {
            "anisotropy": g["anisotropy"],
            "participation_ratio": g["participation_ratio"],
            "correlation": g["corr_z0_z1"],
            "eig_low": g["eig_low"],
            "eig_high": g["eig_high"],
            "likely_near_1d": bool(
                g["participation_ratio"] < 1.2
            ),
        }

    # ------------------------------------------------------------------
    # Save one simple report.
    # ------------------------------------------------------------------
    report = {
        "experiment": "R2_simple_structure_check",
        "seed": args.seed,
        "sizes": sizes,
        "reference_size": reference,
        "geometry": rows,
        "same_image_trajectory": trajectory_report,
        "class_structure": class_report,
        "collapse_diagnostics": collapse_report,
    }

    report_path = (
        args.out /
        "R2_simple_structure_report.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 82)
    print("DONE")
    print("=" * 82)
    print(f"Results: {args.out}")
    print(f"Report:  {report_path}")
    print()
    print("Generated:")
    print("  anisotropy_scaling.png")
    print("  participation_ratio_scaling.png")
    print("  mean_radius_scaling.png")
    print("  same_image_trajectories.png")
    print("  class_centers_<size>.png")
    print("  R2_simple_structure_report.json")


if __name__ == "__main__":
    main()
