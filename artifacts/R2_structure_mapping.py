#!/usr/bin/env python3
"""
R2 — CIFAR-10 STRUCTURE MAPPING

Reads the saved R1 CIFAR-10 scaling runs and analyzes the SAME fixed
10,000-image evaluation set across training sizes.

Questions:
  1. How do individual images move through latent space as more data is seen?
  2. Is the 40k diagonal a true 1-D collapse / degeneracy?
  3. Does latent geometry organize CIFAR classes without using labels for training?
  4. Does latent neighborhood structure preserve pixel-space neighborhoods?
  5. Does reconstruction quality depend on location in latent space?
  6. How does geometry change with dataset size?

The script does NOT retrain anything.

Expected R1 output:
  real_cifar10_scaling_parquet/
    cifar10_autoencoder_<size>.pt
    latent_test_<size>.npy
    summary_<size>.json

The fixed evaluation split is reconstructed from the same seed used by R1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import io
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEFAULT_DATA_ROOT = Path("/home/xetal/Desktop/latent_model_test/cifar10")
DEFAULT_R1_OUT = Path(
    "/home/xetal/Desktop/latent_model_test/"
    "synthetic_latent_v1/real_cifar10_scaling_parquet"
)
DEFAULT_OUT = DEFAULT_R1_OUT / "structure_mapping"

SIZES = [500, 1000, 5000, 10000, 25000, 40000]
TEST_SIZE = 10000
SEED = 12345

INPUT_DIM = 32 * 32 * 3
LATENT_DIM = 2
BATCH_SIZE = 512

# Number of test images used for expensive neighborhood analysis.
KNN_SAMPLE = 2000
K = 10


class TinyRealAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, INPUT_DIM),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class CIFARParquetDataset(Dataset):
    def __init__(self, parquet_path):
        self.parquet_path = Path(parquet_path)
        self.df = pd.read_parquet(self.parquet_path)

        required = {"img", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise RuntimeError(
                f"Parquet file is missing columns: {sorted(missing)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]["img"]
        raw = record["bytes"]

        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0

        x = torch.from_numpy(
            np.transpose(array, (2, 0, 1)).copy()
        )
        label = int(self.df.iloc[idx]["label"])
        return x, label


def find_parquet(data_root: Path) -> Path:
    candidates = [
        data_root / "plain_text" / "train-00000-of-00001.parquet",
        data_root / "train-00000-of-00001.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find CIFAR-10 Parquet file. Checked:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def fixed_test_indices(n: int, test_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n)
    return np.sort(
        rng.choice(all_indices, size=test_size, replace=False)
    )


def load_fixed_test(data_root: Path, seed: int):
    ds = CIFARParquetDataset(find_parquet(data_root))
    indices = fixed_test_indices(len(ds), TEST_SIZE, seed)

    images = []
    labels = []

    print(f"Loading fixed evaluation set: {len(indices):,} images")

    for idx in indices:
        x, label = ds[int(idx)]
        images.append(x.numpy().reshape(-1))
        labels.append(label)

    images = np.stack(images).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return ds, indices, images, labels


def load_latents(r1_out: Path, sizes):
    latents = {}
    for size in sizes:
        path = r1_out / f"latent_test_{size}.npy"
        if not path.exists():
            print(f"WARNING: missing {path}; skipping {size}")
            continue
        latents[size] = np.load(path).astype(np.float64)
    return latents


def center(x):
    return x - np.mean(x, axis=0, keepdims=True)


def align_2d(reference, target):
    """
    Align target to reference using centered orthogonal Procrustes plus
    isotropic scale. This removes arbitrary translation, rotation/reflection,
    and global scale so image trajectories are meaningful.
    """
    a = center(reference)
    b = center(target)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return b.copy()

    a_n = a / norm_a
    b_n = b / norm_b

    m = b_n.T @ a_n
    u, _, vt = np.linalg.svd(m)
    r = u @ vt

    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    # Match reference RMS scale after rotation.
    aligned = b @ r
    scale = norm_a / max(norm_b, 1e-12)
    return aligned * scale


def geometry(z):
    c = np.cov(z.T)
    eig = np.linalg.eigvalsh(c)
    eig = np.maximum(eig, 0)

    total = eig.sum()
    pr = (total * total) / max((eig * eig).sum(), 1e-12)

    corr = np.corrcoef(z.T)[0, 1]

    return {
        "mean_z0": float(z[:, 0].mean()),
        "mean_z1": float(z[:, 1].mean()),
        "std_z0": float(z[:, 0].std()),
        "std_z1": float(z[:, 1].std()),
        "covariance": c.tolist(),
        "eig_low": float(eig[0]),
        "eig_high": float(eig[1]),
        "anisotropy": float(eig[1] / max(eig[0], 1e-12)),
        "participation_ratio": float(pr),
        "corr_z0_z1": float(corr),
        "mean_radius": float(np.linalg.norm(z, axis=1).mean()),
        "p95_radius": float(
            np.percentile(np.linalg.norm(z, axis=1), 95)
        ),
    }


def load_reconstruction_errors(
    r1_out: Path,
    data_root: Path,
    sizes,
    seed,
    device,
):
    """
    Re-run only inference against the fixed test set so we get a
    per-image reconstruction error. No training occurs.
    """
    ds = CIFARParquetDataset(find_parquet(data_root))
    indices = fixed_test_indices(len(ds), TEST_SIZE, seed)

    class FixedSubset(Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, i):
            return ds[int(indices[i])]

    subset = FixedSubset()

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    errors = {}

    for size in sizes:
        checkpoint = r1_out / f"cifar10_autoencoder_{size}.pt"
        if not checkpoint.exists():
            print(f"WARNING: missing checkpoint {checkpoint}")
            continue

        print(f"Inference for reconstruction errors: {size:,}")

        model = TinyRealAutoencoder().to(device)
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        all_errors = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.reshape(x.shape[0], -1).to(device)
                recon, _ = model(x)
                per = ((recon - x) ** 2).mean(dim=1)
                all_errors.append(per.cpu().numpy())

        errors[size] = np.concatenate(all_errors)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return errors


def nearest_neighbors(points, sample_indices, k):
    """
    Exact kNN using torch in chunks. Distances are computed only for the
    sampled query points, against the full point set.
    """
    p = torch.from_numpy(points.astype(np.float32))
    q = torch.from_numpy(points[sample_indices].astype(np.float32))

    result = []
    chunk = 256

    for start in range(0, len(q), chunk):
        qc = q[start:start + chunk]
        d = (
            (qc[:, None, :] - p[None, :, :]) ** 2
        ).sum(dim=2)

        # Exclude self if query belongs to the full set.
        query_ids = sample_indices[start:start + len(qc)]
        d[
            torch.arange(len(qc)),
            torch.from_numpy(query_ids),
        ] = float("inf")

        idx = torch.topk(
            d,
            k=k,
            dim=1,
            largest=False,
        ).indices.numpy()

        result.append(idx)

    return np.concatenate(result, axis=0)


def neighborhood_overlap(
    images,
    latent,
    sample_indices,
    k=10,
):
    pixel_nn = nearest_neighbors(images, sample_indices, k)
    latent_nn = nearest_neighbors(latent, sample_indices, k)

    overlaps = []
    for a, b in zip(pixel_nn, latent_nn):
        overlaps.append(
            len(set(a.tolist()) & set(b.tolist())) / k
        )

    return float(np.mean(overlaps))


def class_geometry(z, labels):
    """
    Labels are analysis-only. Training did not use them.
    """
    overall = z.mean(axis=0)

    between = 0.0
    within = 0.0

    rows = []

    for cls in range(10):
        mask = labels == cls
        if not np.any(mask):
            continue

        points = z[mask]
        mu = points.mean(axis=0)

        between += len(points) * float(
            np.sum((mu - overall) ** 2)
        )
        within += float(
            np.sum((points - mu) ** 2)
        )

        rows.append({
            "class": cls,
            "count": int(len(points)),
            "mean_z0": float(mu[0]),
            "mean_z1": float(mu[1]),
            "radius_mean": float(
                np.linalg.norm(points - mu, axis=1).mean()
            ),
        })

    ratio = between / max(within, 1e-12)

    return {
        "between_within_ratio": float(ratio),
        "classes": rows,
    }


def plot_aligned_trajectories(aligned, sizes, out):
    # One line per sampled evaluation image.
    sample = np.linspace(
        0,
        aligned[sizes[0]].shape[0] - 1,
        300,
        dtype=int,
    )

    plt.figure(figsize=(10, 8))

    for idx in sample:
        xy = np.stack(
            [aligned[s][idx] for s in sizes],
            axis=0,
        )
        plt.plot(xy[:, 0], xy[:, 1], alpha=0.12, linewidth=0.6)

    for s in sizes:
        xy = aligned[s][sample]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=5,
            alpha=0.35,
        )

    plt.xlabel("aligned latent z0")
    plt.ylabel("aligned latent z1")
    plt.title("Same-image latent trajectories as training data increases")
    plt.tight_layout()
    plt.savefig(out / "image_trajectories.png", dpi=180)
    plt.close()


def plot_trajectory_magnitude(aligned, sizes, out):
    reference = sizes[0]
    rows = []

    for s in sizes:
        delta = aligned[s] - aligned[reference]
        mag = np.linalg.norm(delta, axis=1)
        rows.append(
            (
                s,
                float(np.mean(mag)),
                float(np.median(mag)),
                float(np.percentile(mag, 95)),
            )
        )

    with open(out / "trajectory_magnitude.csv", "w") as f:
        f.write("training_size,mean,median,p95\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")

    xs = [r[0] for r in rows]
    means = [r[1] for r in rows]

    plt.figure(figsize=(9, 6))
    plt.plot(xs, means, marker="o")
    plt.xscale("log")
    plt.xlabel("training examples")
    plt.ylabel("mean aligned displacement from smallest run")
    plt.title("Latent movement of the same images")
    plt.tight_layout()
    plt.savefig(out / "trajectory_magnitude.png", dpi=180)
    plt.close()


def plot_class_latents(z, labels, size, out):
    plt.figure(figsize=(9, 8))

    for cls in range(10):
        mask = labels == cls
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=5,
            alpha=0.25,
            label=str(cls),
        )

    plt.xlabel("latent z0")
    plt.ylabel("latent z1")
    plt.title(f"Latent space by CIFAR-10 class — {size:,}")
    plt.legend(
        title="class",
        markerscale=3,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(
        out / f"class_latent_{size}.png",
        dpi=180,
    )
    plt.close()


def plot_reconstruction_map(z, errors, size, out):
    plt.figure(figsize=(9, 8))

    # Color encodes per-image reconstruction error.
    sc = plt.scatter(
        z[:, 0],
        z[:, 1],
        c=errors,
        s=5,
        alpha=0.35,
    )

    plt.colorbar(sc, label="per-image reconstruction MSE")
    plt.xlabel("latent z0")
    plt.ylabel("latent z1")
    plt.title(f"Reconstruction error across latent space — {size:,}")
    plt.tight_layout()
    plt.savefig(
        out / f"reconstruction_map_{size}.png",
        dpi=180,
    )
    plt.close()


def plot_geometry(rows, out):
    sizes = [r["training_size"] for r in rows]

    metrics = [
        ("anisotropy", "latent anisotropy"),
        ("participation_ratio", "effective latent dimensionality"),
        ("mean_radius", "mean latent radius"),
        ("corr_z0_z1", "z0/z1 correlation"),
    ]

    for key, ylabel in metrics:
        plt.figure(figsize=(9, 6))
        plt.plot(
            sizes,
            [r[key] for r in rows],
            marker="o",
        )
        plt.xscale("log")
        plt.xlabel("training examples")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs dataset size")
        plt.tight_layout()
        plt.savefig(
            out / f"geometry_{key}.png",
            dpi=180,
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r1-out", type=Path, default=DEFAULT_R1_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=SIZES,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--knn-sample",
        type=int,
        default=KNN_SAMPLE,
    )

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 90)
    print("R2 — CIFAR-10 STRUCTURE MAPPING")
    print("=" * 90)
    print(f"Device:      {device}")
    print(f"R1 output:   {args.r1_out}")
    print(f"Output:      {args.out}")
    print(f"Sizes:       {args.sizes}")
    print()

    _, test_indices, images, labels = load_fixed_test(
        args.data_root,
        args.seed,
    )

    latents = load_latents(args.r1_out, args.sizes)

    sizes = [s for s in args.sizes if s in latents]
    if len(sizes) < 2:
        raise RuntimeError(
            "Need at least two latent_test_<size>.npy files."
        )

    # Align every representation to the smallest available run.
    reference = sizes[0]
    aligned = {reference: latents[reference]}

    for size in sizes[1:]:
        aligned[size] = align_2d(
            latents[reference],
            latents[size],
        )

    print(f"Reference latent geometry: {reference:,}")

    # 1. Geometry and collapse diagnostics.
    geometry_rows = []
    for size in sizes:
        raw = geometry(latents[size])
        raw["training_size"] = size

        # Also measure aligned geometry for comparison.
        raw["aligned"] = geometry(aligned[size])
        geometry_rows.append(raw)

    plot_geometry(geometry_rows, args.out)

    with open(args.out / "geometry_report.json", "w") as f:
        json.dump(geometry_rows, f, indent=2)

    # 2. Same-image trajectories.
    plot_aligned_trajectories(aligned, sizes, args.out)
    plot_trajectory_magnitude(aligned, sizes, args.out)

    # Per-image displacement table.
    trajectory = {}
    for size in sizes:
        delta = aligned[size] - aligned[reference]
        trajectory[str(size)] = {
            "mean": float(np.linalg.norm(delta, axis=1).mean()),
            "median": float(np.median(np.linalg.norm(delta, axis=1))),
            "p95": float(
                np.percentile(np.linalg.norm(delta, axis=1), 95)
            ),
        }

    # 3. Class structure.
    class_reports = {}
    for size in sizes:
        class_reports[str(size)] = class_geometry(
            latents[size],
            labels,
        )
        plot_class_latents(
            latents[size],
            labels,
            size,
            args.out,
        )

    with open(args.out / "class_geometry.json", "w") as f:
        json.dump(class_reports, f, indent=2)

    # 4. Reconstruction-error maps.
    errors = load_reconstruction_errors(
        args.r1_out,
        args.data_root,
        sizes,
        args.seed,
        device,
    )

    reconstruction_report = {}

    for size in sizes:
        if size not in errors:
            continue

        e = errors[size]
        z = latents[size]

        # Correlate reconstruction error with radius and latent coordinates.
        radius = np.linalg.norm(z - z.mean(axis=0), axis=1)

        reconstruction_report[str(size)] = {
            "mean_mse": float(e.mean()),
            "median_mse": float(np.median(e)),
            "p95_mse": float(np.percentile(e, 95)),
            "corr_mse_radius": float(np.corrcoef(e, radius)[0, 1]),
            "corr_mse_z0": float(np.corrcoef(e, z[:, 0])[0, 1]),
            "corr_mse_z1": float(np.corrcoef(e, z[:, 1])[0, 1]),
        }

        plot_reconstruction_map(
            z,
            e,
            size,
            args.out,
        )

    with open(args.out / "reconstruction_geometry.json", "w") as f:
        json.dump(reconstruction_report, f, indent=2)

    # 5. Pixel-neighborhood -> latent-neighborhood preservation.
    # Use the same sampled image IDs across every model.
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.knn_sample, len(images))
    sample_indices = np.sort(
        rng.choice(
            len(images),
            size=sample_count,
            replace=False,
        )
    )

    neighborhood_report = {}

    print()
    print(f"Neighborhood analysis on {sample_count:,} images")

    for size in sizes:
        print(f"  kNN overlap: {size:,}")
        overlap = neighborhood_overlap(
            images,
            latents[size],
            sample_indices,
            K,
        )
        neighborhood_report[str(size)] = {
            "k": K,
            "sample_count": sample_count,
            "pixel_to_latent_neighbor_overlap": overlap,
        }

    with open(args.out / "neighborhood_report.json", "w") as f:
        json.dump(neighborhood_report, f, indent=2)

    # Summary.
    report = {
        "experiment": "R2_structure_mapping",
        "seed": args.seed,
        "fixed_test_size": len(test_indices),
        "sizes": sizes,
        "reference_size": reference,
        "trajectory": trajectory,
        "geometry": geometry_rows,
        "class_geometry": class_reports,
        "reconstruction_geometry": reconstruction_report,
        "neighborhood": neighborhood_report,
    }

    with open(args.out / "R2_structure_mapping_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 90)
    print("R2 COMPLETE")
    print("=" * 90)
    print(f"Results: {args.out}")
    print()
    print("Key files:")
    print("  R2_structure_mapping_report.json")
    print("  geometry_report.json")
    print("  class_geometry.json")
    print("  reconstruction_geometry.json")
    print("  neighborhood_report.json")
    print("  image_trajectories.png")
    print("  trajectory_magnitude.png")


if __name__ == "__main__":
    main()
