#!/usr/bin/env python3
"""
R1 — REAL DATASET SCALING EXPERIMENT (CIFAR-10)

Purpose
-------
Test whether the internal construction geometry we observed on the controlled
synthetic dataset emerges, stabilizes, or reorganizes when the model learns
from real, unlabeled images.

Dataset:
    CIFAR-10

Training sizes:
    500, 1,000, 5,000, 10,000, 25,000, 50,000

The CIFAR-10 test set is kept fixed and is NEVER used for training.

For each dataset size:
    - train the same small autoencoder
    - save checkpoint
    - evaluate reconstruction on the fixed test set
    - collect the 2-D latent coordinates of the test set
    - save latent geometry statistics
    - save latent scatter plots

No CIFAR labels are used during training or analysis in this first pass.

This is intentionally a clean baseline. We are NOT adding the synthetic
factor/vector-field machinery yet. First we establish whether a real-data
latent structure changes with data scale.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT.parent / "CIFAR-10"
DEFAULT_OUT = ROOT / "real_cifar10_scaling"

SIZES = [500, 1000, 5000, 10000, 25000, 50000]

INPUT_DIM = 32 * 32 * 3
LATENT_DIM = 2

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

SEED = 12345


class TinyRealAutoencoder(nn.Module):
    """
    Deliberately small autoencoder.

    The 2-D bottleneck is retained so the resulting latent space can be
    directly visualized and compared with our earlier experiments.
    """

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
        y = self.decode(z)
        return y, z


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_cifar_root(data_root):
    """
    torchvision accepts either:
        data_root/cifar-10-batches-py
    or a parent directory from which it can download/extract.

    We do not download automatically here.
    """

    candidates = [
        data_root,
        data_root / "cifar-10-batches-py",
        data_root / "cifar-10-python",
    ]

    for candidate in candidates:
        if (candidate / "data_batch_1").exists():
            return candidate.parent

    return data_root


def make_datasets(data_root):
    root = find_cifar_root(data_root)

    transform = transforms.ToTensor()

    train = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=False,
        transform=transform,
    )

    test = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=False,
        transform=transform,
    )

    return train, test


def flatten_dataset_item(x):
    return x.reshape(-1)


def make_subset(train, size, seed):
    if size > len(train):
        raise ValueError(
            f"Requested {size} samples but CIFAR-10 has {len(train)}."
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(train),
        size=size,
        replace=False,
    )

    indices = np.sort(indices)

    return Subset(train, indices.tolist()), indices.tolist()


def collate_flat(batch):
    xs = torch.stack([item[0] for item in batch])
    ys = torch.tensor([item[1] for item in batch])

    return xs.reshape(xs.shape[0], -1), ys


def train_one(model, loader, device, epochs, lr):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    loss_fn = nn.MSELoss()

    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        total = 0.0
        count = 0

        for x, _ in loader:
            x = x.to(device)

            optimizer.zero_grad()

            recon, _ = model(x)

            loss = loss_fn(recon, x)

            loss.backward()
            optimizer.step()

            total += loss.item() * x.shape[0]
            count += x.shape[0]

        mean_loss = total / max(count, 1)
        history.append(mean_loss)

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"epoch {epoch:4d}/{epochs} | "
                f"train MSE {mean_loss:.8f}"
            )

    return history


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=512):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_flat,
    )

    model.eval()

    losses = []
    latent = []
    reconstructions = []
    originals = []

    loss_fn = nn.MSELoss(reduction="none")

    for x, _ in loader:
        x = x.to(device)

        recon, z = model(x)

        per_sample = loss_fn(
            recon,
            x,
        ).mean(dim=1)

        losses.append(
            per_sample.cpu().numpy()
        )

        latent.append(
            z.cpu().numpy()
        )

        reconstructions.append(
            recon.cpu().numpy()
        )

        originals.append(
            x.cpu().numpy()
        )

    losses = np.concatenate(losses)
    latent = np.concatenate(latent)

    return {
        "mse_mean": float(losses.mean()),
        "mse_median": float(np.median(losses)),
        "mse_p95": float(np.percentile(losses, 95)),
        "latent": latent,
    }


def latent_stats(z):
    cov = np.cov(z.T)

    eigvals = np.linalg.eigvalsh(cov)

    spread0 = float(np.std(z[:, 0]))
    spread1 = float(np.std(z[:, 1]))

    return {
        "mean_z0": float(np.mean(z[:, 0])),
        "mean_z1": float(np.mean(z[:, 1])),
        "std_z0": spread0,
        "std_z1": spread1,
        "covariance": cov.tolist(),
        "latent_cov_eig_low": float(eigvals[0]),
        "latent_cov_eig_high": float(eigvals[1]),
        "latent_cov_anisotropy": float(
            eigvals[1] / max(eigvals[0], 1e-12)
        ),
        "latent_radius_mean": float(
            np.mean(np.linalg.norm(z, axis=1))
        ),
        "latent_radius_p95": float(
            np.percentile(
                np.linalg.norm(z, axis=1),
                95,
            )
        ),
    }


def save_latent_plot(z, size, out):
    plt.figure(figsize=(8, 7))

    plt.scatter(
        z[:, 0],
        z[:, 1],
        s=4,
        alpha=0.35,
    )

    plt.xlabel("latent z0")
    plt.ylabel("latent z1")
    plt.title(
        f"CIFAR-10 latent space — training size {size:,}"
    )

    plt.tight_layout()

    plt.savefig(
        out / f"latent_{size}.png",
        dpi=160,
    )

    plt.close()


def save_reconstruction_examples(model, dataset, device, size, out):
    """
    Save a small visual sanity check.

    This is not used for training or measurement.
    """
    model.eval()

    indices = np.linspace(
        0,
        len(dataset) - 1,
        8,
        dtype=int,
    )

    images = []

    with torch.no_grad():
        for idx in indices:
            x, _ = dataset[idx]

            flat = x.reshape(1, -1).to(device)

            recon, _ = model(flat)

            recon = recon.reshape(3, 32, 32)
            recon = recon.cpu().numpy()

            images.append(
                (x.numpy(), recon)
            )

    fig, axes = plt.subplots(
        2,
        len(images),
        figsize=(16, 4),
    )

    for i, (original, recon) in enumerate(images):
        axes[0, i].imshow(
            np.transpose(original, (1, 2, 0))
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(
            np.transpose(recon, (1, 2, 0))
        )
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("recon")

    fig.suptitle(
        f"CIFAR-10 reconstruction — training size {size:,}"
    )

    plt.tight_layout()

    plt.savefig(
        out / f"reconstruction_{size}.png",
        dpi=160,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
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

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print("R1 — CIFAR-10 REAL-DATASET SCALING")
    print("=" * 90)
    print(f"Device:     {device}")
    print(f"Data root:  {args.data_root}")
    print(f"Output:     {args.out}")
    print(f"Sizes:      {args.sizes}")
    print(f"Epochs:     {args.epochs}")
    print(f"Latent dim: {LATENT_DIM}")
    print()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, test = make_datasets(args.data_root)

    print(
        f"Loaded CIFAR-10: "
        f"{len(train)} train / {len(test)} test"
    )
    print()

    # Fixed test set for EVERY model.
    fixed_test = test

    all_results = []

    for size in args.sizes:
        print()
        print("=" * 90)
        print(f"TRAINING SIZE: {size:,}")
        print("=" * 90)

        subset, indices = make_subset(
            train,
            size,
            args.seed,
        )

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_flat,
        )

        model = TinyRealAutoencoder().to(device)

        start = time.time()

        history = train_one(
            model,
            loader,
            device,
            args.epochs,
            LR,
        )

        elapsed = time.time() - start

        evaluation = evaluate(
            model,
            fixed_test,
            device,
        )

        z = evaluation["latent"]

        stats = latent_stats(z)

        result = {
            "training_size": size,
            "seed": args.seed,
            "epochs": args.epochs,
            "elapsed_seconds": elapsed,
            "train_mse_final": history[-1],
            "test_mse_mean": evaluation["mse_mean"],
            "test_mse_median": evaluation["mse_median"],
            "test_mse_p95": evaluation["mse_p95"],
            **stats,
        }

        all_results.append(result)

        model_path = (
            args.out /
            f"cifar10_autoencoder_{size}.pt"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_size": size,
                "seed": args.seed,
                "epochs": args.epochs,
                "input_dim": INPUT_DIM,
                "latent_dim": LATENT_DIM,
            },
            model_path,
        )

        np.save(
            args.out /
            f"latent_test_{size}.npy",
            z,
        )

        with open(
            args.out /
            f"summary_{size}.json",
            "w",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        save_latent_plot(
            z,
            size,
            args.out,
        )

        save_reconstruction_examples(
            model,
            fixed_test,
            device,
            size,
            args.out,
        )

        with open(
            args.out /
            f"train_history_{size}.json",
            "w",
        ) as f:
            json.dump(
                history,
                f,
            )

        print()
        print(
            f"FINAL | size={size:,} | "
            f"test MSE={evaluation['mse_mean']:.8f} | "
            f"latent anisotropy="
            f"{stats['latent_cov_anisotropy']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(
        args.out / "scaling_summary.json",
        "w",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    # Summary plots.
    sizes = np.array(
        [r["training_size"] for r in all_results]
    )

    mse = np.array(
        [r["test_mse_mean"] for r in all_results]
    )

    anis = np.array(
        [r["latent_cov_anisotropy"] for r in all_results]
    )

    radius = np.array(
        [r["latent_radius_mean"] for r in all_results]
    )

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, mse, marker="o")
    plt.xscale("log")
    plt.xlabel("training examples")
    plt.ylabel("fixed-test reconstruction MSE")
    plt.title("CIFAR-10 reconstruction vs dataset size")
    plt.tight_layout()
    plt.savefig(
        args.out / "scaling_reconstruction.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, anis, marker="o")
    plt.xscale("log")
    plt.xlabel("training examples")
    plt.ylabel("latent covariance anisotropy")
    plt.title("CIFAR-10 latent anisotropy vs dataset size")
    plt.tight_layout()
    plt.savefig(
        args.out / "scaling_latent_anisotropy.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, radius, marker="o")
    plt.xscale("log")
    plt.xlabel("training examples")
    plt.ylabel("mean latent radius")
    plt.title("CIFAR-10 latent extent vs dataset size")
    plt.tight_layout()
    plt.savefig(
        args.out / "scaling_latent_radius.png",
        dpi=160,
    )
    plt.close()

    print()
    print("=" * 90)
    print("R1 COMPLETE")
    print("=" * 90)
    print()
    print(
        json.dumps(
            all_results,
            indent=2,
        )
    )
    print()
    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()
