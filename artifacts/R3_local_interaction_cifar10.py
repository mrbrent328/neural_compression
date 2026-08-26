#!/usr/bin/env python3
"""
R3 — LOCAL REPRESENTATIONS + SHARED INTERACTION

First real-data test of the architecture suggested by the R1/R2 results.

Core idea
---------
Do NOT force the whole image through one global 2-D bottleneck.

Instead:

    image
      |
      v
    local patches
      |
      v
    local representations
      |
      v
    shared interaction layer
      |
      v
    reconstructed local patches
      |
      v
    image

There is deliberately NO learned class label and NO per-image identity.

The architecture is intentionally small so this can be run on the same
CIFAR-10 Parquet dataset without turning the machine into a space heater.

CIFAR-10 is 32x32 RGB.

Default:
    patch size     = 8x8
    number tokens  = 16
    local dim      = 32
    interaction    = 1 self-attention block
    FFN dim        = 64
    epochs         = 100
    batch size     = 128

The fixed 10,000-image evaluation split and the 40,000-image training pool
match R1/R2.

This experiment is NOT intended to beat an image autoencoder.
It asks whether a distributed/local construction behaves differently from
the single-global-latent architecture.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_ROOT = Path(
    "/home/xetal/Desktop/latent_model_test/cifar10"
)

DEFAULT_OUT = ROOT / "real_cifar10_local_interaction"

SEED = 12345

IMAGE_SIZE = 32
CHANNELS = 3

PATCH_SIZE = 8
NUM_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS

LOCAL_DIM = 32
FF_DIM = 64
NUM_HEADS = 4
NUM_INTERACTION_LAYERS = 1

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

TEST_SIZE = 10_000

SIZES = [500, 1000, 5000, 10000, 25000, 40000]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CIFARParquetDataset(Dataset):
    """
    Same Hugging Face-style Parquet adapter used by R1/R2.

    Expected:
        img   -> {"bytes": PNG_BYTES, "path": None}
        label -> integer

    Labels are returned for analysis only.
    They are NEVER supplied to the model.
    """

    def __init__(self, parquet_path: Path):
        self.parquet_path = Path(parquet_path)

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10 Parquet file: {self.parquet_path}"
            )

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

        if not isinstance(record, dict) or "bytes" not in record:
            raise RuntimeError(
                f"Unexpected image record at row {idx}: {type(record)}"
            )

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

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the CIFAR-10 Parquet file.\n"
        "Checked:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def fixed_split_indices(
    n: int,
    test_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    all_indices = np.arange(n)

    test_indices = np.sort(
        rng.choice(
            all_indices,
            size=test_size,
            replace=False,
        )
    )

    mask = np.ones(n, dtype=bool)
    mask[test_indices] = False

    train_pool = np.flatnonzero(mask)

    return train_pool, test_indices


def make_subset(
    dataset,
    pool_indices,
    size: int,
    seed: int,
):
    if size > len(pool_indices):
        raise ValueError(
            f"Requested {size:,} examples but only "
            f"{len(pool_indices):,} are available after "
            f"the fixed test holdout."
        )

    rng = np.random.default_rng(seed)

    positions = rng.choice(
        len(pool_indices),
        size=size,
        replace=False,
    )

    selected = pool_indices[
        np.sort(positions)
    ]

    return Subset(
        dataset,
        selected.tolist(),
    )


# ---------------------------------------------------------------------------
# Patch construction
# ---------------------------------------------------------------------------

def patchify(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 3, 32, 32]

    returns:
        [B, 16, 192]

    Each token is one 8x8 RGB patch.
    """
    b, c, h, w = x.shape

    if h != IMAGE_SIZE or w != IMAGE_SIZE:
        raise ValueError(
            f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {h}x{w}"
        )

    p = PATCH_SIZE

    x = x.reshape(
        b,
        c,
        h // p,
        p,
        w // p,
        p,
    )

    x = x.permute(
        0, 2, 4, 1, 3, 5
    ).contiguous()

    return x.reshape(
        b,
        NUM_TOKENS,
        PATCH_DIM,
    )


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    """
    patches: [B, 16, 192]

    returns:
        [B, 3, 32, 32]
    """
    b = patches.shape[0]
    p = PATCH_SIZE
    grid = IMAGE_SIZE // p

    x = patches.reshape(
        b,
        grid,
        grid,
        CHANNELS,
        p,
        p,
    )

    x = x.permute(
        0, 3, 1, 4, 2, 5
    ).contiguous()

    return x.reshape(
        b,
        CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class InteractionBlock(nn.Module):
    """
    One shared relational operation.

    Every local representation uses the SAME interaction machinery.

    No token-specific learned ID embedding is used.
    """

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        heads: int,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x):
        y = self.norm1(x)

        attended, _ = self.attn(
            y,
            y,
            y,
            need_weights=False,
        )

        x = x + attended
        x = x + self.ff(self.norm2(x))

        return x


class LocalInteractionAutoencoder(nn.Module):
    """
    Distributed reconstruction system.

    Architecture:

        image
          |
        16 patches
          |
        shared local encoder
          |
        16 local states
          |
        shared interaction layer
          |
        16 interacting states
          |
        shared local decoder
          |
        16 reconstructed patches
          |
        image

    There is no global 2-D bottleneck.

    The pooled state is only a diagnostic measurement of the system's
    collective state; it is NOT used as the reconstruction bottleneck.
    """

    def __init__(
        self,
        local_dim=LOCAL_DIM,
        ff_dim=FF_DIM,
        heads=NUM_HEADS,
        interaction_layers=NUM_INTERACTION_LAYERS,
    ):
        super().__init__()

        self.local_encoder = nn.Sequential(
            nn.Linear(PATCH_DIM, local_dim),
            nn.GELU(),
            nn.Linear(local_dim, local_dim),
        )

        # Shared positional information.
        # This identifies spatial position, not individual image identity.
        self.position = nn.Parameter(
            torch.zeros(1, NUM_TOKENS, local_dim)
        )

        nn.init.normal_(
            self.position,
            mean=0.0,
            std=0.02,
        )

        self.interaction = nn.ModuleList(
            [
                InteractionBlock(
                    dim=local_dim,
                    ff_dim=ff_dim,
                    heads=heads,
                )
                for _ in range(interaction_layers)
            ]
        )

        self.local_decoder = nn.Sequential(
            nn.Linear(local_dim, local_dim),
            nn.GELU(),
            nn.Linear(local_dim, PATCH_DIM),
            nn.Sigmoid(),
        )

    def encode_local(self, x):
        patches = patchify(x)

        local = self.local_encoder(patches)

        local = local + self.position

        for block in self.interaction:
            local = block(local)

        return local

    def decode_local(self, local):
        patches = self.local_decoder(local)
        return unpatchify(patches)

    def forward(self, x):
        local = self.encode_local(x)
        recon = self.decode_local(local)

        # Diagnostic only.
        global_state = local.mean(dim=1)

        return recon, local, global_state


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one(
    model,
    loader,
    device,
    epochs,
    lr,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    history = []

    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_n = 0

        for x, _ in loader:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            recon, _, _ = model(x)

            loss = F.mse_loss(
                recon,
                x,
            )

            loss.backward()
            optimizer.step()

            n = x.shape[0]

            total_loss += (
                loss.detach().item() * n
            )
            total_n += n

        mean_loss = total_loss / max(total_n, 1)

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
def evaluate(
    model,
    dataset,
    device,
    batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()

    errors = []
    global_states = []
    local_states = []

    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        recon, local, global_state = model(x)

        per_image = (
            (recon - x) ** 2
        ).mean(dim=(1, 2, 3))

        errors.append(
            per_image.cpu().numpy()
        )

        global_states.append(
            global_state.cpu().numpy()
        )

        # Keep only evaluation-size local states.
        local_states.append(
            local.cpu().numpy()
        )

    errors = np.concatenate(errors)
    global_states = np.concatenate(global_states)
    local_states = np.concatenate(local_states)

    return {
        "errors": errors,
        "global_states": global_states,
        "local_states": local_states,
    }


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def covariance_geometry(z):
    """
    Geometry of a point cloud.

    Works for the diagnostic pooled state.
    """
    z = np.asarray(z, dtype=np.float64)

    if z.ndim != 2:
        raise ValueError("Expected [N, D] array.")

    centered = z - z.mean(axis=0, keepdims=True)

    cov = np.cov(
        centered,
        rowvar=False,
    )

    eig = np.linalg.eigvalsh(
        cov
    )

    eig = np.maximum(
        np.asarray(eig),
        0.0,
    )

    total = eig.sum()

    participation = (
        total * total
        / max(
            np.sum(eig * eig),
            1e-12,
        )
    )

    return {
        "mean": z.mean(axis=0).tolist(),
        "std": z.std(axis=0).tolist(),
        "eigenvalues": eig.tolist(),
        "participation_ratio": float(
            participation
        ),
        "trace": float(total),
        "mean_radius": float(
            np.linalg.norm(
                z - z.mean(axis=0),
                axis=1,
            ).mean()
        ),
    }


def local_geometry(local_states):
    """
    Treat all local states as a population and measure their covariance.

    Shape:
        [N, tokens, local_dim]

    This tells us whether the local representation itself is using
    multiple directions instead of collapsing.
    """
    flat = np.asarray(
        local_states,
        dtype=np.float64,
    ).reshape(
        -1,
        local_states.shape[-1],
    )

    # Center locally before covariance.
    flat = flat - flat.mean(
        axis=0,
        keepdims=True,
    )

    cov = np.cov(
        flat,
        rowvar=False,
    )

    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)

    total = eig.sum()

    pr = (
        total * total
        / max(
            np.sum(eig * eig),
            1e-12,
        )
    )

    return {
        "local_dim": int(local_states.shape[-1]),
        "tokens": int(local_states.shape[1]),
        "eigenvalues": eig.tolist(),
        "participation_ratio": float(pr),
        "trace": float(total),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_reconstruction_examples(
    model,
    dataset,
    device,
    size,
    out,
):
    model.eval()

    count = min(8, len(dataset))

    indices = np.linspace(
        0,
        len(dataset) - 1,
        count,
        dtype=int,
    )

    originals = []
    reconstructions = []

    for idx in indices:
        x, _ = dataset[int(idx)]

        recon, _, _ = model(
            x.unsqueeze(0).to(device)
        )

        originals.append(
            x.numpy()
        )

        reconstructions.append(
            recon[0].cpu().numpy()
        )

    fig, axes = plt.subplots(
        2,
        count,
        figsize=(16, 4),
    )

    if count == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for i in range(count):
        axes[0, i].imshow(
            np.transpose(
                originals[i],
                (1, 2, 0),
            )
        )

        axes[1, i].imshow(
            np.transpose(
                reconstructions[i],
                (1, 2, 0),
            )
        )

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("recon")

    fig.suptitle(
        f"R3 local-interaction reconstruction — {size:,}"
    )

    plt.tight_layout()

    plt.savefig(
        out / f"reconstruction_{size}.png",
        dpi=160,
    )

    plt.close()


def save_global_state_plot(
    states,
    size,
    out,
):
    """
    Diagnostic only.

    The architecture has no 2-D bottleneck, so we use PCA to visualize
    the pooled global state. This does NOT affect training.
    """
    z = np.asarray(states)

    if z.shape[1] < 2:
        return

    centered = z - z.mean(
        axis=0,
        keepdims=True,
    )

    _, _, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    pca = centered @ vt[:2].T

    if len(pca) > 5000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(
            len(pca),
            5000,
            replace=False,
        )
        pca = pca[np.sort(idx)]

    plt.figure(figsize=(7, 7))

    plt.scatter(
        pca[:, 0],
        pca[:, 1],
        s=3,
        alpha=0.25,
    )

    plt.xlabel("pooled state PC1")
    plt.ylabel("pooled state PC2")

    plt.title(
        f"R3 pooled global state — {size:,}"
    )

    plt.tight_layout()

    plt.savefig(
        out / f"pooled_state_{size}.png",
        dpi=160,
    )

    plt.close()


def save_token_position_plot(
    local_states,
    size,
    out,
):
    """
    Visualize where each of the 16 local tokens lives after interaction.

    Each token is reduced independently to two PCA coordinates.
    """
    states = np.asarray(local_states)

    # Average each token over evaluation images.
    centers = states.mean(axis=0)

    centered = centers - centers.mean(
        axis=0,
        keepdims=True,
    )

    _, _, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    if vt.shape[0] < 2:
        return

    xy = centered @ vt[:2].T

    plt.figure(figsize=(7, 6))

    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=80,
    )

    for token, (px, py) in enumerate(xy):
        grid_y = token // (IMAGE_SIZE // PATCH_SIZE)
        grid_x = token % (IMAGE_SIZE // PATCH_SIZE)

        plt.annotate(
            f"{token} ({grid_x},{grid_y})",
            (px, py),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("token-state PC1")
    plt.ylabel("token-state PC2")

    plt.title(
        f"R3 local token states — {size:,}"
    )

    plt.tight_layout()

    plt.savefig(
        out / f"token_states_{size}.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        "--sizes",
        type=int,
        nargs="+",
        default=SIZES,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--local-dim",
        type=int,
        default=LOCAL_DIM,
    )

    parser.add_argument(
        "--interaction-layers",
        type=int,
        default=NUM_INTERACTION_LAYERS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 88)
    print("R3 — LOCAL REPRESENTATIONS + SHARED INTERACTION")
    print("=" * 88)
    print(f"Device             : {device}")
    print(f"Data root          : {args.data_root}")
    print(f"Output             : {args.out}")
    print(f"Sizes              : {args.sizes}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Patch size         : {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Local tokens       : {NUM_TOKENS}")
    print(f"Local dimension    : {args.local_dim}")
    print(f"Interaction layers : {args.interaction_layers}")
    print(f"Attention heads    : {NUM_HEADS}")
    print()
    print("IMPORTANT:")
    print("  No global 2-D bottleneck.")
    print("  No class labels.")
    print("  No per-image identity.")
    print("  No token-specific identity embedding.")
    print("  Shared interaction machinery connects local states.")
    print()

    parquet = find_parquet(
        args.data_root
    )

    dataset = CIFARParquetDataset(
        parquet
    )

    train_pool, test_indices = fixed_split_indices(
        len(dataset),
        TEST_SIZE,
        args.seed,
    )

    fixed_test = Subset(
        dataset,
        test_indices.tolist(),
    )

    invalid = [
        s
        for s in args.sizes
        if s > len(train_pool)
    ]

    if invalid:
        raise ValueError(
            "Requested sizes exceed available training pool.\n"
            f"Available: {len(train_pool):,}\n"
            f"Invalid: {invalid}"
        )

    print(
        f"Loaded CIFAR-10 Parquet : {len(dataset):,} images"
    )

    print(
        f"Fixed evaluation set    : {len(fixed_test):,}"
    )

    print(
        f"Training pool           : {len(train_pool):,}"
    )

    print()

    all_results = []

    for size in args.sizes:
        print()
        print("=" * 88)
        print(f"TRAINING SIZE: {size:,}")
        print("=" * 88)

        subset = make_subset(
            dataset,
            train_pool,
            size,
            args.seed,
        )

        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = LocalInteractionAutoencoder(
            local_dim=args.local_dim,
            ff_dim=FF_DIM,
            heads=NUM_HEADS,
            interaction_layers=args.interaction_layers,
        ).to(device)

        parameter_count = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(
            f"Trainable parameters: {parameter_count:,}"
        )

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
            args.batch_size,
        )

        errors = evaluation["errors"]
        pooled = evaluation["global_states"]
        local = evaluation["local_states"]

        pooled_geometry = covariance_geometry(
            pooled
        )

        local_stats = local_geometry(
            local
        )

        result = {
            "experiment": "R3_local_interaction",
            "training_size": size,
            "seed": args.seed,
            "epochs": args.epochs,
            "elapsed_seconds": elapsed,
            "parameter_count": parameter_count,
            "train_mse_final": history[-1],
            "test_mse_mean": float(errors.mean()),
            "test_mse_median": float(np.median(errors)),
            "test_mse_p95": float(np.percentile(errors, 95)),
            "pooled_state_geometry": pooled_geometry,
            "local_state_geometry": local_stats,
            "architecture": {
                "patch_size": PATCH_SIZE,
                "tokens": NUM_TOKENS,
                "patch_dim": PATCH_DIM,
                "local_dim": args.local_dim,
                "ff_dim": FF_DIM,
                "heads": NUM_HEADS,
                "interaction_layers": args.interaction_layers,
            },
        }

        all_results.append(result)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_size": size,
                "seed": args.seed,
                "architecture": result["architecture"],
            },
            args.out
            / f"r3_local_interaction_{size}.pt",
        )

        np.save(
            args.out
            / f"pooled_state_{size}.npy",
            pooled,
        )

        # Local states are small:
        # 10,000 x 16 x 32 ~= 5.1M float32 values ~= 20 MB.
        np.save(
            args.out
            / f"local_states_{size}.npy",
            local,
        )

        with open(
            args.out
            / f"summary_{size}.json",
            "w",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        with open(
            args.out
            / f"train_history_{size}.json",
            "w",
        ) as f:
            json.dump(
                history,
                f,
            )

        save_reconstruction_examples(
            model,
            fixed_test,
            device,
            size,
            args.out,
        )

        save_global_state_plot(
            pooled,
            size,
            args.out,
        )

        save_token_position_plot(
            local,
            size,
            args.out,
        )

        print()
        print(
            f"FINAL | size={size:,} | "
            f"test MSE={errors.mean():.8f} | "
            f"pooled PR={pooled_geometry['participation_ratio']:.4f} | "
            f"local PR={local_stats['participation_ratio']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(
        args.out / "r3_scaling_summary.json",
        "w",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    # Compact scaling plot.
    sizes = np.array(
        [r["training_size"] for r in all_results]
    )

    mse = np.array(
        [r["test_mse_mean"] for r in all_results]
    )

    pooled_pr = np.array(
        [
            r["pooled_state_geometry"][
                "participation_ratio"
            ]
            for r in all_results
        ]
    )

    local_pr = np.array(
        [
            r["local_state_geometry"][
                "participation_ratio"
            ]
            for r in all_results
        ]
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        sizes,
        mse,
        marker="o",
    )
    plt.xscale("log")
    plt.xlabel("training examples")
    plt.ylabel("fixed-test reconstruction MSE")
    plt.title("R3 reconstruction vs dataset size")
    plt.tight_layout()
    plt.savefig(
        args.out / "r3_reconstruction_scaling.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(
        sizes,
        pooled_pr,
        marker="o",
        label="pooled state PR",
    )
    plt.plot(
        sizes,
        local_pr,
        marker="o",
        label="local state PR",
    )
    plt.xscale("log")
    plt.xlabel("training examples")
    plt.ylabel("participation ratio")
    plt.title("R3 representational dimensionality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        args.out / "r3_participation_ratio.png",
        dpi=160,
    )
    plt.close()

    print()
    print("=" * 88)
    print("R3 COMPLETE")
    print("=" * 88)
    print(f"Results: {args.out}")
    print()
    print("This experiment intentionally measures:")
    print("  1. reconstruction")
    print("  2. pooled/global state geometry")
    print("  3. local-state geometry")
    print("  4. movement of local representations with data scale")
    print()
    print("It does NOT assume that a 2-D latent is the correct answer.")


if __name__ == "__main__":
    main()
