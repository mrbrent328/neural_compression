#!/usr/bin/env python3
"""
Experiment 7b: REAL WEIGHT -> LATENT -> WEIGHT (SAFE CUDA VERSION)

This is the corrected version of Experiment 7.

Changes:
- Selects a substantial 2-D WEIGHT matrix instead of a bias tensor.
- Explicitly copies GGUF/NumPy data into ordinary owned float32 memory.
- Uses CPU by default to avoid CUDA/PyTorch/GGUF memory interaction.
- CUDA can be enabled with --cuda after the CPU version works.
- Keeps the experiment deliberately small enough to diagnose safely.
- Uses held-out tiles to test whether the latent learned weight structure
  rather than memorizing the training tiles.

Usage:
    python3 experiment7b_latent_weight_recovery.py /path/to/model.gguf

Optional:
    python3 experiment7b_latent_weight_recovery.py /path/to/model.gguf --cuda

Optional tensor selection:
    python3 experiment7b_latent_weight_recovery.py /path/to/model.gguf --tensor NAME

If no tensor is supplied, the script chooses the largest suitable 2-D
floating-point tensor up to a reasonable size.

IMPORTANT:
The first run should be CPU. Once that works, try --cuda if desired.
"""

import os
import sys
import gc
import argparse
import hashlib
import time

import numpy as np
import torch
import torch.nn as nn
import gguf


DEFAULT_GGUF = (
    "/home/xetal/.lmstudio/models/"
    "lmstudio-community/gpt-oss-20b-GGUF/"
    "gpt-oss-20b-MXFP4.gguf"
)

SEED = 42
TILE_SIZE = 64
LATENT_DIMS = [4, 8, 16, 32]

EPOCHS = 25
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

TRAIN_FRACTION = 0.80

# Keep the first experiment bounded.
# Set to None to use every suitable tile.
MAX_TILES = 256


class WeightAutoencoder(nn.Module):
    def __init__(self, tile_size, latent_dim):
        super().__init__()

        n = tile_size * tile_size

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, n),
            nn.Unflatten(1, (1, tile_size, tile_size)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


def sha256_array(x):
    return hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()


def choose_tensor(reader, requested=None):
    if requested:
        for tensor in reader.tensors:
            if str(tensor.name) == requested:
                arr = tensor.data

                if not np.issubdtype(arr.dtype, np.floating):
                    raise RuntimeError(
                        f"Requested tensor is not floating point: "
                        f"{tensor.name}"
                    )

                if arr.ndim < 2:
                    raise RuntimeError(
                        f"Requested tensor is not at least 2-D: "
                        f"{tensor.name} shape={tensor.shape}"
                    )

                return tensor

        raise RuntimeError(
            f"Tensor not found: {requested}"
        )

    candidates = []

    for tensor in reader.tensors:
        arr = tensor.data

        if not np.issubdtype(arr.dtype, np.floating):
            continue

        # We specifically want ordinary weight matrices for this test.
        if arr.ndim != 2:
            continue

        shape = [int(x) for x in tensor.shape]

        if min(shape) < TILE_SIZE:
            continue

        count = int(np.prod(shape))

        # Avoid tiny special tensors/biases.
        if count < TILE_SIZE * TILE_SIZE * 16:
            continue

        candidates.append((count, tensor))

    if not candidates:
        raise RuntimeError(
            "No suitable 2-D floating-point weight matrix found."
        )

    # Largest suitable matrix gives us a meaningful weight-structure test.
    candidates.sort(key=lambda x: x[0], reverse=True)

    return candidates[0][1]


def tensor_to_matrix(tensor):
    """
    Make an independent float32 CPU copy.

    We intentionally sever any possible relationship with GGUF-backed memory
    before PyTorch sees the data.
    """
    raw = np.array(
        tensor.data,
        dtype=np.float32,
        copy=True,
        order="C",
    )

    shape = [int(x) for x in tensor.shape]

    matrix = torch.from_numpy(raw).clone().contiguous().numpy()

    if matrix.ndim == 2 and list(matrix.shape) != shape:
        matrix = matrix.T.copy()

    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    return matrix, shape


def extract_tiles(matrix, max_tiles=None):
    rows, cols = matrix.shape

    pad_r = (TILE_SIZE - rows % TILE_SIZE) % TILE_SIZE
    pad_c = (TILE_SIZE - cols % TILE_SIZE) % TILE_SIZE

    padded = np.pad(
        matrix,
        ((0, pad_r), (0, pad_c)),
        mode="constant",
    ).astype(np.float32, copy=True)

    tiles = []

    for r in range(0, padded.shape[0], TILE_SIZE):
        for c in range(0, padded.shape[1], TILE_SIZE):
            tiles.append(
                padded[
                    r:r + TILE_SIZE,
                    c:c + TILE_SIZE,
                ].copy()
            )

            if max_tiles is not None and len(tiles) >= max_tiles:
                return (
                    np.stack(tiles).astype(np.float32, copy=True),
                    padded.shape,
                )

    return (
        np.stack(tiles).astype(np.float32, copy=True),
        padded.shape,
    )


def normalize_tiles(tiles):
    mean = np.float32(tiles.mean())
    std = np.float32(tiles.std() + 1e-8)

    normalized = (
        (tiles.astype(np.float32) - mean) / std
    ).astype(np.float32)

    return normalized, float(mean), float(std)


def train_autoencoder(train_tiles, latent_dim, device):
    model = WeightAutoencoder(
        TILE_SIZE,
        latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = nn.MSELoss()

    x = torch.from_numpy(
        np.ascontiguousarray(train_tiles)
    ).unsqueeze(1).contiguous()

    loader = torch.utils.data.DataLoader(
        x,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_count = 0

        for batch in loader:
            batch = batch.to(
                device,
                non_blocking=False,
            )

            optimizer.zero_grad(set_to_none=True)

            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)

            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.item()) * len(batch)
            )
            total_count += len(batch)

        mean_loss = (
            total_loss / max(total_count, 1)
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"    epoch {epoch + 1:3d}/{EPOCHS}"
                f" | train MSE {mean_loss:.8f}"
            )

    return model


@torch.no_grad()
def encode_decode(model, tiles, mean, std, device):
    model.eval()

    x = torch.from_numpy(
        np.ascontiguousarray(tiles)
    ).unsqueeze(1).contiguous()

    x = x.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    z = model.encode(x)
    y = model.decode(z)

    if device.type == "cuda":
        torch.cuda.synchronize()

    reconstructed = (
        y.squeeze(1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    reconstructed = (
        reconstructed * np.float32(std)
        + np.float32(mean)
    ).astype(np.float32)

    latent = (
        z.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return latent, reconstructed


def evaluate(original, reconstructed):
    diff = (
        reconstructed.astype(np.float64)
        - original.astype(np.float64)
    )

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    denom = (
        float(np.sqrt(np.mean(
            original.astype(np.float64) ** 2
        )))
        + 1e-12
    )

    relative_rmse = rmse / denom

    max_error = float(np.max(np.abs(diff)))

    corr = np.corrcoef(
        original.reshape(-1),
        reconstructed.reshape(-1),
    )[0, 1]

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "max_error": max_error,
        "correlation": float(corr),
    }


def restore_tiles(
    tiles,
    padded_shape,
    original_shape,
):
    rows, cols = padded_shape

    matrix = np.zeros(
        padded_shape,
        dtype=np.float32,
    )

    count = 0

    for r in range(0, rows, TILE_SIZE):
        for c in range(0, cols, TILE_SIZE):
            if count >= len(tiles):
                break

            matrix[
                r:r + TILE_SIZE,
                c:c + TILE_SIZE,
            ] = tiles[count]

            count += 1

    return matrix[
        :original_shape[0],
        :original_shape[1],
    ].copy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "gguf",
        nargs="?",
        default=DEFAULT_GGUF,
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA. CPU is the recommended first run.",
    )

    parser.add_argument(
        "--tensor",
        default=None,
        help="Exact GGUF tensor name.",
    )

    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="Use all suitable tiles instead of MAX_TILES.",
    )

    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 78)
    print("EXPERIMENT 7b: CAN WE GET WEIGHTS BACK OUT OF THE LATENT?")
    print("=" * 78)
    print(f"GGUF: {args.gguf}")
    print()

    if not os.path.exists(args.gguf):
        raise FileNotFoundError(args.gguf)

    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--cuda requested, but torch.cuda.is_available() is False."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Latent sizes: {LATENT_DIMS}")
    print()

    print("[1] Opening GGUF...")

    reader = gguf.GGUFReader(args.gguf)

    tensor = choose_tensor(
        reader,
        args.tensor,
    )

    print(f"Selected tensor: {tensor.name}")
    print(f"GGUF shape     : {tensor.shape}")
    print(f"GGUF dtype     : {tensor.data.dtype}")
    print()

    print("[2] Making an independent float32 CPU copy...")

    matrix, original_shape = tensor_to_matrix(tensor)

    print(f"2D shape       : {matrix.shape}")
    print(f"Raw values     : {matrix.size}")
    print(
        f"Raw memory     : "
        f"{matrix.nbytes / 1024**2:.3f} MiB"
    )
    print()

    # Free reader before training. This is especially useful when another
    # 20B model is already resident in RAM.
    del reader
    del tensor
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("[3] Extracting weight tiles...")

    max_tiles = None if args.all_tiles else MAX_TILES

    tiles, padded_shape = extract_tiles(
        matrix,
        max_tiles=max_tiles,
    )

    print(f"Tile count     : {len(tiles)}")
    print(f"Padded shape   : {padded_shape}")
    print(f"Tile shape     : {tiles[0].shape}")
    print()

    if len(tiles) < 5:
        raise RuntimeError(
            "Not enough tiles for train/test split."
        )

    rng = np.random.default_rng(SEED)

    indices = np.arange(len(tiles))
    rng.shuffle(indices)

    split = max(
        1,
        int(len(indices) * TRAIN_FRACTION),
    )

    if split >= len(indices):
        split = len(indices) - 1

    train_idx = indices[:split]
    test_idx = indices[split:]

    train_tiles = tiles[train_idx].copy()
    test_tiles = tiles[test_idx].copy()

    train_norm, mean, std = normalize_tiles(
        train_tiles
    )

    test_norm = (
        (test_tiles - np.float32(mean))
        / np.float32(std)
    ).astype(np.float32)

    print(f"Training tiles: {len(train_tiles)}")
    print(f"Held-out tiles: {len(test_tiles)}")
    print()

    print("[4] Original tensor checksum...")
    print(sha256_array(matrix))
    print()

    results = []

    for latent_dim in LATENT_DIMS:
        print()
        print("-" * 78)
        print(f"LATENT DIMENSION = {latent_dim}")
        print("-" * 78)

        start = time.time()

        model = train_autoencoder(
            train_norm,
            latent_dim,
            device,
        )

        train_latent, train_recon = encode_decode(
            model,
            train_norm,
            mean,
            std,
            device,
        )

        test_latent, test_recon = encode_decode(
            model,
            test_norm,
            mean,
            std,
            device,
        )

        train_metrics = evaluate(
            train_tiles,
            train_recon,
        )

        test_metrics = evaluate(
            test_tiles,
            test_recon,
        )

        parameter_bytes = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )

        original_tile_bytes = (
            TILE_SIZE * TILE_SIZE * 4
        )

        latent_tile_bytes = (
            latent_dim * 4
        )

        ratio = (
            original_tile_bytes
            / latent_tile_bytes
        )

        print()
        print("TRAINING TILES")
        print(
            f"  RMSE         : "
            f"{train_metrics['rmse']:.8f}"
        )
        print(
            f"  Correlation  : "
            f"{train_metrics['correlation']:.6f}"
        )

        print()
        print("HELD-OUT TILES")
        print(
            f"  RMSE         : "
            f"{test_metrics['rmse']:.8f}"
        )
        print(
            f"  Relative RMSE: "
            f"{test_metrics['relative_rmse']:.6f}"
        )
        print(
            f"  Max error    : "
            f"{test_metrics['max_error']:.8f}"
        )
        print(
            f"  Correlation  : "
            f"{test_metrics['correlation']:.6f}"
        )

        print()
        print("LATENT")
        print(
            f"  Original tile: "
            f"{original_tile_bytes} bytes"
        )
        print(
            f"  Latent tile  : "
            f"{latent_tile_bytes} bytes"
        )
        print(
            f"  Ratio        : "
            f"{ratio:.2f}x"
        )

        print()
        print("SHARED DECODER")
        print(
            f"  Parameters   : "
            f"{parameter_bytes / 1024**2:.2f} MiB"
        )

        elapsed = time.time() - start
        print(f"Time           : {elapsed:.2f}s")

        results.append(
            {
                "latent": latent_dim,
                "test": test_metrics,
                "train": train_metrics,
                "ratio": ratio,
            }
        )

        # Drop this model before constructing the next one.
        del model
        del train_latent
        del test_latent
        del train_recon
        del test_recon

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print()
    print("=" * 78)
    print("RESULT SUMMARY")
    print("=" * 78)

    print(
        f"{'Latent':>8} | "
        f"{'Held RMSE':>12} | "
        f"{'Rel RMSE':>10} | "
        f"{'Correlation':>12} | "
        f"{'Ratio':>8}"
    )

    print("-" * 78)

    for r in results:
        m = r["test"]

        print(
            f"{r['latent']:>8} | "
            f"{m['rmse']:>12.6f} | "
            f"{m['relative_rmse']:>10.6f} | "
            f"{m['correlation']:>12.6f} | "
            f"{r['ratio']:>7.2f}x"
        )

    best = min(
        results,
        key=lambda x: x["test"]["rmse"],
    )

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    print(
        "This is the important result:"
    )
    print(
        "If HELD-OUT correlation remains high, the latent is recovering "
        "structure from REAL model weights that it was not trained on."
    )
    print()
    print(
        "If training correlation is high but held-out correlation collapses, "
        "the latent is memorizing."
    )
    print()
    print(
        "If both are high at a substantial compression ratio, we have "
        "evidence that the learned latent contains recoverable information "
        "about real model weights."
    )
    print()
    print(
        "That still does NOT prove the latent can reconstruct a functioning "
        "20B model. The next test after a successful result would be to "
        "replace one actual model tensor with its reconstructed version and "
        "compare logits/outputs."
    )
    print()
    print(
        f"Best held-out latent: {best['latent']}"
    )
    print(
        f"Best held-out correlation: "
        f"{best['test']['correlation']:.6f}"
    )


if __name__ == "__main__":
    main()
