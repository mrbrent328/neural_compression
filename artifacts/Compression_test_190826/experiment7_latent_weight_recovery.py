#!/usr/bin/env python3
"""
Experiment 7: Can We Get Weights Back Out of the Latent?

Purpose
-------
Directly test the question:

    "If we put real model weights through the latent representation,
     can the latent give the weights back?"

This uses a REAL GGUF tensor from the local 20B model.

It deliberately separates three questions:

1. Reconstruction:
       weights -> encoder -> latent -> decoder -> weights

2. Held-out reconstruction:
       train the representation on some tiles, then test it on tiles
       it never saw.

3. Functional weight recovery:
       replace the original tensor with the reconstructed tensor and
       measure how much the tensor itself changed.

The script does NOT attempt to modify or rewrite the full 20B model.
It works on ONE selected tensor at a time so the experiment remains
inspectable.

Default model path matches the user's existing setup.

Dependencies:
    torch
    numpy
    gguf

Usage:
    python3 experiment7_latent_weight_recovery.py

Optional:
    python3 experiment7_latent_weight_recovery.py /path/to/model.gguf

The tensor can be selected with TENSOR_NAME below. If it is None, the
script chooses the first sufficiently large floating-point tensor.

Important:
    The autoencoder is TRAINED for this experiment. This is different
    from the earlier GGUFToLatentConverter, whose WeightAutoencoder was
    instantiated and used without training. Therefore this experiment
    tests whether a learned latent can actually contain recoverable
    information about real weights.
"""

import os
import sys
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import gguf


DEFAULT_GGUF = (
    "/home/xetal/.lmstudio/models/"
    "lmstudio-community/gpt-oss-20b-GGUF/"
    "gpt-oss-20b-MXFP4.gguf"
)

TENSOR_NAME = None

TILE_SIZE = 64
LATENT_DIMS = [4, 8, 16, 32]

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

TRAIN_FRACTION = 0.80
SEED = 42


class WeightAutoencoder(nn.Module):
    """
    Tile -> latent -> tile.

    The architecture intentionally stays small and explicit so we can
    inspect exactly what is being asked of the latent.
    """

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


def choose_tensor(reader):
    if TENSOR_NAME is not None:
        for tensor in reader.tensors:
            if str(tensor.name) == TENSOR_NAME:
                return tensor
        raise RuntimeError(
            f"TENSOR_NAME not found: {TENSOR_NAME}"
        )

    candidates = []

    for tensor in reader.tensors:
        arr = tensor.data

        if not np.issubdtype(arr.dtype, np.floating):
            continue

        if arr.ndim < 2:
            continue

        count = int(np.prod(arr.shape))

        if count < TILE_SIZE * TILE_SIZE * 4:
            continue

        candidates.append((count, tensor))

    if not candidates:
        raise RuntimeError(
            "No sufficiently large floating-point tensor found."
        )

    candidates.sort(key=lambda x: x[0])

    return candidates[0][1]


def tensor_to_2d(tensor):
    """
    Convert GGUF tensor into a 2D floating array.

    This follows the same basic shape handling used in the user's existing
    GGUF latent converter: reverse 2D GGUF ordering where necessary, then
    flatten higher-dimensional tensors into rows x columns.
    """
    raw_data = tensor.data
    shape = [int(x) for x in tensor.shape]

    raw = torch.from_numpy(raw_data.copy()).float()

    if raw.ndim == 2 and list(raw.shape) != shape:
        raw = raw.T.contiguous()

    if raw.ndim > 2:
        raw = raw.reshape(-1, raw.shape[-1])

    return raw.numpy(), shape


def extract_tiles(matrix):
    rows, cols = matrix.shape

    pad_r = (TILE_SIZE - rows % TILE_SIZE) % TILE_SIZE
    pad_c = (TILE_SIZE - cols % TILE_SIZE) % TILE_SIZE

    padded = np.pad(
        matrix,
        ((0, pad_r), (0, pad_c)),
        mode="constant",
    )

    tiles = []

    for r in range(0, padded.shape[0], TILE_SIZE):
        for c in range(0, padded.shape[1], TILE_SIZE):
            tile = padded[
                r:r + TILE_SIZE,
                c:c + TILE_SIZE,
            ]
            tiles.append(tile)

    return np.stack(tiles), padded.shape


def restore_tiles(tiles, padded_shape, original_shape):
    rows, cols = padded_shape

    matrix = np.zeros(
        padded_shape,
        dtype=np.float32,
    )

    count = 0

    for r in range(0, rows, TILE_SIZE):
        for c in range(0, cols, TILE_SIZE):
            matrix[
                r:r + TILE_SIZE,
                c:c + TILE_SIZE,
            ] = tiles[count]
            count += 1

    return matrix[:original_shape[0], :original_shape[1]]


def normalize_tiles(tiles):
    """
    Global normalization is part of the shared reconstruction machinery.
    """
    mean = float(tiles.mean())
    std = float(tiles.std()) + 1e-8

    normalized = (tiles - mean) / std

    return normalized.astype(np.float32), mean, std


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

    x = torch.from_numpy(train_tiles).unsqueeze(1)
    loader = torch.utils.data.DataLoader(
        x,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_count = 0

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch)
            total_count += len(batch)

        mean_loss = total_loss / max(total_count, 1)

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"    epoch {epoch + 1:3d}/{EPOCHS} "
                f"| train MSE {mean_loss:.8f}"
            )

    return model


@torch.no_grad()
def encode_decode(model, tiles, mean, std, device):
    model.eval()

    x = torch.from_numpy(tiles).unsqueeze(1).to(device)

    z = model.encode(x)
    y = model.decode(z)

    reconstructed = (
        y.squeeze(1).cpu().numpy() * std + mean
    )

    latent = z.cpu().numpy()

    return latent, reconstructed


def evaluate(original, reconstructed):
    diff = reconstructed - original

    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    denom = float(np.sqrt(np.mean(original ** 2))) + 1e-12
    relative_rmse = rmse / denom

    max_error = float(np.max(np.abs(diff)))

    correlation = float(
        np.corrcoef(
            original.reshape(-1),
            reconstructed.reshape(-1),
        )[0, 1]
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "max_error": max_error,
        "correlation": correlation,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    gguf_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_GGUF
    )

    print("=" * 78)
    print("EXPERIMENT 7: CAN WE GET WEIGHTS BACK OUT OF THE LATENT?")
    print("=" * 78)
    print(f"GGUF: {gguf_path}")
    print()

    if not os.path.exists(gguf_path):
        print("ERROR: GGUF file does not exist.")
        print("Set the path as the first command-line argument.")
        sys.exit(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Latent sizes: {LATENT_DIMS}")
    print()

    print("[1] Opening GGUF and selecting a real weight tensor...")

    reader = gguf.GGUFReader(gguf_path)
    tensor = choose_tensor(reader)

    print(f"Tensor name : {tensor.name}")
    print(f"GGUF shape  : {tensor.shape}")
    print(f"GGUF dtype  : {tensor.data.dtype}")

    matrix, original_shape = tensor_to_2d(tensor)

    print(f"2D shape    : {matrix.shape}")
    print(f"Raw values  : {matrix.size}")
    print()

    print("[2] Extracting weight tiles...")

    tiles, padded_shape = extract_tiles(matrix)

    print(f"Tile count  : {len(tiles)}")
    print(f"Padded shape: {padded_shape}")
    print(f"Tile shape  : {tiles[0].shape}")
    print()

    if len(tiles) < 4:
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

    train_idx = indices[:split]
    test_idx = indices[split:]

    train_tiles = tiles[train_idx]
    test_tiles = tiles[test_idx]

    train_norm, mean, std = normalize_tiles(train_tiles)

    # Test tiles are normalized using ONLY training statistics.
    test_norm = (
        (test_tiles - mean) / std
    ).astype(np.float32)

    print(f"Training tiles: {len(train_tiles)}")
    print(f"Held-out tiles: {len(test_tiles)}")
    print()

    print("[3] Original weight checksum...")
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

        train_original = train_tiles
        test_original = test_tiles

        train_metrics = evaluate(
            train_original,
            train_recon,
        )

        test_metrics = evaluate(
            test_original,
            test_recon,
        )

        parameter_bytes = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )

        latent_bytes_per_tile = (
            latent_dim * 4
        )

        original_bytes_per_tile = (
            TILE_SIZE * TILE_SIZE * 4
        )

        latent_ratio = (
            original_bytes_per_tile
            / latent_bytes_per_tile
        )

        print()
        print("TRAINING TILE RECONSTRUCTION")
        print(
            f"  RMSE        : "
            f"{train_metrics['rmse']:.8f}"
        )
        print(
            f"  Relative RMSE: "
            f"{train_metrics['relative_rmse']:.6f}"
        )
        print(
            f"  Correlation : "
            f"{train_metrics['correlation']:.6f}"
        )

        print()
        print("HELD-OUT TILE RECONSTRUCTION")
        print(
            f"  RMSE        : "
            f"{test_metrics['rmse']:.8f}"
        )
        print(
            f"  Relative RMSE: "
            f"{test_metrics['relative_rmse']:.6f}"
        )
        print(
            f"  Max error   : "
            f"{test_metrics['max_error']:.8f}"
        )
        print(
            f"  Correlation : "
            f"{test_metrics['correlation']:.6f}"
        )

        print()
        print("LATENT SIZE")
        print(
            f"  Original tile: "
            f"{original_bytes_per_tile} bytes"
        )
        print(
            f"  Latent tile  : "
            f"{latent_bytes_per_tile} bytes"
        )
        print(
            f"  Latent ratio : "
            f"{latent_ratio:.2f}x"
        )

        print()
        print("SHARED DECODER")
        print(
            f"  Parameters: "
            f"{parameter_bytes / 1024**2:.2f} MiB"
        )

        elapsed = time.time() - start

        print()
        print(f"Experiment time: {elapsed:.2f}s")

        results.append(
            {
                "latent_dim": latent_dim,
                "train": train_metrics,
                "test": test_metrics,
                "parameter_bytes": parameter_bytes,
                "original_bytes_per_tile": original_bytes_per_tile,
                "latent_bytes_per_tile": latent_bytes_per_tile,
                "latent_ratio": latent_ratio,
            }
        )

    # ---------------------------------------------------------------
    # Final experiment: reconstruct the whole tensor using the best
    # latent size by held-out RMSE.
    # ---------------------------------------------------------------
    best = min(
        results,
        key=lambda r: r["test"]["rmse"],
    )

    best_latent = best["latent_dim"]

    print()
    print("=" * 78)
    print("BEST LATENT RECONSTRUCTION")
    print("=" * 78)
    print(
        f"Best latent dimension by held-out RMSE: "
        f"{best_latent}"
    )
    print(
        f"Held-out RMSE: "
        f"{best['test']['rmse']:.8f}"
    )
    print(
        f"Held-out correlation: "
        f"{best['test']['correlation']:.6f}"
    )

    # Retrain best model so we can reconstruct all tiles.
    print()
    print("[4] Rebuilding best model and reconstructing ALL tiles...")

    best_rng = np.random.default_rng(
        SEED + best_latent
    )

    best_model = train_autoencoder(
        train_norm,
        best_latent,
        device,
    )

    all_norm = (
        (tiles - mean) / std
    ).astype(np.float32)

    all_latent, all_reconstructed = encode_decode(
        best_model,
        all_norm,
        mean,
        std,
        device,
    )

    reconstructed_matrix = restore_tiles(
        all_reconstructed,
        padded_shape,
        matrix.shape,
    )

    whole_metrics = evaluate(
        matrix,
        reconstructed_matrix,
    )

    print()
    print("WHOLE-TENSOR RECONSTRUCTION")
    print(f"  RMSE         : {whole_metrics['rmse']:.8f}")
    print(
        f"  Relative RMSE: "
        f"{whole_metrics['relative_rmse']:.6f}"
    )
    print(
        f"  Correlation  : "
        f"{whole_metrics['correlation']:.6f}"
    )
    print(
        f"  Max error    : "
        f"{whole_metrics['max_error']:.8f}"
    )

    print()
    print("ORIGINAL TENSOR HASH")
    print(f"  {sha256_array(matrix)}")

    print()
    print("RECONSTRUCTED TENSOR HASH")
    print(f"  {sha256_array(reconstructed_matrix)}")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(
        "This test answers a narrow question:"
    )
    print(
        "Can a learned latent representation of REAL model weights "
        "retain enough information to reconstruct those weights?"
    )
    print()
    print(
        "The held-out tile test is the important anti-memorization check."
    )
    print(
        "If held-out tiles reconstruct well, the latent learned something "
        "about the weight structure rather than simply memorizing the "
        "training tiles."
    )
    print()
    print(
        "This does NOT yet prove that the latent can reconstruct a useful "
        "working model. That would be the next experiment: replace a real "
        "tensor in a model with its reconstructed version and compare model "
        "outputs/logits."
    )


if __name__ == "__main__":
    main()
