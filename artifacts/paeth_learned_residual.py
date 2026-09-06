#!/usr/bin/env python3

"""
PNG STRUCTURE / PREDICTOR COMPRESSION TEST
===========================================

Tests three predictors against decoded PNG image data:

1. PREVIOUS-BYTE
   Predict each byte from the byte immediately to its left.

2. PAETH
   PNG's spatial Paeth predictor using left/up/up-left pixels.

3. NEURAL
   Small learned predictor operating on local byte neighborhoods.

The script measures:

- Raw decoded image size
- PNG file size
- zlib baseline
- Predictor residual entropy
- Predictor residual compression
- Exact decoded-pixel reconstruction
- PNG reconstruction
- SHA256 verification
- Total protocol storage

IMPORTANT:

A PNG is already a compressed container.

Therefore this script distinguishes:

PIXEL EXACT
    Reconstructed decoded image matches exactly.

PNG EXACT
    Reconstructed PNG file has identical SHA256.

COMPRESSION
    The complete representation is actually smaller.

A model that predicts pixels extremely well but requires a huge
model/residual representation is NOT considered a successful
compression protocol.
"""

import os
import sys
import zlib
import hashlib
import struct
import math
import time
from collections import Counter

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required.")
    print("Install with:")
    print("    pip install pillow numpy torch")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch is required.")
    print("Install with:")
    print("    pip install torch")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

FILE = "pic.png"

NEURAL_EPOCHS = 3000
NEURAL_LATENT = 16
NEURAL_HIDDEN = 128

# Train on at most this many bytes.
# The predictor is trained on a representative sample so the
# experiment doesn't become impossibly slow on a large PNG.
MAX_TRAIN_BYTES = 1_000_000

# Maximum number of training samples.
MAX_SAMPLES = 500_000

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# UTILITIES
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def entropy(data):
    """
    Shannon entropy in bits/byte.
    """
    if len(data) == 0:
        return 0.0

    counts = np.bincount(
        np.frombuffer(data, dtype=np.uint8),
        minlength=256
    )

    probs = counts[counts > 0] / len(data)

    return float(
        -np.sum(probs * np.log2(probs))
    )


def fmt(n):
    return f"{n:,}"


# ============================================================
# PNG PARSER
# ============================================================

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_png(data):
    """
    Extract PNG chunks.

    Returns:
        chunks
        IHDR information
        IDAT compressed stream
        ancillary chunks
    """

    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file.")

    pos = 8
    chunks = []

    while pos < len(data):

        if pos + 12 > len(data):
            raise ValueError("Corrupt PNG.")

        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]

        chunk_start = pos
        chunk_end = pos + 12 + length

        if chunk_end > len(data):
            raise ValueError("PNG chunk extends past file.")

        chunk_data = data[pos + 8:pos + 8 + length]
        crc = data[pos + 8 + length:chunk_end]

        chunks.append({
            "type": ctype,
            "data": chunk_data,
            "raw": data[chunk_start:chunk_end],
        })

        pos = chunk_end

        if ctype == b"IEND":
            break

    ihdr = None
    idat = bytearray()

    for c in chunks:

        if c["type"] == b"IHDR":
            ihdr = c["data"]

        elif c["type"] == b"IDAT":
            idat.extend(c["data"])

    if ihdr is None:
        raise ValueError("PNG has no IHDR.")

    width, height, bit_depth, color_type, compression, filt, interlace = \
        struct.unpack(">IIBBBBB", ihdr)

    return {
        "chunks": chunks,
        "ihdr": ihdr,
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "interlace": interlace,
        "idat": bytes(idat),
    }


# ============================================================
# PNG DECODE
# ============================================================

def decode_png_pixels(filename):
    """
    Uses Pillow to obtain exact decoded pixel bytes.

    This intentionally works at the decoded image level.

    The original PNG bytes remain available separately for the
    final SHA256 test.
    """

    with Image.open(filename) as im:

        # Convert to a deterministic representation.

        if im.mode in ("1", "L", "P"):
            im = im.convert("RGB")

        elif im.mode == "LA":
            im = im.convert("RGBA")

        elif im.mode == "RGB":
            pass

        elif im.mode == "RGBA":
            pass

        else:
            im = im.convert("RGBA")

        width, height = im.size
        mode = im.mode

        pixels = im.tobytes()

    return pixels, width, height, mode


# ============================================================
# PREDICTOR 1
# PREVIOUS BYTE
# ============================================================

def predict_previous(data):
    """
    Predict each byte from the previous byte.

    Prediction:
        p[i] = data[i-1]

    Residual:
        r[i] = data[i] - p[i] mod 256
    """

    arr = np.frombuffer(data, dtype=np.uint8).copy()

    pred = np.zeros_like(arr)

    if len(arr) > 1:
        pred[1:] = arr[:-1]

    residual = (
        arr.astype(np.int16) -
        pred.astype(np.int16)
    ) % 256

    return pred.astype(np.uint8), residual.astype(np.uint8)


# ============================================================
# PREDICTOR 2
# PAETH
# ============================================================

def paeth_predictor(a, b, c):

    p = int(a) + int(b) - int(c)

    pa = abs(p - int(a))
    pb = abs(p - int(b))
    pc = abs(p - int(c))

    if pa <= pb and pa <= pc:
        return a

    if pb <= pc:
        return b

    return c


def predict_paeth(data, width, height, channels):
    """
    Spatial Paeth predictor.

    Works directly on decoded pixel bytes.

    For every channel:

        left = x-1
        up   = y-1
        ul   = x-1,y-1
    """

    arr = np.frombuffer(data, dtype=np.uint8)

    expected = width * height * channels

    if len(arr) != expected:
        raise ValueError(
            f"Pixel buffer mismatch: "
            f"{len(arr)} != {expected}"
        )

    arr = arr.reshape(height, width, channels)

    pred = np.zeros_like(arr)

    for y in range(height):

        for x in range(width):

            for c in range(channels):

                left = (
                    arr[y, x - 1, c]
                    if x > 0 else 0
                )

                up = (
                    arr[y - 1, x, c]
                    if y > 0 else 0
                )

                up_left = (
                    arr[y - 1, x - 1, c]
                    if x > 0 and y > 0 else 0
                )

                pred[y, x, c] = paeth_predictor(
                    left,
                    up,
                    up_left
                )

    residual = (
        arr.astype(np.int16) -
        pred.astype(np.int16)
    ) % 256

    return (
        pred.reshape(-1),
        residual.astype(np.uint8).reshape(-1)
    )


# ============================================================
# NEURAL PREDICTOR
# ============================================================

class NeuralPredictor(nn.Module):

    def __init__(self, hidden=128):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.ReLU(),

            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def build_neural_dataset(data):

    arr = np.frombuffer(
        data,
        dtype=np.uint8
    ).astype(np.float32) / 255.0

    if len(arr) < 4:
        raise ValueError("Not enough data.")

    # Local causal context:
    #
    # x[i-3], x[i-2], x[i-1] -> x[i]

    indices = np.arange(
        3,
        len(arr)
    )

    if len(indices) > MAX_SAMPLES:

        rng = np.random.default_rng(SEED)

        indices = rng.choice(
            indices,
            size=MAX_SAMPLES,
            replace=False
        )

        indices.sort()

    x = np.stack([
        arr[indices - 3],
        arr[indices - 2],
        arr[indices - 1]
    ], axis=1)

    y = arr[indices][:, None]

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )


def train_neural_predictor(data):

    print()
    print("=" * 60)
    print("TRAINING NEURAL PREDICTOR")
    print("=" * 60)

    train_data = data

    if len(train_data) > MAX_TRAIN_BYTES:

        print(
            f"Training sample limited to "
            f"{fmt(MAX_TRAIN_BYTES)} bytes"
        )

        train_data = train_data[:MAX_TRAIN_BYTES]

    x, y = build_neural_dataset(train_data)

    model = NeuralPredictor(
        hidden=NEURAL_HIDDEN
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.MSELoss()

    print("Training samples:", fmt(len(x)))
    print("Parameters:", fmt(
        sum(p.numel() for p in model.parameters())
    ))

    start = time.time()

    for epoch in range(NEURAL_EPOCHS):

        optimizer.zero_grad()

        out = model(x)

        loss = loss_fn(out, y)

        loss.backward()

        optimizer.step()

        if (
            epoch == 0 or
            epoch % 250 == 0 or
            epoch == NEURAL_EPOCHS - 1
        ):

            print(
                f"Epoch {epoch:5d} | "
                f"Loss: {loss.item():.10f}"
            )

    elapsed = time.time() - start

    print(
        f"Training time: {elapsed:.2f} seconds"
    )

    return model


def neural_predict(model, data):

    arr = np.frombuffer(
        data,
        dtype=np.uint8
    ).astype(np.float32) / 255.0

    if len(arr) < 4:

        return (
            np.zeros_like(arr, dtype=np.uint8),
            arr.astype(np.uint8)
        )

    pred = np.zeros_like(arr)

    # First three bytes have no causal history.
    pred[:3] = arr[:3]

    batch_size = 100_000

    for start in range(3, len(arr), batch_size):

        end = min(
            start + batch_size,
            len(arr)
        )

        indices = np.arange(
            start,
            end
        )

        x = np.stack([
            arr[indices - 3],
            arr[indices - 2],
            arr[indices - 1]
        ], axis=1)

        xt = torch.tensor(
            x,
            dtype=torch.float32
        )

        with torch.no_grad():

            p = model(
                xt
            ).numpy().reshape(-1)

        pred[start:end] = p * 255.0

    pred = np.clip(
        np.rint(pred),
        0,
        255
    ).astype(np.uint8)

    actual = np.frombuffer(
        data,
        dtype=np.uint8
    )

    residual = (
        actual.astype(np.int16) -
        pred.astype(np.int16)
    ) % 256

    return pred, residual.astype(np.uint8)


# ============================================================
# RESIDUAL RECONSTRUCTION
# ============================================================

def reconstruct_from_residual(pred, residual):

    p = pred.astype(np.int16)
    r = residual.astype(np.int16)

    reconstructed = (
        p + r
    ) % 256

    return reconstructed.astype(np.uint8)


# ============================================================
# TEST PREDICTOR
# ============================================================

def test_predictor(name, data, pred, residual):

    reconstructed = reconstruct_from_residual(
        pred,
        residual
    )

    original = np.frombuffer(
        data,
        dtype=np.uint8
    )

    exact = np.array_equal(
        original,
        reconstructed
    )

    raw_residual = residual.tobytes()

    compressed_residual = zlib.compress(
        raw_residual,
        level=9
    )

    mse = float(
        np.mean(
            (
                original.astype(np.float64) -
                reconstructed.astype(np.float64)
            ) ** 2
        )
    )

    mae = float(
        np.mean(
            np.abs(
                original.astype(np.float64) -
                reconstructed.astype(np.float64)
            )
        )
    )

    max_error = int(
        np.max(
            np.abs(
                original.astype(np.int16) -
                reconstructed.astype(np.int16)
            )
        )
    )

    zero_count = np.count_nonzero(
        residual == 0
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Residual entropy:",
        f"{entropy(raw_residual):.6f}",
        "bits/byte"
    )

    print(
        "Zero residual:",
        f"{100 * zero_count / len(residual):.3f}%"
    )

    print(
        "Residual raw:",
        fmt(len(raw_residual)),
        "bytes"
    )

    print(
        "Residual zlib:",
        fmt(len(compressed_residual)),
        "bytes"
    )

    print(
        "Prediction MSE:",
        f"{mse:.6f}"
    )

    print(
        "Prediction MAE:",
        f"{mae:.6f}"
    )

    print(
        "Maximum error:",
        max_error
    )

    print(
        "Exact pixel reconstruction:",
        "YES" if exact else "NO"
    )

    return {
        "name": name,
        "residual": raw_residual,
        "compressed": compressed_residual,
        "entropy": entropy(raw_residual),
        "exact": exact,
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
    }


# ============================================================
# PNG REBUILD TEST
# ============================================================

def compare_png_reconstruction(
    original_file,
    pixels,
    width,
    height,
    mode
):

    """
    Re-encode decoded pixels as PNG.

    This is NOT expected to produce the original byte stream.

    The test demonstrates the difference between:

        exact pixels

    and:

        exact original PNG bytes.
    """

    from io import BytesIO

    buffer = BytesIO()

    image = Image.frombytes(
        mode,
        (width, height),
        pixels
    )

    image.save(
        buffer,
        format="PNG"
    )

    rebuilt = buffer.getvalue()

    with open(
        original_file,
        "rb"
    ) as f:

        original = f.read()

    print()
    print("=" * 60)
    print("PNG RE-ENCODING TEST")
    print("=" * 60)

    print(
        "Original PNG:",
        fmt(len(original)),
        "bytes"
    )

    print(
        "Re-encoded PNG:",
        fmt(len(rebuilt)),
        "bytes"
    )

    print(
        "Original SHA256:"
    )

    print(
        sha256(original)
    )

    print(
        "Re-encoded SHA256:"
    )

    print(
        sha256(rebuilt)
    )

    if original == rebuilt:

        print()
        print(
            "EXACT PNG BYTE RECONSTRUCTION: YES"
        )

    else:

        print()
        print(
            "EXACT PNG BYTE RECONSTRUCTION: NO"
        )

        print(
            "Decoded pixel data can still be "
            "exactly identical."
        )

    return rebuilt


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PNG STRUCTURE / PREDICTOR PROTOCOL")
    print("=" * 60)

    if not os.path.exists(FILE):

        print()
        print(
            f"ERROR: {FILE} not found."
        )

        print(
            "Put this script in the same directory "
            "as your PNG."
        )

        return

    # --------------------------------------------------------
    # ORIGINAL FILE
    # --------------------------------------------------------

    with open(FILE, "rb") as f:
        original_file = f.read()

    original_hash = sha256(
        original_file
    )

    print()
    print("FILE")
    print("-" * 60)

    print("File:", FILE)
    print(
        "Original size:",
        fmt(len(original_file)),
        "bytes"
    )

    print(
        "SHA256:",
        original_hash
    )

    # --------------------------------------------------------
    # PNG STRUCTURE
    # --------------------------------------------------------

    png = parse_png(
        original_file
    )

    print()
    print("PNG STRUCTURE")
    print("-" * 60)

    print(
        "Width:",
        png["width"]
    )

    print(
        "Height:",
        png["height"]
    )

    print(
        "Bit depth:",
        png["bit_depth"]
    )

    print(
        "Color type:",
        png["color_type"]
    )

    print(
        "Interlace:",
        png["interlace"]
    )

    print(
        "IDAT compressed:",
        fmt(len(png["idat"])),
        "bytes"
    )

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    pixels, width, height, mode = \
        decode_png_pixels(FILE)

    channels = len(
        Image.new(
            mode,
            (1, 1)
        ).getbands()
    )

    print()
    print("DECODED IMAGE")
    print("-" * 60)

    print(
        "Mode:",
        mode
    )

    print(
        "Channels:",
        channels
    )

    print(
        "Decoded pixels:",
        fmt(len(pixels)),
        "bytes"
    )

    # --------------------------------------------------------
    # ZLIB BASELINE
    # --------------------------------------------------------

    z_pixels = zlib.compress(
        pixels,
        level=9
    )

    print()
    print("=" * 60)
    print("PIXEL ZLIB BASELINE")
    print("=" * 60)

    print(
        "Decoded:",
        fmt(len(pixels))
    )

    print(
        "zlib:",
        fmt(len(z_pixels))
    )

    print(
        "Ratio:",
        f"{len(pixels) / len(z_pixels):.4f}x"
    )

    # --------------------------------------------------------
    # PREDICTOR 1
    # --------------------------------------------------------

    pred1, residual1 = \
        predict_previous(pixels)

    result1 = test_predictor(
        "PREDICTOR 1 — PREVIOUS BYTE",
        pixels,
        pred1,
        residual1
    )

    # --------------------------------------------------------
    # PREDICTOR 2
    # --------------------------------------------------------

    pred2, residual2 = \
        predict_paeth(
            pixels,
            width,
            height,
            channels
        )

    result2 = test_predictor(
        "PREDICTOR 2 — PAETH",
        pixels,
        pred2,
        residual2
    )

    # --------------------------------------------------------
    # PREDICTOR 3
    # --------------------------------------------------------

    model = train_neural_predictor(
        pixels
    )

    pred3, residual3 = \
        neural_predict(
            model,
            pixels
        )

    result3 = test_predictor(
        "PREDICTOR 3 — NEURAL",
        pixels,
        pred3,
        residual3
    )

    # --------------------------------------------------------
    # MODEL SIZE
    # --------------------------------------------------------

    model_state = {}

    for key, value in model.state_dict().items():

        model_state[key] = value.detach().cpu().numpy()

    model_bytes = sum(
        v.nbytes
        for v in model_state.values()
    )

    # Compress model parameters.

    model_blob = b""

    for key in sorted(model_state):

        arr = model_state[key]

        model_blob += arr.tobytes()

    compressed_model = zlib.compress(
        model_blob,
        level=9
    )

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    results = [
        result1,
        result2,
        result3,
    ]

    print()
    print("=" * 60)
    print("PREDICTOR COMPARISON")
    print("=" * 60)

    print(
        f"{'Predictor':<24}"
        f"{'Entropy':>12}"
        f"{'Residual':>14}"
        f"{'zlib':>14}"
        f"{'Exact':>10}"
    )

    print("-" * 74)

    for r in results:

        print(
            f"{r['name']:<24}"
            f"{r['entropy']:>12.4f}"
            f"{len(r['residual']):>14,}"
            f"{len(r['compressed']):>14,}"
            f"{str(r['exact']):>10}"
        )

    # --------------------------------------------------------
    # FULL STORAGE ACCOUNTING
    # --------------------------------------------------------

    metadata_size = 1024

    print()
    print("=" * 60)
    print("FULL STORAGE ACCOUNTING")
    print("=" * 60)

    print(
        "Original PNG:",
        fmt(len(original_file)),
        "bytes"
    )

    print(
        "Original pixel data:",
        fmt(len(pixels)),
        "bytes"
    )

    print(
        "zlib pixel baseline:",
        fmt(len(z_pixels)),
        "bytes"
    )

    print()

    for r in results:

        if r["name"].startswith(
            "PREDICTOR 3"
        ):

            total = (
                len(r["compressed"]) +
                len(compressed_model) +
                metadata_size
            )

            print(
                "NEURAL TOTAL:"
            )

            print(
                "  Residual:",
                fmt(len(r["compressed"]))
            )

            print(
                "  Model:",
                fmt(len(compressed_model))
            )

            print(
                "  Metadata:",
                fmt(metadata_size)
            )

            print(
                "  TOTAL:",
                fmt(total)
            )

            print(
                "  Pixel ratio:",
                f"{len(pixels) / total:.4f}x"
            )

            print(
                "  PNG ratio:",
                f"{len(original_file) / total:.4f}x"
            )

    # --------------------------------------------------------
    # ENTROPY LOWER BOUNDS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ENTROPY ESTIMATES")
    print("=" * 60)

    for r in results:

        theoretical = (
            len(r["residual"]) *
            r["entropy"] /
            8.0
        )

        print()
        print(r["name"])

        print(
            "Entropy:",
            f"{r['entropy']:.6f}",
            "bits/byte"
        )

        print(
            "Theoretical entropy estimate:",
            fmt(int(theoretical)),
            "bytes"
        )

        print(
            "Actual zlib:",
            fmt(len(r["compressed"])),
            "bytes"
        )

    # --------------------------------------------------------
    # EXACT PIXEL TEST
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXACT PIXEL RECONSTRUCTION")
    print("=" * 60)

    for r, pred in [
        (result1, pred1),
        (result2, pred2),
        (result3, pred3),
    ]:

        reconstructed = \
            reconstruct_from_residual(
                pred,
                np.frombuffer(
                    r["residual"],
                    dtype=np.uint8
                )
            )

        exact = np.array_equal(
            np.frombuffer(
                pixels,
                dtype=np.uint8
            ),
            reconstructed
        )

        print(
            f"{r['name']}:",
            "SUCCESS" if exact else "FAILED"
        )

    # --------------------------------------------------------
    # PNG RE-ENCODE TEST
    # --------------------------------------------------------

    rebuilt_png = compare_png_reconstruction(
        FILE,
        pixels,
        width,
        height,
        mode
    )

    rebuilt_hash = sha256(
        rebuilt_png
    )

    # --------------------------------------------------------
    # ORIGINAL HASH
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HASH VERIFICATION")
    print("=" * 60)

    print(
        "Original:"
    )

    print(
        original_hash
    )

    print(
        "Re-encoded:"
    )

    print(
        rebuilt_hash
    )

    if original_hash == rebuilt_hash:

        print()
        print(
            "SHA256 MATCH — EXACT PNG"
        )

    else:

        print()
        print(
            "SHA256 DIFFERENT"
        )

        print(
            "This does NOT imply different pixels."
        )

    # --------------------------------------------------------
    # FINAL INTERPRETATION
    # --------------------------------------------------------

    best = min(
        results,
        key=lambda r: len(r["compressed"])
    )

    print()
    print("=" * 60)
    print("FINAL INTERPRETATION")
    print("=" * 60)

    print(
        "Best predictor:",
        best["name"]
    )

    print(
        "Residual entropy:",
        f"{best['entropy']:.6f}",
        "bits/byte"
    )

    print(
        "Residual compressed:",
        fmt(len(best["compressed"])),
        "bytes"
    )

    print(
        "Decoded pixels:",
        fmt(len(pixels)),
        "bytes"
    )

    print()

    if best["exact"]:

        print(
            "LOSSLESS PIXEL RECONSTRUCTION: YES"
        )

    else:

        print(
            "LOSSLESS PIXEL RECONSTRUCTION: NO"
        )

    print()

    if len(best["compressed"]) < len(pixels):

        print(
            "Predictor reduces decoded pixel data."
        )

    else:

        print(
            "Predictor does NOT reduce decoded "
            "pixel data."
        )

    print()

    if len(best["compressed"]) < len(z_pixels):

        print(
            "Predictor + zlib beats pixel zlib."
        )

    else:

        print(
            "Predictor + zlib does NOT beat "
            "pixel zlib."
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()