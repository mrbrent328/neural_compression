#!/usr/bin/env python3

"""
============================================================
NEURAL PREDICTOR + EXACT RESIDUAL COMPRESSION PROTOCOL
============================================================

The neural network predicts blocks of a file.

We NEVER throw away the prediction error.

For every byte:

    residual = (actual - prediction) mod 256

The residual is stored exactly.

Reconstruction:

    prediction + residual = original byte

Therefore this protocol is ALWAYS LOSSLESS.

The experiment is:

    original
        |
        v
    neural predictor
        |
        +----> prediction
        |
        +----> exact residual
                    |
                    v
                  zlib

Final storage:

    model + residual + metadata

We compare this against:

    original
    zlib(original)

============================================================
"""

import os
import sys
import zlib
import hashlib
import pickle
import struct
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch is required.")
    print("Install with:")
    print("    pip install torch")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

FILE_NAME = "pic.png"

BLOCK_SIZE = 1024

LATENT_SIZE = 16
HIDDEN_SIZE = 64

EPOCHS = 2000
PRINT_EVERY = 100

LEARNING_RATE = 0.001

# Number of blocks used during training.
# None = all blocks.
MAX_TRAIN_BLOCKS = None

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ============================================================
# HELPERS
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_file(filename):
    with open(filename, "rb") as f:
        return f.read()


def split_blocks(data, block_size):
    """
    Split bytes into fixed-size blocks.

    Last block is zero padded.
    The original file size is stored separately.
    """

    data = np.frombuffer(data, dtype=np.uint8)

    num_blocks = int(np.ceil(len(data) / block_size))

    padded_size = num_blocks * block_size

    padded = np.zeros(padded_size, dtype=np.uint8)
    padded[:len(data)] = data

    blocks = padded.reshape(num_blocks, block_size)

    return blocks, len(data)


# ============================================================
# MODEL
# ============================================================

class Predictor(nn.Module):

    def __init__(self, block_size, hidden_size, latent_size):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(block_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, block_size),
            nn.Sigmoid(),
        )

    def forward(self, x):

        z = self.encoder(x)
        y = self.decoder(z)

        return y


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("NEURAL PREDICTOR + EXACT RESIDUAL")
print("=" * 60)

if not os.path.exists(FILE_NAME):
    print()
    print("ERROR:")
    print(f"Could not find: {FILE_NAME}")
    print()
    print("Put the file next to this script or change:")
    print()
    print('FILE_NAME = "pic.png"')
    sys.exit(1)


original = load_file(FILE_NAME)

original_size = len(original)

original_hash = sha256(original)

print()
print("File:")
print(" ", FILE_NAME)

print()
print("Original size:")
print(f" {original_size:,} bytes")

print()
print("SHA256:")
print(" ", original_hash)


# ============================================================
# ZLIB BASELINE
# ============================================================

zlib_original = zlib.compress(original, level=9)

print()
print("=" * 60)
print("ZLIB BASELINE")
print("=" * 60)

print()
print(f"Original: {original_size:,} bytes")
print(f"Zlib:     {len(zlib_original):,} bytes")

print(
    f"Ratio:    {original_size / len(zlib_original):.4f}x"
)


# ============================================================
# BLOCKS
# ============================================================

blocks, true_size = split_blocks(
    original,
    BLOCK_SIZE
)

num_blocks = len(blocks)

print()
print("=" * 60)
print("BLOCK STRUCTURE")
print("=" * 60)

print()
print("Block size:")
print(f" {BLOCK_SIZE:,} bytes")

print()
print("Number of blocks:")
print(f" {num_blocks:,}")

print()
print("Total padded bytes:")
print(f" {num_blocks * BLOCK_SIZE:,}")


# ============================================================
# NORMALIZE DATA
# ============================================================

x = torch.tensor(
    blocks,
    dtype=torch.float32,
    device=DEVICE
) / 255.0


# ============================================================
# TRAINING DATA
# ============================================================

if MAX_TRAIN_BLOCKS is not None:
    train_count = min(
        MAX_TRAIN_BLOCKS,
        num_blocks
    )

    train_x = x[:train_count]

else:
    train_x = x


# ============================================================
# CREATE MODEL
# ============================================================

model = Predictor(
    BLOCK_SIZE,
    HIDDEN_SIZE,
    LATENT_SIZE
).to(DEVICE)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


model_raw_size = parameter_count * 4


print()
print("=" * 60)
print("MODEL")
print("=" * 60)

print()
print("Parameters:")
print(f" {parameter_count:,}")

print()
print("Approx raw model size:")
print(f" {model_raw_size:,} bytes")

print()
print("Latent size:")
print(f" {LATENT_SIZE}")


# ============================================================
# TRAIN
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

start_time = time.time()

model.train()

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    prediction = model(train_x)

    loss = criterion(
        prediction,
        train_x
    )

    loss.backward()

    optimizer.step()

    if (
        epoch % PRINT_EVERY == 0
        or epoch == EPOCHS - 1
    ):

        print(
            f"Epoch {epoch:5d} | "
            f"Loss: {loss.item():.10f}"
        )


training_time = time.time() - start_time


# ============================================================
# PREDICT EVERYTHING
# ============================================================

print()
print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

model.eval()

with torch.no_grad():

    predictions = model(x)

predictions = (
    predictions
    .cpu()
    .numpy()
)

# Convert model output back into bytes.

predicted_bytes = np.rint(
    predictions * 255.0
).astype(np.int16)

predicted_bytes = np.clip(
    predicted_bytes,
    0,
    255
)


# ============================================================
# EXACT RESIDUAL
# ============================================================

actual_bytes = blocks.astype(np.int16)

"""
Modulo arithmetic is important.

For example:

actual     = 10
prediction = 250

10 - 250 = -240

Modulo 256:

16

Later:

250 + 16 = 266

266 mod 256 = 10

Exact reconstruction.
"""

residual = (
    actual_bytes -
    predicted_bytes
) % 256

residual = residual.astype(np.uint8)


# ============================================================
# PREDICTION QUALITY
# ============================================================

prediction_error = (
    actual_bytes -
    predicted_bytes
)

prediction_mse = np.mean(
    prediction_error.astype(np.float64) ** 2
)

prediction_mae = np.mean(
    np.abs(
        prediction_error
    )
)

prediction_max = np.max(
    np.abs(prediction_error)
)

zero_residual_percent = (
    np.mean(residual == 0) * 100.0
)


print()
print("=" * 60)
print("PREDICTION QUALITY")
print("=" * 60)

print()
print(
    f"Prediction MSE:       "
    f"{prediction_mse:.6f}"
)

print(
    f"Prediction MAE:       "
    f"{prediction_mae:.6f}"
)

print(
    f"Maximum prediction error: "
    f"{prediction_max}"
)

print(
    f"Zero residual bytes:   "
    f"{zero_residual_percent:.2f}%"
)


# ============================================================
# RESIDUAL SERIALIZATION
# ============================================================

residual_raw = residual.tobytes()


# ============================================================
# COMPRESS RESIDUAL
# ============================================================

print()
print("=" * 60)
print("RESIDUAL COMPRESSION")
print("=" * 60)

residual_zlib = zlib.compress(
    residual_raw,
    level=9
)

print()
print(
    f"Residual raw: "
    f"{len(residual_raw):,} bytes"
)

print(
    f"Residual zlib: "
    f"{len(residual_zlib):,} bytes"
)

print(
    f"Residual ratio: "
    f"{len(residual_raw) / len(residual_zlib):.4f}x"
)


# ============================================================
# MODEL SERIALIZATION
# ============================================================

"""
Store the model weights.

The metadata is deliberately explicit so that
the protocol can reconstruct everything.
"""

model_state = {
    key: value.detach().cpu().numpy()
    for key, value in model.state_dict().items()
}

model_blob = pickle.dumps(
    model_state,
    protocol=pickle.HIGHEST_PROTOCOL
)

model_compressed = zlib.compress(
    model_blob,
    level=9
)


# ============================================================
# PROTOCOL PACKAGE
# ============================================================

metadata = {
    "version": 1,

    "original_size": original_size,

    "block_size": BLOCK_SIZE,

    "num_blocks": num_blocks,

    "latent_size": LATENT_SIZE,

    "hidden_size": HIDDEN_SIZE,

    "parameter_count": parameter_count,

    "original_sha256": original_hash,

    "model_format": "numpy_state_dict",

    "residual_format": "uint8_mod_256",

    "compression": "zlib",
}


metadata_blob = pickle.dumps(
    metadata,
    protocol=pickle.HIGHEST_PROTOCOL
)


# Header:

# [metadata length][metadata][model][residual]

header = struct.pack(
    "<Q",
    len(metadata_blob)
)

package = (
    header +
    metadata_blob +
    model_compressed +
    residual_zlib
)


# ============================================================
# STORAGE RESULTS
# ============================================================

protocol_file = FILE_NAME + ".neuralres"

with open(protocol_file, "wb") as f:
    f.write(package)


protocol_size = len(package)

print()
print("=" * 60)
print("STORAGE")
print("=" * 60)

print()
print(
    f"Original:              "
    f"{original_size:,} bytes"
)

print(
    f"Original zlib:         "
    f"{len(zlib_original):,} bytes"
)

print(
    f"Model compressed:      "
    f"{len(model_compressed):,} bytes"
)

print(
    f"Residual raw:          "
    f"{len(residual_raw):,} bytes"
)

print(
    f"Residual compressed:   "
    f"{len(residual_zlib):,} bytes"
)

print(
    f"Metadata:              "
    f"{len(metadata_blob):,} bytes"
)

print()
print(
    f"TOTAL PROTOCOL:        "
    f"{protocol_size:,} bytes"
)

print()
print(
    f"Original / protocol:   "
    f"{original_size / protocol_size:.4f}x"
)

print(
    f"Zlib / protocol:       "
    f"{len(zlib_original) / protocol_size:.4f}x"
)


# ============================================================
# RECONSTRUCTION TEST
# ============================================================

print()
print("=" * 60)
print("LOSSLESS RECONSTRUCTION")
print("=" * 60)

"""
Reconstruct from:

    predicted bytes
    +
    exact residual
"""

reconstructed_padded = (
    predicted_bytes +
    residual.astype(np.int16)
) % 256

reconstructed_padded = (
    reconstructed_padded
    .astype(np.uint8)
    .reshape(-1)
)

reconstructed = (
    reconstructed_padded[
        :original_size
    ]
    .tobytes()
)


reconstructed_hash = sha256(
    reconstructed
)

exact = (
    reconstructed ==
    original
)


print()
print("Original SHA256:")
print(" ", original_hash)

print()
print("Reconstructed SHA256:")
print(" ", reconstructed_hash)

print()

if exact:

    print("SUCCESS")
    print()
    print(
        "The neural predictor + residual "
        "protocol reconstructed the file "
        "EXACTLY."
    )

else:

    print("FAILURE")
    print()
    print(
        "The reconstruction was NOT exact."
    )

    # This should never happen unless there
    # is a bug in the protocol.

    difference_count = sum(
        a != b
        for a, b in zip(
            original,
            reconstructed
        )
    )

    print()
    print(
        f"Different bytes: "
        f"{difference_count:,}"
    )


# ============================================================
# RESIDUAL ENTROPY
# ============================================================

print()
print("=" * 60)
print("RESIDUAL INFORMATION")
print("=" * 60)

histogram = np.bincount(
    residual.flatten(),
    minlength=256
)

probabilities = (
    histogram /
    histogram.sum()
)

nonzero = probabilities[
    probabilities > 0
]

entropy = -np.sum(
    nonzero *
    np.log2(nonzero)
)

print()
print(
    f"Residual entropy: "
    f"{entropy:.4f} bits/byte"
)

print()
print(
    "The theoretical lower bound from "
    "the measured residual entropy is:"
)

entropy_bytes = (
    entropy / 8.0 *
    original_size
)

print(
    f" {entropy_bytes:,.0f} bytes"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print()
print(
    f"{'Representation':<25}"
    f"{'Bytes':>15}"
    f"{'Ratio':>12}"
)

print("-" * 52)

print(
    f"{'Original':<25}"
    f"{original_size:>15,}"
    f"{1.0:>12.4f}x"
)

print(
    f"{'Zlib':<25}"
    f"{len(zlib_original):>15,}"
    f"{original_size / len(zlib_original):>11.4f}x"
)

print(
    f"{'Neural + residual':<25}"
    f"{protocol_size:>15,}"
    f"{original_size / protocol_size:>11.4f}x"
)


# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)

if not exact:

    print()
    print(
        "LOSSLESS TEST FAILED."
    )

elif protocol_size < len(zlib_original):

    improvement = (
        1 -
        protocol_size /
        len(zlib_original)
    ) * 100

    print()
    print(
        "NEURAL RESIDUAL PROTOCOL "
        "BEATS ZLIB."
    )

    print()
    print(
        f"Improvement over zlib: "
        f"{improvement:.2f}%"
    )

elif protocol_size < original_size:

    improvement = (
        1 -
        protocol_size /
        original_size
    ) * 100

    print()
    print(
        "LOSSLESS COMPRESSION ACHIEVED,"
    )

    print(
        "but zlib is still better."
    )

    print()
    print(
        f"Reduction vs original: "
        f"{improvement:.2f}%"
    )

else:

    print()
    print(
        "The protocol is lossless,"
    )

    print(
        "but does not compress this file."
    )


print()
print("=" * 60)
print("PROTOCOL FILE")
print("=" * 60)

print()
print(protocol_file)

print()
print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)

print()
print("DONE.")