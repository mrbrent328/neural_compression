#!/usr/bin/env python3

import argparse
import hashlib
import os
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONTEXTS = [4, 8, 16, 32]

DEFAULT_EPOCHS = 1500
DEFAULT_TRAIN_SAMPLES = 500_000

TRAIN_FRACTION = 0.75

HIDDEN_SIZE = 64
EMBED_SIZE = 16

BATCH_SIZE = 4096
LEARNING_RATE = 1e-3

PRINT_EVERY = 100

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# FILE HANDLING
# ============================================================

def load_file(path):
    with open(path, "rb") as f:
        return f.read()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# MODEL
# ============================================================

class BytePredictor(nn.Module):
    """
    Predicts the next byte from a fixed-width byte context.

    Instead of one-hot encoding every byte into 256 values,
    each byte gets a learned embedding.

    This makes wider contexts substantially cheaper.
    """

    def __init__(
        self,
        context_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
    ):
        super().__init__()

        self.context_size = context_size

        self.embedding = nn.Embedding(
            256,
            embed_size
        )

        self.net = nn.Sequential(
            nn.Linear(
                context_size * embed_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                256
            )
        )

    def forward(self, x):

        embedded = self.embedding(
            x.long()
        )

        embedded = embedded.reshape(
            x.shape[0],
            self.context_size * EMBED_SIZE
        )

        return self.net(
            embedded
        )


# ============================================================
# TRAINING DATA
# ============================================================

def make_training_samples(
    data,
    context_size,
    indices
):

    x = np.empty(
        (
            len(indices),
            context_size
        ),
        dtype=np.uint8
    )

    y = np.empty(
        len(indices),
        dtype=np.uint8
    )

    for i, pos in enumerate(indices):

        x[i] = np.frombuffer(
            data[
                pos - context_size:
                pos
            ],
            dtype=np.uint8
        )

        y[i] = data[pos]

    return x, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    data,
    context_size,
    device,
    epochs,
    train_samples
):

    split = int(
        len(data) * TRAIN_FRACTION
    )

    valid_positions = np.arange(
        context_size,
        split,
        dtype=np.int64
    )

    if len(valid_positions) == 0:

        raise ValueError(
            "File is too small for this context size."
        )

    sample_count = min(
        train_samples,
        len(valid_positions)
    )

    rng = np.random.default_rng(
        SEED
    )

    indices = rng.choice(
        valid_positions,
        size=sample_count,
        replace=False
    )

    x_np, y_np = make_training_samples(
        data,
        context_size,
        indices
    )

    x = torch.from_numpy(
        x_np
    ).to(device)

    y = torch.from_numpy(
        y_np.astype(np.int64)
    ).to(device)

    model = BytePredictor(
        context_size
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    start_time = time.time()

    n = len(x)

    for epoch in range(epochs):

        permutation = torch.randperm(
            n,
            device=device
        )

        total_loss = 0.0

        for batch_start in range(
            0,
            n,
            BATCH_SIZE
        ):

            batch_idx = permutation[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            xb = x[batch_idx]
            yb = y[batch_idx]

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(batch_idx)
            )

        avg_loss = (
            total_loss / n
        )

        if (
            epoch == 0
            or (epoch + 1) % PRINT_EVERY == 0
            or epoch == epochs - 1
        ):

            print(
                f"Epoch {epoch:5d} | "
                f"Loss: {avg_loss:.8f}"
            )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Training time: "
        f"{elapsed:.2f} seconds"
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_bytes(
    model,
    data,
    context_size,
    start,
    end,
    device
):

    model.eval()

    length = end - start

    predictions = np.empty(
        length,
        dtype=np.uint8
    )

    for batch_start in range(
        start,
        end,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            end
        )

        contexts = []

        for pos in range(
            batch_start,
            batch_end
        ):

            contexts.append(
                np.frombuffer(
                    data[
                        pos - context_size:
                        pos
                    ],
                    dtype=np.uint8
                )
            )

        x_np = np.stack(
            contexts
        )

        x = torch.from_numpy(
            x_np
        ).to(device)

        with torch.no_grad():

            logits = model(x)

            pred = torch.argmax(
                logits,
                dim=1
            ).cpu().numpy()

        offset = (
            batch_start - start
        )

        predictions[
            offset:
            offset + len(pred)
        ] = pred

    return predictions


# ============================================================
# TEST LOSS
# ============================================================

def calculate_test_loss(
    model,
    data,
    context_size,
    start,
    end,
    device
):

    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    total_loss = 0.0
    total_count = 0

    for batch_start in range(
        start,
        end,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            end
        )

        contexts = []

        targets = []

        for pos in range(
            batch_start,
            batch_end
        ):

            contexts.append(
                np.frombuffer(
                    data[
                        pos - context_size:
                        pos
                    ],
                    dtype=np.uint8
                )
            )

            targets.append(
                data[pos]
            )

        x_np = np.stack(
            contexts
        )

        y_np = np.asarray(
            targets,
            dtype=np.int64
        )

        x = torch.from_numpy(
            x_np
        ).to(device)

        y = torch.from_numpy(
            y_np
        ).to(device)

        with torch.no_grad():

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

        total_loss += loss.item()

        total_count += len(y)

    return (
        total_loss
        / total_count
    )


# ============================================================
# RESIDUAL
# ============================================================

def make_residual(
    actual,
    predictions
):

    actual_array = np.frombuffer(
        actual,
        dtype=np.uint8
    )

    residual = (
        actual_array.astype(np.int16)
        -
        predictions.astype(np.int16)
    ) % 256

    return residual.astype(
        np.uint8
    )


# ============================================================
# ENTROPY
# ============================================================

def entropy(data):

    values = np.asarray(
        data,
        dtype=np.uint8
    )

    counts = np.bincount(
        values,
        minlength=256
    )

    probabilities = (
        counts[counts > 0]
        / len(values)
    )

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


# ============================================================
# MODEL SERIALIZATION
# ============================================================

def model_bytes(model):

    chunks = []

    for tensor in (
        model.state_dict().values()
    ):

        chunks.append(
            tensor.detach()
            .cpu()
            .numpy()
            .tobytes()
        )

    return b"".join(
        chunks
    )


# ============================================================
# RESULT STRUCTURE
# ============================================================

@dataclass
class Result:

    context: int

    test_loss: float

    entropy: float

    zero_percent: float

    residual_zlib: int

    model_raw: int

    model_compressed: int

    metadata: int

    total: int

    mse: float

    mae: float

    max_error: int

    exact: bool


# ============================================================
# RUN ONE CONTEXT WIDTH
# ============================================================

def run_context(
    data,
    original_size,
    context_size,
    device,
    epochs,
    train_samples
):

    print()
    print("=" * 70)

    print(
        f"CONTEXT WIDTH: "
        f"{context_size} BYTES"
    )

    print("=" * 70)

    split = int(
        len(data)
        * TRAIN_FRACTION
    )

    test_start = split
    test_end = len(data)

    model = train_model(
        data=data,
        context_size=context_size,
        device=device,
        epochs=epochs,
        train_samples=train_samples
    )

    print()
    print(
        "HELD-OUT TEST"
    )

    test_loss = calculate_test_loss(
        model,
        data,
        context_size,
        test_start,
        test_end,
        device
    )

    print(
        f"Test loss: "
        f"{test_loss:.8f}"
    )

    predictions = predict_bytes(
        model,
        data,
        context_size,
        test_start,
        test_end,
        device
    )

    actual = np.frombuffer(
        data[
            test_start:
            test_end
        ],
        dtype=np.uint8
    )

    errors = (
        actual.astype(np.int16)
        -
        predictions.astype(np.int16)
    )

    mse = float(
        np.mean(
            errors.astype(
                np.float64
            ) ** 2
        )
    )

    mae = float(
        np.mean(
            np.abs(
                errors.astype(
                    np.float64
                )
            )
        )
    )

    max_error = int(
        np.max(
            np.abs(errors)
        )
    )

    residual = make_residual(
        data[
            test_start:
            test_end
        ],
        predictions
    )

    reconstructed = (
        predictions.astype(
            np.uint16
        )
        +
        residual.astype(
            np.uint16
        )
    ) % 256

    exact = bool(
        np.array_equal(
            reconstructed.astype(
                np.uint8
            ),
            actual
        )
    )

    residual_entropy = entropy(
        residual
    )

    zero_percent = (
        np.count_nonzero(
            residual == 0
        )
        /
        len(residual)
        *
        100.0
    )

    residual_raw = (
        residual.tobytes()
    )

    residual_compressed = zlib.compress(
        residual_raw,
        level=9
    )

    model_raw = model_bytes(
        model
    )

    model_compressed = zlib.compress(
        model_raw,
        level=9
    )

    metadata = 1024

    total = (
        len(residual_compressed)
        +
        len(model_compressed)
        +
        metadata
    )

    print()
    print(
        f"Prediction MSE:       "
        f"{mse:.6f}"
    )

    print(
        f"Prediction MAE:       "
        f"{mae:.6f}"
    )

    print(
        f"Maximum prediction error: "
        f"{max_error}"
    )

    print(
        f"Zero residual:         "
        f"{zero_percent:.3f}%"
    )

    print(
        f"Residual entropy:      "
        f"{residual_entropy:.6f} bits/byte"
    )

    print(
        f"Residual raw:          "
        f"{len(residual_raw):,} bytes"
    )

    print(
        f"Residual zlib:         "
        f"{len(residual_compressed):,} bytes"
    )

    print(
        f"Model raw:             "
        f"{len(model_raw):,} bytes"
    )

    print(
        f"Model compressed:      "
        f"{len(model_compressed):,} bytes"
    )

    print(
        f"Metadata:              "
        f"{metadata:,} bytes"
    )

    print(
        f"TOTAL REPRESENTATION:  "
        f"{total:,} bytes"
    )

    print(
        f"Original / total:      "
        f"{original_size / total:.4f}x"
    )

    print(
        f"Exact reconstruction:  "
        f"{exact}"
    )

    return Result(
        context=context_size,
        test_loss=test_loss,
        entropy=residual_entropy,
        zero_percent=zero_percent,
        residual_zlib=len(
            residual_compressed
        ),
        model_raw=len(
            model_raw
        ),
        model_compressed=len(
            model_compressed
        ),
        metadata=metadata,
        total=total,
        mse=mse,
        mae=mae,
        max_error=max_error,
        exact=exact
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Learned next-byte prediction "
            "with multiple context widths."
        )
    )

    parser.add_argument(
        "file",
        help="File to analyze."
    )

    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=DEFAULT_CONTEXTS,
        help=(
            "Context widths. "
            "Default: 4 8 16 32"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            "Training epochs. "
            "Default: 1500"
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES,
        help=(
            "Training samples. "
            "Default: 500000"
        )
    )

    args = parser.parse_args()

    filename = args.file

    if not os.path.isfile(
        filename
    ):

        print()
        print(
            f"ERROR: File not found:"
        )

        print(
            filename
        )

        return

    data = load_file(
        filename
    )

    original_size = len(data)

    original_sha = sha256_bytes(
        data
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print(
        "LEARNED CONTEXT-WIDTH "
        "STORAGE EXPERIMENT"
    )
    print("=" * 70)

    print()
    print(
        f"Device:       {device}"
    )

    print(
        f"File:         {filename}"
    )

    print(
        f"Original:     "
        f"{original_size:,} bytes"
    )

    print(
        f"SHA256:       "
        f"{original_sha}"
    )

    print()

    original_zlib = zlib.compress(
        data,
        level=9
    )

    print(
        f"Zlib:         "
        f"{len(original_zlib):,} bytes"
    )

    print(
        f"Zlib ratio:   "
        f"{original_size / len(original_zlib):.4f}x"
    )

    print()

    print(
        f"Contexts:     "
        f"{args.contexts}"
    )

    print(
        f"Epochs:       "
        f"{args.epochs}"
    )

    print(
        f"Train samples:"
        f" {args.samples:,}"
    )

    print(
        f"Hidden size:  "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"Embedding:    "
        f"{EMBED_SIZE}"
    )

    print()

    results = []

    for context_size in args.contexts:

        if context_size >= (
            len(data)
            * TRAIN_FRACTION
        ):

            print(
                f"Skipping {context_size}-byte "
                f"context: file too small."
            )

            continue

        result = run_context(
            data=data,
            original_size=original_size,
            context_size=context_size,
            device=device,
            epochs=args.epochs,
            train_samples=args.samples
        )

        results.append(
            result
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FINAL CONTEXT-WIDTH COMPARISON"
    )
    print("=" * 110)

    print()

    header = (
        f"{'Context':>8} "
        f"{'Test Loss':>12} "
        f"{'Entropy':>10} "
        f"{'Zero %':>9} "
        f"{'Residual':>12} "
        f"{'Model':>10} "
        f"{'TOTAL':>12} "
        f"{'vs Orig':>9}"
    )

    print(header)
    print("-" * 110)

    for r in results:

        ratio = (
            r.total
            /
            original_size
        )

        print(
            f"{r.context:>8} "
            f"{r.test_loss:>12.6f} "
            f"{r.entropy:>10.4f} "
            f"{r.zero_percent:>8.2f}% "
            f"{r.residual_zlib:>12,} "
            f"{r.model_compressed:>10,} "
            f"{r.total:>12,} "
            f"{ratio:>9.4f}x"
        )

    print()

    print(
        f"{'Original':>8} "
        f"{'—':>12} "
        f"{'—':>10} "
        f"{'—':>9} "
        f"{'—':>12} "
        f"{'—':>10} "
        f"{original_size:>12,} "
        f"{1.0000:>9.4f}x"
    )

    print(
        f"{'zlib':>8} "
        f"{'—':>12} "
        f"{'—':>10} "
        f"{'—':>9} "
        f"{'—':>12} "
        f"{'—':>10} "
        f"{len(original_zlib):>12,} "
        f"{len(original_zlib) / original_size:>9.4f}x"
    )

    if results:

        best = min(
            results,
            key=lambda r: r.total
        )

        print()
        print("=" * 70)
        print("BEST LEARNED MODEL")
        print("=" * 70)

        print(
            f"Context:      "
            f"{best.context} bytes"
        )

        print(
            f"Total:        "
            f"{best.total:,} bytes"
        )

        print(
            f"Original:     "
            f"{original_size:,} bytes"
        )

        print(
            f"Zlib:         "
            f"{len(original_zlib):,} bytes"
        )

        print(
            f"Difference vs original: "
            f"{best.total - original_size:+,} bytes"
        )

        print(
            f"Difference vs zlib:     "
            f"{best.total - len(original_zlib):+,} bytes"
        )

        print(
            f"Exact reconstruction: "
            f"{best.exact}"
        )

        print()

        if (
            best.total
            <
            len(original_zlib)
        ):

            print(
                "*** LEARNED PROTOCOL "
                "BEATS ZLIB ***"
            )

        elif (
            best.total
            <
            original_size
        ):

            print(
                "*** LEARNED PROTOCOL "
                "BEATS ORIGINAL ***"
            )

        else:

            print(
                "Learned protocol does not "
                "beat the baseline yet."
            )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()