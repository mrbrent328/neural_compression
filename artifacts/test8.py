#!/usr/bin/env python3

"""
TEST8 — REAL SOURCE-CODE PREDICTIVE COMPRESSION

Tests predictive compression against real source-code files.

Source:
    /home/xetal/Downloads/

Methods:
    1. zlib baseline
    2. previous-byte predictor
    3. 4-byte context predictor
    4. neural predictor
    5. hybrid context + neural predictor

Important:
    - Training data is the first 75% of each dataset.
    - The final 25% is held out.
    - Held-out bytes are NEVER used to train the models.
    - Every predictor must reconstruct the held-out bytes exactly.
    - Model size + residual size + metadata are counted.

Neural model:
    4-byte causal context
    -> embedding
    -> hidden layer
    -> 256 byte classes

Requires:
    Python 3
    numpy
    torch

This version is designed for the AMD ROCm PyTorch environment.
"""

import os
import sys
import zlib
import json
import csv
import time
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch is required.")
    print("Install PyTorch first.")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path("/home/xetal/Downloads")

TRAIN_FRACTION = 0.75
CONTEXT_SIZE = 4

# Neural training
#
# 300k samples is enough for a useful first experiment.
# 25 epochs is deliberately much more reasonable than 500.
NEURAL_SAMPLES = 300_000
NEURAL_EPOCHS = 25
NEURAL_HIDDEN = 128
NEURAL_EMBED = 32
NEURAL_BATCH = 4096
NEURAL_LR = 0.001

RANDOM_SEED = 1337

# Fixed accounting overhead.
METADATA_BYTES = 1024

TARGET_SIZES = [
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
]

MAX_FILE_SIZE = 5_000_000


# ============================================================
# SOURCE EXTENSIONS
# ============================================================

SOURCE_EXTENSIONS = {
    ".py", ".pyw", ".pyi",
    ".c", ".h", ".cc", ".cpp", ".cxx",
    ".hpp", ".hh",
    ".java",
    ".kt", ".kts",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".rs",
    ".go",
    ".swift",
    ".m", ".mm",
    ".cs",
    ".fs", ".fsx",
    ".vb",
    ".php",
    ".rb",
    ".lua",
    ".pl", ".pm",
    ".r",
    ".dart",
    ".scala",
    ".sh", ".bash", ".zsh", ".fish",
    ".ps1",
    ".bat", ".cmd",
    ".sql",
    ".html", ".htm",
    ".css", ".scss", ".sass", ".less",
    ".xml", ".xsl", ".xslt",
    ".json", ".jsonl",
    ".yaml", ".yml",
    ".toml",
    ".ini", ".cfg", ".conf",
    ".md",
    ".txt",
}


# ============================================================
# DIRECTORIES TO SKIP
# ============================================================

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".bzr",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "bower_components",
    "target",
    "build",
    "dist",
    "out",
    "bin",
    "obj",
    ".idea",
    ".vscode",
    ".cache",
    "cache",
    "coverage",
    ".coverage",
    ".next",
    ".nuxt",
    ".gradle",
    ".cargo",
    ".npm",
}


OUTPUT_FILES = {
    "test8.py",
    "test8_results.csv",
    "test8_results.json",
    "test8_manifest.json",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# UTILITIES
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def format_bytes(n):
    if n < 1024:
        return f"{n:,} B"

    if n < 1024 ** 2:
        return f"{n / 1024:.2f} KB"

    if n < 1024 ** 3:
        return f"{n / (1024 ** 2):.2f} MB"

    return f"{n / (1024 ** 3):.2f} GB"


def entropy_bytes(data):
    if not data:
        return 0.0

    arr = np.frombuffer(data, dtype=np.uint8)

    counts = np.bincount(
        arr,
        minlength=256
    )

    probabilities = counts[counts > 0] / len(arr)

    return float(
        -np.sum(
            probabilities * np.log2(probabilities)
        )
    )


def residual_stats(residual):
    if not residual:
        return {
            "entropy": 0.0,
            "zero_percent": 0.0,
            "mse": 0.0,
            "mae": 0.0,
            "max_error": 0,
        }

    arr = np.frombuffer(
        residual,
        dtype=np.uint8
    )

    signed = arr.astype(np.int16)

    signed[signed >= 128] -= 256

    return {
        "entropy": entropy_bytes(residual),
        "zero_percent": float(
            np.mean(arr == 0) * 100.0
        ),
        "mse": float(
            np.mean(
                signed.astype(np.float64) ** 2
            )
        ),
        "mae": float(
            np.mean(
                np.abs(
                    signed.astype(np.float64)
                )
            )
        ),
        "max_error": int(
            np.max(
                np.abs(signed)
            )
        ),
    }


def zlib_compress(data):
    return zlib.compress(
        data,
        level=9
    )


# ============================================================
# SOURCE DISCOVERY
# ============================================================

def looks_textual(path):
    try:
        with open(path, "rb") as f:
            sample = f.read(8192)

        if b"\x00" in sample:
            return False

        return True

    except Exception:
        return False


def discover_source_files(root):
    files = []

    print()
    print("=" * 70)
    print("SOURCE DISCOVERY")
    print("=" * 70)
    print(f"Root: {root}")

    if not root.exists():
        raise RuntimeError(
            f"Directory does not exist: {root}"
        )

    for current_root, dirs, filenames in os.walk(root):

        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not d.startswith(".git")
        ]

        current_path = Path(current_root)

        for filename in filenames:

            path = current_path / filename

            if filename in OUTPUT_FILES:
                continue

            suffix = path.suffix.lower()

            if suffix not in SOURCE_EXTENSIONS:
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            if stat.st_size == 0:
                continue

            if stat.st_size > MAX_FILE_SIZE:
                print(
                    f"Skipping huge source file: "
                    f"{path} "
                    f"({format_bytes(stat.st_size)})"
                )
                continue

            if not looks_textual(path):
                continue

            files.append(path)

    files.sort(
        key=lambda p:
        str(p.relative_to(root)).lower()
    )

    return files


# ============================================================
# CORPUS
# ============================================================

def read_file(path):
    with open(path, "rb") as f:
        return f.read()


def build_corpus(files, root, target_size=None):

    chunks = []
    boundaries = []

    total = 0

    for path in files:

        try:
            data = read_file(path)
        except Exception as e:
            print(
                f"Could not read {path}: {e}"
            )
            continue

        if not data:
            continue

        if target_size is not None:

            remaining = target_size - total

            if remaining <= 0:
                break

            if len(data) > remaining:
                data = data[:remaining]

        start = total
        end = total + len(data)

        chunks.append(data)

        boundaries.append({
            "path": str(
                path.relative_to(root)
            ),
            "start": start,
            "end": end,
            "size": len(data),
        })

        total = end

        if (
            target_size is not None
            and total >= target_size
        ):
            break

    return b"".join(chunks), boundaries


def build_boundary_set(boundaries):

    starts = {0}

    for b in boundaries:
        starts.add(b["start"])

    return starts


# ============================================================
# PREVIOUS BYTE
# ============================================================

def previous_predict(
    data,
    train_end,
    boundary_starts
):

    heldout = data[train_end:]

    if not heldout:
        return b""

    predictions = np.empty(
        len(heldout),
        dtype=np.uint8
    )

    for i in range(len(heldout)):

        global_index = train_end + i

        if global_index in boundary_starts:
            predictions[i] = 0
        else:
            predictions[i] = data[
                global_index - 1
            ]

    actual = np.frombuffer(
        heldout,
        dtype=np.uint8
    )

    residual = (
        actual.astype(np.int16)
        -
        predictions.astype(np.int16)
    ) % 256

    return residual.astype(
        np.uint8
    ).tobytes()


# ============================================================
# CONTEXT MODEL
# ============================================================

def train_context_model(
    data,
    train_end,
    boundaries,
    context_size
):

    counts = defaultdict(Counter)

    for b in boundaries:

        start = b["start"]

        end = min(
            b["end"],
            train_end
        )

        if end - start <= context_size:
            continue

        local = data[start:end]

        for i in range(
            context_size,
            len(local)
        ):

            context = tuple(
                local[
                    i - context_size:i
                ]
            )

            counts[context][
                local[i]
            ] += 1

    model = {}

    for context, counter in counts.items():

        model[context] = (
            counter.most_common(1)[0][0]
        )

    return model


def context_predict(
    data,
    train_end,
    boundaries,
    context_model,
    context_size
):

    heldout = data[train_end:]

    if not heldout:
        return b""

    boundary_starts = build_boundary_set(
        boundaries
    )

    predictions = np.empty(
        len(heldout),
        dtype=np.uint8
    )

    for i in range(len(heldout)):

        global_index = train_end + i

        if global_index in boundary_starts:
            predictions[i] = 0
            continue

        start = (
            global_index
            - context_size
        )

        if start < 0:
            predictions[i] = data[
                global_index - 1
            ]
            continue

        context = tuple(
            data[start:global_index]
        )

        predictions[i] = context_model.get(
            context,
            data[global_index - 1]
        )

    actual = np.frombuffer(
        heldout,
        dtype=np.uint8
    )

    residual = (
        actual.astype(np.int16)
        -
        predictions.astype(np.int16)
    ) % 256

    return residual.astype(
        np.uint8
    ).tobytes()


def estimate_context_model_size(
    context_model
):

    if not context_model:
        return 0

    context_size = len(
        next(iter(context_model))
    )

    per_entry = (
        context_size
        + 1
        + 8
    )

    return (
        len(context_model)
        * per_entry
    )


# ============================================================
# NEURAL MODEL
# ============================================================

class NeuralPredictor(nn.Module):

    def __init__(
        self,
        context_size=4,
        embed_size=32,
        hidden_size=128
    ):

        super().__init__()

        self.context_size = context_size

        self.embedding = nn.Embedding(
            256,
            embed_size
        )

        self.fc1 = nn.Linear(
            context_size * embed_size,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            256
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.embedding(x)

        x = x.reshape(
            x.shape[0],
            -1
        )

        x = self.activation(
            self.fc1(x)
        )

        return self.fc2(x)


def neural_model_size(model):

    return sum(
        tensor.numel()
        * tensor.element_size()
        for tensor in model.state_dict().values()
    )


# ============================================================
# NEURAL TRAINING DATA
# ============================================================

def make_neural_training_samples(
    data,
    train_end,
    boundaries,
    context_size,
    sample_count
):

    # Count eligible positions first.
    eligible_count = 0

    for b in boundaries:

        start = b["start"]
        end = min(
            b["end"],
            train_end
        )

        if end - start > context_size:
            eligible_count += (
                end
                - start
                - context_size
            )

    if eligible_count <= 0:
        raise RuntimeError(
            "Not enough training data "
            "for neural model."
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sample_count = min(
        sample_count,
        eligible_count
    )

    # Generate global ordinal positions
    # rather than building millions of Python objects.
    chosen = np.sort(
        rng.choice(
            eligible_count,
            size=sample_count,
            replace=False
        )
    )

    X = np.empty(
        (
            sample_count,
            context_size
        ),
        dtype=np.int64
    )

    y = np.empty(
        sample_count,
        dtype=np.int64
    )

    chosen_index = 0
    cumulative = 0

    for b in boundaries:

        start = b["start"]
        end = min(
            b["end"],
            train_end
        )

        count = (
            end
            - start
            - context_size
        )

        if count <= 0:
            continue

        lower = cumulative
        upper = cumulative + count

        while (
            chosen_index < sample_count
            and chosen[chosen_index] < upper
        ):

            ordinal = int(
                chosen[chosen_index]
            )

            pos = (
                start
                + context_size
                + (
                    ordinal
                    - lower
                )
            )

            X[
                chosen_index
            ] = np.frombuffer(
                data[
                    pos - context_size:pos
                ],
                dtype=np.uint8
            )

            y[
                chosen_index
            ] = data[pos]

            chosen_index += 1

        cumulative = upper

        if chosen_index >= sample_count:
            break

    return X, y


# ============================================================
# TRAIN NEURAL MODEL
# ============================================================

def train_neural_model(
    data,
    train_end,
    boundaries
):

    print()
    print("-" * 70)
    print("NEURAL TRAINING")
    print("-" * 70)

    print(
        f"Device:       {DEVICE}"
    )

    if DEVICE.type == "cuda":
        print(
            f"GPU:          "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Samples:      {NEURAL_SAMPLES:,}"
    )

    print(
        f"Epochs:       {NEURAL_EPOCHS}"
    )

    print(
        f"Batch:        {NEURAL_BATCH:,}"
    )

    X, y = make_neural_training_samples(
        data,
        train_end,
        boundaries,
        CONTEXT_SIZE,
        NEURAL_SAMPLES
    )

    model = NeuralPredictor(
        context_size=CONTEXT_SIZE,
        embed_size=NEURAL_EMBED,
        hidden_size=NEURAL_HIDDEN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NEURAL_LR
    )

    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=NEURAL_BATCH,
        shuffle=True,
        generator=generator,
        drop_last=False
    )

    model.train()

    start_time = time.time()

    for epoch in range(
        NEURAL_EPOCHS
    ):

        running_loss = 0.0
        batches = 0

        for xb, yb in loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

            batches += 1

        avg_loss = (
            running_loss / batches
            if batches
            else 0.0
        )

        print(
            f"Epoch "
            f"{epoch + 1:3d}/"
            f"{NEURAL_EPOCHS}"
            f" | loss={avg_loss:.6f}"
        )

    elapsed = (
        time.time()
        - start_time
    )

    model.eval()

    print(
        f"Training time: "
        f"{elapsed:.2f} s"
    )

    print(
        f"Model size: "
        f"{neural_model_size(model):,} bytes"
    )

    return model


# ============================================================
# NEURAL PREDICTION
# ============================================================

@torch.no_grad()
def neural_predict(
    model,
    data,
    train_end,
    boundaries
):

    heldout = data[train_end:]

    if not heldout:
        return b""

    boundary_starts = build_boundary_set(
        boundaries
    )

    predictions = np.empty(
        len(heldout),
        dtype=np.uint8
    )

    valid_positions = []

    contexts = []

    for i in range(
        len(heldout)
    ):

        global_index = (
            train_end + i
        )

        if global_index in boundary_starts:
            predictions[i] = 0
            continue

        start = (
            global_index
            - CONTEXT_SIZE
        )

        if start < 0:
            predictions[i] = data[
                global_index - 1
            ]
            continue

        contexts.append(
            data[
                start:global_index
            ]
        )

        valid_positions.append(i)

    if contexts:

        X = np.frombuffer(
            b"".join(contexts),
            dtype=np.uint8
        ).reshape(
            -1,
            CONTEXT_SIZE
        ).astype(np.int64)

        X = torch.from_numpy(
            X
        ).to(DEVICE)

        logits = model(X)

        preds = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy().astype(
            np.uint8
        )

        predictions[
            valid_positions
        ] = preds

    actual = np.frombuffer(
        heldout,
        dtype=np.uint8
    )

    residual = (
        actual.astype(np.int16)
        -
        predictions.astype(np.int16)
    ) % 256

    return residual.astype(
        np.uint8
    ).tobytes()


# ============================================================
# HYBRID MODEL
# ============================================================

def train_hybrid_selector(
    data,
    train_end,
    boundaries,
    context_model,
    neural_model
):

    print()
    print(
        "Building hybrid selector..."
    )

    # Use the final 10% of the training
    # region as validation.
    validation_start = int(
        train_end * 0.90
    )

    context_correct = Counter()
    neural_correct = Counter()

    neural_model.eval()

    for b in boundaries:

        start = max(
            b["start"] + CONTEXT_SIZE,
            validation_start
        )

        end = min(
            b["end"],
            train_end
        )

        if end <= start:
            continue

        positions = range(
            start,
            end
        )

        contexts = []
        valid_positions = []

        for pos in positions:

            contexts.append(
                data[
                    pos - CONTEXT_SIZE:pos
                ]
            )

            valid_positions.append(pos)

        if not contexts:
            continue

        X = np.frombuffer(
            b"".join(contexts),
            dtype=np.uint8
        ).reshape(
            -1,
            CONTEXT_SIZE
        ).astype(np.int64)

        X = torch.from_numpy(
            X
        ).to(DEVICE)

        with torch.no_grad():
            logits = neural_model(X)

        neural_preds = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

        for pos, neural_pred in zip(
            valid_positions,
            neural_preds
        ):

            actual = data[pos]

            context = tuple(
                data[
                    pos - CONTEXT_SIZE:pos
                ]
            )

            context_pred = (
                context_model.get(
                    context,
                    data[pos - 1]
                )
            )

            if context_pred == actual:
                context_correct[
                    context
                ] += 1

            if int(neural_pred) == actual:
                neural_correct[
                    context
                ] += 1

    return (
        context_correct,
        neural_correct
    )


@torch.no_grad()
def hybrid_predict(
    model,
    data,
    train_end,
    boundaries,
    context_model,
    context_correct,
    neural_correct
):

    heldout = data[train_end:]

    if not heldout:
        return b""

    boundary_starts = build_boundary_set(
        boundaries
    )

    predictions = np.empty(
        len(heldout),
        dtype=np.uint8
    )

    contexts = []
    positions = []

    for i in range(
        len(heldout)
    ):

        global_index = (
            train_end + i
        )

        if global_index in boundary_starts:
            predictions[i] = 0
            continue

        start = (
            global_index
            - CONTEXT_SIZE
        )

        if start < 0:
            predictions[i] = data[
                global_index - 1
            ]
            continue

        contexts.append(
            data[
                start:global_index
            ]
        )

        positions.append(i)

    if contexts:

        X = np.frombuffer(
            b"".join(contexts),
            dtype=np.uint8
        ).reshape(
            -1,
            CONTEXT_SIZE
        ).astype(np.int64)

        X = torch.from_numpy(
            X
        ).to(DEVICE)

        logits = model(X)

        neural_preds = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy()

        for local_index, (
            output_position,
            neural_pred
        ) in enumerate(
            zip(
                positions,
                neural_preds
            )
        ):

            global_index = (
                train_end
                + output_position
            )

            context = tuple(
                data[
                    global_index
                    - CONTEXT_SIZE:
                    global_index
                ]
            )

            context_pred = (
                context_model.get(
                    context,
                    data[global_index - 1]
                )
            )

            c_score = (
                context_correct.get(
                    context,
                    0
                )
            )

            n_score = (
                neural_correct.get(
                    context,
                    0
                )
            )

            if n_score > c_score:
                predictions[
                    output_position
                ] = int(neural_pred)
            else:
                predictions[
                    output_position
                ] = context_pred

    actual = np.frombuffer(
        heldout,
        dtype=np.uint8
    )

    residual = (
        actual.astype(np.int16)
        -
        predictions.astype(np.int16)
    ) % 256

    return residual.astype(
        np.uint8
    ).tobytes()


# ============================================================
# RECONSTRUCTION
# ============================================================

@torch.no_grad()
def reconstruct_from_residual(
    data,
    train_end,
    residual,
    predictor_type,
    context_model=None,
    neural_model=None,
    boundaries=None,
    context_correct=None,
    neural_correct=None
):

    heldout_length = (
        len(data)
        - train_end
    )

    residual_arr = np.frombuffer(
        residual,
        dtype=np.uint8
    )

    if len(residual_arr) != heldout_length:
        raise RuntimeError(
            "Residual length does not "
            "match held-out length."
        )

    reconstructed = bytearray(
        data[:train_end]
    )

    boundary_starts = (
        build_boundary_set(
            boundaries
        )
    )

    for i in range(
        heldout_length
    ):

        global_index = (
            train_end + i
        )

        residual_value = int(
            residual_arr[i]
        )

        if global_index in boundary_starts:
            prediction = 0

        elif predictor_type == "previous":

            prediction = reconstructed[
                global_index - 1
            ]

        elif predictor_type == "context":

            start = (
                global_index
                - CONTEXT_SIZE
            )

            if start < 0:
                prediction = reconstructed[
                    global_index - 1
                ]
            else:

                context = tuple(
                    reconstructed[
                        start:global_index
                    ]
                )

                prediction = (
                    context_model.get(
                        context,
                        reconstructed[
                            global_index - 1
                        ]
                    )
                )

        elif predictor_type in (
            "neural",
            "hybrid"
        ):

            start = (
                global_index
                - CONTEXT_SIZE
            )

            if start < 0:

                prediction = reconstructed[
                    global_index - 1
                ]

            else:

                context_bytes = bytes(
                    reconstructed[
                        start:global_index
                    ]
                )

                context = tuple(
                    context_bytes
                )

                X = np.frombuffer(
                    context_bytes,
                    dtype=np.uint8
                ).astype(
                    np.int64
                ).reshape(
                    1,
                    -1
                )

                X = torch.from_numpy(
                    X
                ).to(DEVICE)

                logits = neural_model(X)

                neural_prediction = int(
                    torch.argmax(
                        logits,
                        dim=1
                    ).item()
                )

                if predictor_type == "neural":

                    prediction = (
                        neural_prediction
                    )

                else:

                    context_prediction = (
                        context_model.get(
                            context,
                            reconstructed[
                                global_index - 1
                            ]
                        )
                    )

                    c_score = (
                        context_correct.get(
                            context,
                            0
                        )
                    )

                    n_score = (
                        neural_correct.get(
                            context,
                            0
                        )
                    )

                    if n_score > c_score:
                        prediction = (
                            neural_prediction
                        )
                    else:
                        prediction = (
                            context_prediction
                        )

        else:
            raise ValueError(
                f"Unknown predictor: "
                f"{predictor_type}"
            )

        value = (
            prediction
            + residual_value
        ) % 256

        reconstructed.append(
            value
        )

    return bytes(
        reconstructed[train_end:]
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_residual(
    residual
):

    stats = residual_stats(
        residual
    )

    compressed = zlib_compress(
        residual
    )

    return {
        "residual_raw": len(residual),
        "residual_zlib": len(compressed),
        "entropy": stats["entropy"],
        "zero_percent": stats["zero_percent"],
        "mse": stats["mse"],
        "mae": stats["mae"],
        "max_error": stats["max_error"],
    }


def total_storage(
    residual_zlib,
    model_size,
    metadata_size
):

    return (
        residual_zlib
        + model_size
        + metadata_size
    )


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    data,
    boundaries,
    dataset_name,
    neural_model
):

    print()
    print("=" * 70)
    print(
        f"DATASET: {dataset_name}"
    )
    print("=" * 70)

    original_size = len(data)

    train_end = int(
        original_size
        * TRAIN_FRACTION
    )

    training = data[:train_end]
    heldout = data[train_end:]

    if len(heldout) <= CONTEXT_SIZE:
        print(
            "Held-out region too small."
        )
        return []

    original_hash = sha256(
        data
    )

    heldout_hash = sha256(
        heldout
    )

    zlib_full = zlib_compress(
        data
    )

    print()
    print(
        f"Original:     "
        f"{format_bytes(original_size)}"
    )

    print(
        f"Training:     "
        f"{format_bytes(len(training))}"
    )

    print(
        f"Held-out:     "
        f"{format_bytes(len(heldout))}"
    )

    print(
        f"Zlib full:    "
        f"{format_bytes(len(zlib_full))}"
    )

    print(
        f"Entropy:      "
        f"{entropy_bytes(data):.6f}"
    )

    print(
        f"SHA256:       "
        f"{original_hash}"
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    print()
    print(
        "Training context model..."
    )

    context_model = train_context_model(
        data,
        train_end,
        boundaries,
        CONTEXT_SIZE
    )

    context_model_size = (
        estimate_context_model_size(
            context_model
        )
    )

    print(
        f"Context entries: "
        f"{len(context_model):,}"
    )

    print(
        f"Context estimate: "
        f"{format_bytes(context_model_size)}"
    )

    # --------------------------------------------------------
    # Hybrid selector
    # --------------------------------------------------------

    context_correct, neural_correct = (
        train_hybrid_selector(
            data,
            train_end,
            boundaries,
            context_model,
            neural_model
        )
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print()
    print(
        "Generating residuals..."
    )

    boundary_starts = (
        build_boundary_set(
            boundaries
        )
    )

    previous_residual = previous_predict(
        data,
        train_end,
        boundary_starts
    )

    context_residual = context_predict(
        data,
        train_end,
        boundaries,
        context_model,
        CONTEXT_SIZE
    )

    neural_residual = neural_predict(
        neural_model,
        data,
        train_end,
        boundaries
    )

    hybrid_residual = hybrid_predict(
        neural_model,
        data,
        train_end,
        boundaries,
        context_model,
        context_correct,
        neural_correct
    )

    neural_size = neural_model_size(
        neural_model
    )

    methods = {
        "previous": (
            previous_residual,
            0
        ),

        "context": (
            context_residual,
            context_model_size
        ),

        "neural": (
            neural_residual,
            neural_size
        ),

        "hybrid": (
            hybrid_residual,
            neural_size
            + context_model_size
        ),
    }

    rows = []

    print()
    print(
        f"{'METHOD':<14}"
        f"{'RESIDUAL':>14}"
        f"{'MODEL':>14}"
        f"{'TOTAL':>14}"
        f"{'VS ZLIB':>12}"
    )

    print(
        f"{'zlib':<14}"
        f"{'--':>14}"
        f"{'--':>14}"
        f"{len(zlib_full):>14,}"
        f"{1.0000:>12.4f}x"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    for method_name, (
        residual,
        model_size
    ) in methods.items():

        metrics = evaluate_residual(
            residual
        )

        total = total_storage(
            metrics["residual_zlib"],
            model_size,
            METADATA_BYTES
        )

        ratio = (
            total / len(zlib_full)
        )

        reconstructed = (
            reconstruct_from_residual(
                data,
                train_end,
                residual,
                method_name,
                context_model=context_model,
                neural_model=neural_model,
                boundaries=boundaries,
                context_correct=context_correct,
                neural_correct=neural_correct
            )
        )

        reconstructed_hash = sha256(
            reconstructed
        )

        exact = (
            reconstructed
            == heldout
        )

        hash_match = (
            reconstructed_hash
            == heldout_hash
        )

        if not exact or not hash_match:

            print()
            print(
                "!!! RECONSTRUCTION FAILURE !!!"
            )

            print(
                f"Method: "
                f"{method_name}"
            )

            print(
                f"Expected: "
                f"{heldout_hash}"
            )

            print(
                f"Got:      "
                f"{reconstructed_hash}"
            )

            raise RuntimeError(
                f"{method_name} failed "
                f"exact reconstruction."
            )

        print(
            f"{method_name:<14}"
            f"{metrics['residual_zlib']:>14,}"
            f"{model_size:>14,}"
            f"{total:>14,}"
            f"{ratio:>12.4f}x"
        )

        rows.append({
            "dataset": dataset_name,
            "dataset_size": original_size,
            "training_size": len(training),
            "heldout_size": len(heldout),

            "zlib_size": len(zlib_full),

            "method": method_name,

            "residual_raw":
                metrics["residual_raw"],

            "residual_zlib":
                metrics["residual_zlib"],

            "model_size":
                model_size,

            "metadata_size":
                METADATA_BYTES,

            "total_size":
                total,

            "ratio_vs_zlib":
                ratio,

            "improvement_percent":
                (1.0 - ratio) * 100.0,

            "entropy":
                metrics["entropy"],

            "zero_percent":
                metrics["zero_percent"],

            "mse":
                metrics["mse"],

            "mae":
                metrics["mae"],

            "max_error":
                metrics["max_error"],

            "exact_reconstruction":
                exact,

            "hash_match":
                hash_match,

            "original_sha256":
                original_hash,

            "heldout_sha256":
                heldout_hash,

            "reconstructed_sha256":
                reconstructed_hash,

            "context_entries":
                len(context_model),

            "context_model_estimate":
                context_model_size,

            "neural_model_raw":
                neural_size,
        })

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    best = min(
        rows,
        key=lambda r:
        r["total_size"]
    )

    print()
    print(
        f"BEST: {best['method']}"
    )

    print(
        f"Total: "
        f"{format_bytes(best['total_size'])}"
    )

    print(
        f"Improvement vs zlib: "
        f"{best['improvement_percent']:.4f}%"
    )

    if best["improvement_percent"] > 0:
        print(
            "*** BEATS ZLIB ***"
        )
    else:
        print(
            "*** DOES NOT BEAT ZLIB ***"
        )

    return rows


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    files,
    root,
    corpus_size
):

    manifest = []

    for path in files:

        try:
            size = path.stat().st_size
        except OSError:
            size = None

        manifest.append({
            "path": str(
                path.relative_to(root)
            ),
            "size": size,
        })

    obj = {
        "root": str(root),
        "file_count": len(files),
        "corpus_size": corpus_size,
        "files": manifest,
    }

    with open(
        "test8_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            indent=2
        )


# ============================================================
# SCALING EXPERIMENT
# ============================================================

def run_scaling_experiment(
    files,
    root
):

    print()
    print("=" * 70)
    print(
        "TEST8 — REAL SOURCE-CODE "
        "PREDICTIVE COMPRESSION"
    )
    print("=" * 70)

    print()
    print(
        f"Source root: {root}"
    )

    print(
        f"Device:      {DEVICE}"
    )

    print(
        f"CPU threads: "
        f"{torch.get_num_threads()}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU:         "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU:         unavailable"
        )

    largest_target = max(
        TARGET_SIZES
    )

    print()
    print(
        "Building largest possible corpus..."
    )

    corpus, boundaries = build_corpus(
        files,
        root,
        target_size=largest_target
    )

    print(
        f"Corpus: "
        f"{format_bytes(len(corpus))}"
    )

    print(
        f"Files:  "
        f"{len(boundaries):,}"
    )

    if len(corpus) < 100_000:
        raise RuntimeError(
            "Less than 100 KB of usable "
            "source code was found."
        )

    save_manifest(
        files,
        root,
        len(corpus)
    )

    available_sizes = [
        size
        for size in TARGET_SIZES
        if size <= len(corpus)
    ]

    print()
    print(
        "Available scaling points:"
    )

    for size in available_sizes:

        print(
            f"  {format_bytes(size)}"
        )

    all_rows = []

    # --------------------------------------------------------
    # Each scaling point gets its OWN neural model.
    # --------------------------------------------------------

    for target_size in available_sizes:

        print()
        print("=" * 70)

        print(
            f"SCALING POINT: "
            f"{format_bytes(target_size)}"
        )

        print("=" * 70)

        data, data_boundaries = (
            build_corpus(
                files,
                root,
                target_size=target_size
            )
        )

        train_end = int(
            len(data)
            * TRAIN_FRACTION
        )

        model = train_neural_model(
            data,
            train_end,
            data_boundaries
        )

        rows = run_experiment(
            data,
            data_boundaries,
            f"REAL_SOURCE_CODE_{len(data)}",
            model
        )

        all_rows.extend(rows)

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return all_rows


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(rows):

    if not rows:
        return

    csv_path = (
        "test8_results.csv"
    )

    json_path = (
        "test8_results.json"
    )

    fields = list(
        rows[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            rows,
            f,
            indent=2
        )

    print()
    print(
        f"CSV:  {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(rows):

    print()
    print("=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)

    if not rows:
        print(
            "No results."
        )
        return

    datasets = sorted(
        set(
            r["dataset_size"]
            for r in rows
        )
    )

    methods = [
        "previous",
        "context",
        "neural",
        "hybrid",
    ]

    print()
    print(
        f"{'SIZE':<12}"
        f"{'ZLIB':>14}"
        f"{'PREV':>14}"
        f"{'CONTEXT':>14}"
        f"{'NEURAL':>14}"
        f"{'HYBRID':>14}"
    )

    for size in datasets:

        subset = [
            r
            for r in rows
            if r["dataset_size"] == size
        ]

        zlib_size = (
            subset[0]["zlib_size"]
        )

        values = {}

        for method in methods:

            match = next(
                (
                    r
                    for r in subset
                    if r["method"] == method
                ),
                None
            )

            values[method] = (
                match["total_size"]
                if match
                else None
            )

        print(
            f"{size:<12,}"
            f"{zlib_size:>14,}"
            f"{values['previous']:>14,}"
            f"{values['context']:>14,}"
            f"{values['neural']:>14,}"
            f"{values['hybrid']:>14,}"
        )

    # --------------------------------------------------------
    # Break-even
    # --------------------------------------------------------

    print()
    print(
        "BREAK-EVEN"
    )

    for method in methods:

        winning = [
            r
            for r in rows
            if r["method"] == method
            and r["total_size"]
            < r["zlib_size"]
        ]

        if winning:

            first = min(
                winning,
                key=lambda r:
                r["dataset_size"]
            )

            print(
                f"{method:<12}: "
                f"{format_bytes(first['dataset_size'])}"
            )

        else:

            print(
                f"{method:<12}: "
                f"not observed"
            )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    print()
    print(
        "IMPROVEMENT CURVE"
    )

    for size in datasets:

        print()
        print(
            format_bytes(size)
        )

        subset = [
            r
            for r in rows
            if r["dataset_size"] == size
        ]

        for method in methods:

            match = next(
                (
                    r
                    for r in subset
                    if r["method"] == method
                ),
                None
            )

            if match:

                print(
                    f"  {method:<10}"
                    f"{match['improvement_percent']:>10.3f}%"
                )

    # --------------------------------------------------------
    # Best
    # --------------------------------------------------------

    print()
    print(
        "BEST AT EACH SIZE"
    )

    for size in datasets:

        subset = [
            r
            for r in rows
            if r["dataset_size"] == size
        ]

        best = min(
            subset,
            key=lambda r:
            r["total_size"]
        )

        print(
            f"{format_bytes(size):>12}: "
            f"{best['method']:<8}"
            f" {format_bytes(best['total_size'])}"
            f" "
            f"({best['improvement_percent']:.2f}% "
            f"vs zlib)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("=" * 70)
    print(
        "TEST8 — REAL SOURCE-CODE "
        "PREDICTIVE COMPRESSION"
    )
    print("=" * 70)

    print()
    print(
        "REAL source code."
    )

    print(
        "75% training / 25% held-out."
    )

    print(
        "Exact reconstruction required."
    )

    print(
        "Model + residual + metadata counted."
    )

    print()

    files = discover_source_files(
        SOURCE_ROOT
    )

    if not files:

        print()
        print(
            "No supported source files found."
        )

        print(
            f"Put source files under:"
        )

        print(
            f"    {SOURCE_ROOT}"
        )

        return

    print()
    print(
        f"Discovered "
        f"{len(files):,} source files."
    )

    total_source_bytes = 0

    for path in files:

        try:
            total_source_bytes += (
                path.stat().st_size
            )
        except OSError:
            pass

    print(
        "Available source bytes: "
        f"{format_bytes(total_source_bytes)}"
    )

    print()
    print(
        "Largest files:"
    )

    largest = sorted(
        files,
        key=lambda p:
        p.stat().st_size,
        reverse=True
    )[:10]

    for path in largest:

        print(
            f"  "
            f"{format_bytes(path.stat().st_size):>12}"
            f"  "
            f"{path.relative_to(SOURCE_ROOT)}"
        )

    rows = run_scaling_experiment(
        files,
        SOURCE_ROOT
    )

    save_results(
        rows
    )

    print_summary(
        rows
    )

    elapsed = (
        time.time()
        - total_start
    )

    print()
    print("=" * 70)

    print(
        f"Total runtime: "
        f"{elapsed:.2f} seconds"
    )

    print("=" * 70)

    print()
    print(
        "TOTAL STORAGE ="
    )

    print(
        "    compressed residual"
        " + model"
        " + metadata"
    )

    print()
    print(
        "Every predictive method was"
    )

    print(
        "required to reconstruct the"
    )

    print(
        "held-out bytes exactly."
    )

    print()
    print(
        "BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()