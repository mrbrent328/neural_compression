#!/usr/bin/env python3

"""
TEST9 — ROCm GPU-OPTIMIZED REAL SOURCE-CODE PREDICTIVE COMPRESSION

Predictive compression benchmark against real source-code files.

EXPERIMENTAL RULES
------------------
- First 75% of each dataset is training data.
- Final 25% is completely held out.
- Held-out bytes are NEVER used to train the neural model.
- Every predictor must reconstruct the held-out bytes exactly.
- SHA-256 validation is performed.
- Model size + residual size + metadata are counted.
- zlib is used to compress residuals.
- Each scaling point receives its own neural model.

METHODS
-------
1. zlib baseline
2. previous-byte predictor
3. 4-byte context predictor
4. neural predictor
5. hybrid context + neural predictor

GPU
---
Designed for AMD ROCm PyTorch.

The ROCm stack exposes the AMD GPU through PyTorch's normal
"cuda" interface.

IMPORTANT:
Launch with:

    HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 test9.py

The override belongs in the shell environment, NOT in this file.

GPU strategy:
- Training samples are created on CPU.
- Training tensors are transferred to GPU in controlled batches.
- Neural model remains resident on GPU.
- Neural encoding/prediction is batched.
- Hybrid validation is batched.
- Exact decoder remains sequential because predictive decoding
  fundamentally depends on previously reconstructed bytes.
- GPU synchronization is used around timed GPU operations.
- No mixed precision is used; fp32 is deliberately retained for
  ROCm stability and reproducibility.
"""

# ============================================================
# IMPORTS
# ============================================================

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
    print("Install a ROCm-compatible PyTorch build first.")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path("/home/xetal/Downloads")

TRAIN_FRACTION = 0.75
CONTEXT_SIZE = 4

# ------------------------------------------------------------
# Neural model
# ------------------------------------------------------------

NEURAL_SAMPLES = 300_000
NEURAL_EPOCHS = 25

NEURAL_HIDDEN = 128
NEURAL_EMBED = 32

# Deliberately conservative for RX 6600 / ROCm stability.
NEURAL_BATCH = 4096

NEURAL_LR = 0.001

# GPU inference batches.
PREDICTION_BATCH = 16384

# Exact reconstruction is sequential.
# This is NOT used to incorrectly parallelize dependent decoding.
RECONSTRUCTION_BATCH = 1

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
    "test9.py",
    "test9_results.csv",
    "test9_results.json",
    "test9_manifest.json",
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
# GPU UTILITIES
# ============================================================

def gpu_available():
    return DEVICE.type == "cuda"


def gpu_sync():
    """
    Synchronize CUDA/ROCm before timing GPU work.
    """
    if gpu_available():
        torch.cuda.synchronize()


def gpu_memory_report():
    """
    Return current GPU memory statistics.
    """
    if not gpu_available():
        return {
            "allocated": 0,
            "reserved": 0,
            "peak_allocated": 0,
            "peak_reserved": 0,
        }

    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def reset_gpu_peak_memory():
    if gpu_available():
        torch.cuda.reset_peak_memory_stats()


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

    arr = np.frombuffer(
        data,
        dtype=np.uint8
    )

    counts = np.bincount(
        arr,
        minlength=256
    )

    probabilities = (
        counts[counts > 0]
        / len(arr)
    )

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
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
            np.mean(arr == 0)
            * 100.0
        ),

        "mse": float(
            np.mean(
                signed.astype(
                    np.float64
                ) ** 2
            )
        ),

        "mae": float(
            np.mean(
                np.abs(
                    signed.astype(
                        np.float64
                    )
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
            d
            for d in dirs
            if d not in SKIP_DIRS
            and not d.startswith(".git")
        ]

        current_path = Path(current_root)

        for filename in filenames:

            path = (
                current_path
                / filename
            )

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
                    "Skipping huge source file: "
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


def build_corpus(
    files,
    root,
    target_size=None
):

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

            remaining = (
                target_size
                - total
            )

            if remaining <= 0:
                break

            if len(data) > remaining:
                data = data[:remaining]

        start = total

        end = (
            total
            + len(data)
        )

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

    return (
        b"".join(chunks),
        boundaries
    )


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

        global_index = (
            train_end
            + i
        )

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

        if (
            end - start
            <= context_size
        ):
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
            counter
            .most_common(1)[0][0]
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

    boundary_starts = (
        build_boundary_set(
            boundaries
        )
    )

    predictions = np.empty(
        len(heldout),
        dtype=np.uint8
    )

    for i in range(len(heldout)):

        global_index = (
            train_end
            + i
        )

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

        predictions[i] = (
            context_model.get(
                context,
                data[global_index - 1]
            )
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

        self.context_size = (
            context_size
        )

        self.embedding = nn.Embedding(
            256,
            embed_size
        )

        self.fc1 = nn.Linear(
            context_size
            * embed_size,
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
        for tensor
        in model.state_dict().values()
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

    eligible_count = 0

    for b in boundaries:

        start = b["start"]

        end = min(
            b["end"],
            train_end
        )

        if (
            end - start
            > context_size
        ):

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

        upper = (
            cumulative
            + count
        )

        while (
            chosen_index
            < sample_count
            and chosen[chosen_index]
            < upper
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

    if gpu_available():

        print(
            "Backend:      "
            "ROCm / PyTorch CUDA interface"
        )

        print(
            f"GPU:          "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = (
            torch.cuda.get_device_properties(0)
        )

        print(
            f"VRAM:         "
            f"{format_bytes(props.total_memory)}"
        )

    else:

        print(
            "GPU:          unavailable"
        )

    print(
        f"Samples:      "
        f"{NEURAL_SAMPLES:,}"
    )

    print(
        f"Epochs:       "
        f"{NEURAL_EPOCHS}"
    )

    print(
        f"Batch:        "
        f"{NEURAL_BATCH:,}"
    )

    print(
        f"Learning rate: "
        f"{NEURAL_LR}"
    )

    # --------------------------------------------------------
    # Build compact training dataset on CPU.
    # --------------------------------------------------------

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

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    sample_count = len(X)

    model.train()

    reset_gpu_peak_memory()

    gpu_sync()

    start_time = time.time()

    for epoch in range(
        NEURAL_EPOCHS
    ):

        # Deterministic shuffle generated on CPU.
        permutation = torch.randperm(
            sample_count
        )

        running_loss = 0.0
        batches = 0

        for offset in range(
            0,
            sample_count,
            NEURAL_BATCH
        ):

            indices = permutation[
                offset:
                offset + NEURAL_BATCH
            ]

            xb = X_tensor.index_select(
                0,
                indices
            )

            yb = y_tensor.index_select(
                0,
                indices
            )

            xb = xb.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=False
            )

            yb = yb.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=False
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
                float(loss.detach().cpu())
            )

            batches += 1

            del xb
            del yb
            del logits
            del loss

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

    gpu_sync()

    elapsed = (
        time.time()
        - start_time
    )

    model.eval()

    print(
        f"Training time: "
        f"{elapsed:.2f} s"
    )

    if gpu_available():

        memory = gpu_memory_report()

        print(
            f"Peak GPU allocated: "
            f"{format_bytes(memory['peak_allocated'])}"
        )

        print(
            f"Peak GPU reserved:  "
            f"{format_bytes(memory['peak_reserved'])}"
        )

    print(
        f"Model size: "
        f"{neural_model_size(model):,} bytes"
    )

    del X
    del y
    del X_tensor
    del y_tensor
    del permutation

    return model


# ============================================================
# BATCHED NEURAL PREDICTION
# ============================================================

@torch.no_grad()
def neural_predict_batched(
    model,
    data,
    train_end,
    boundaries,
    batch_size=PREDICTION_BATCH
):

    heldout = data[train_end:]

    if not heldout:
        return b""

    boundary_starts = (
        build_boundary_set(
            boundaries
        )
    )

    heldout_length = len(
        heldout
    )

    predictions = np.empty(
        heldout_length,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Instead of constructing thousands of Python byte
    # objects and joining them, directly construct one compact
    # numpy matrix.
    # --------------------------------------------------------

    valid_positions = []

    for i in range(
        heldout_length
    ):

        global_index = (
            train_end
            + i
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

        valid_positions.append(i)

    model.eval()

    gpu_sync()

    start_time = time.time()

    total_gpu_samples = 0

    for offset in range(
        0,
        len(valid_positions),
        batch_size
    ):

        positions_batch = (
            valid_positions[
                offset:
                offset + batch_size
            ]
        )

        X = np.empty(
            (
                len(positions_batch),
                CONTEXT_SIZE
            ),
            dtype=np.int64
        )

        for row, output_position in enumerate(
            positions_batch
        ):

            global_index = (
                train_end
                + output_position
            )

            X[row] = np.frombuffer(
                data[
                    global_index
                    - CONTEXT_SIZE:
                    global_index
                ],
                dtype=np.uint8
            )

        X_tensor = torch.from_numpy(
            X
        ).to(
            DEVICE,
            dtype=torch.long
        )

        logits = model(
            X_tensor
        )

        preds = torch.argmax(
            logits,
            dim=1
        ).cpu().numpy().astype(
            np.uint8
        )

        predictions[
            positions_batch
        ] = preds

        total_gpu_samples += len(
            preds
        )

        del X
        del X_tensor
        del logits
        del preds

    gpu_sync()

    elapsed = (
        time.time()
        - start_time
    )

    if elapsed > 0:

        print(
            f"Neural GPU inference: "
            f"{total_gpu_samples:,} samples "
            f"in {elapsed:.2f}s "
            f"("
            f"{total_gpu_samples / elapsed:,.0f}"
            f" samples/s)"
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


# ============================================================
# HYBRID VALIDATION SELECTOR
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

    # Final 10% of training is validation.
    validation_start = int(
        train_end * 0.90
    )

    context_correct = Counter()
    neural_correct = Counter()

    neural_model.eval()

    validation_contexts = []
    validation_positions = []

    # --------------------------------------------------------
    # Collect validation contexts.
    # --------------------------------------------------------

    for b in boundaries:

        start = max(
            b["start"]
            + CONTEXT_SIZE,
            validation_start
        )

        end = min(
            b["end"],
            train_end
        )

        if end <= start:
            continue

        for pos in range(
            start,
            end
        ):

            validation_contexts.append(
                data[
                    pos - CONTEXT_SIZE:
                    pos
                ]
            )

            validation_positions.append(
                pos
            )

    # --------------------------------------------------------
    # GPU neural predictions.
    # --------------------------------------------------------

    gpu_sync()

    start_time = time.time()

    for offset in range(
        0,
        len(validation_contexts),
        PREDICTION_BATCH
    ):

        batch_contexts = (
            validation_contexts[
                offset:
                offset + PREDICTION_BATCH
            ]
        )

        X = np.empty(
            (
                len(batch_contexts),
                CONTEXT_SIZE
            ),
            dtype=np.int64
        )

        for row, context in enumerate(
            batch_contexts
        ):

            X[row] = np.frombuffer(
                context,
                dtype=np.uint8
            )

        X_tensor = torch.from_numpy(
            X
        ).to(
            DEVICE,
            dtype=torch.long
        )

        logits = neural_model(
            X_tensor
        )

        neural_preds = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
        )

        batch_positions = (
            validation_positions[
                offset:
                offset + len(neural_preds)
            ]
        )

        for pos, neural_pred in zip(
            batch_positions,
            neural_preds
        ):

            actual = data[pos]

            context = tuple(
                data[
                    pos - CONTEXT_SIZE:
                    pos
                ]
            )

            context_pred = (
                context_model.get(
                    context,
                    data[pos - 1]
                )
            )

            if (
                context_pred
                == actual
            ):

                context_correct[
                    context
                ] += 1

            if (
                int(neural_pred)
                == actual
            ):

                neural_correct[
                    context
                ] += 1

        del X
        del X_tensor
        del logits
        del neural_preds

    gpu_sync()

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Hybrid validation: "
        f"{len(validation_contexts):,} "
        f"samples in {elapsed:.2f}s"
    )

    del validation_contexts
    del validation_positions

    return (
        context_correct,
        neural_correct
    )


# ============================================================
# HYBRID PREDICTION
# ============================================================

@torch.no_grad()
def hybrid_predict_batched(
    model,
    data,
    train_end,
    boundaries,
    context_model,
    context_correct,
    neural_correct,
    batch_size=PREDICTION_BATCH
):

    heldout = data[train_end:]

    if not heldout:
        return b""

    boundary_starts = (
        build_boundary_set(
            boundaries
        )
    )

    predictions = np.empty(
        len(heldout),
        dtype=np.uint8
    )

    valid_positions = []

    for i in range(
        len(heldout)
    ):

        global_index = (
            train_end
            + i
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

        valid_positions.append(i)

    model.eval()

    gpu_sync()

    start_time = time.time()

    for offset in range(
        0,
        len(valid_positions),
        batch_size
    ):

        positions_batch = (
            valid_positions[
                offset:
                offset + batch_size
            ]
        )

        X = np.empty(
            (
                len(positions_batch),
                CONTEXT_SIZE
            ),
            dtype=np.int64
        )

        contexts = []

        for row, output_position in enumerate(
            positions_batch
        ):

            global_index = (
                train_end
                + output_position
            )

            context = data[
                global_index
                - CONTEXT_SIZE:
                global_index
            ]

            contexts.append(context)

            X[row] = np.frombuffer(
                context,
                dtype=np.uint8
            )

        X_tensor = torch.from_numpy(
            X
        ).to(
            DEVICE,
            dtype=torch.long
        )

        logits = model(
            X_tensor
        )

        neural_preds = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
        )

        for row, (
            output_position,
            neural_pred,
            raw_context
        ) in enumerate(
            zip(
                positions_batch,
                neural_preds,
                contexts
            )
        ):

            global_index = (
                train_end
                + output_position
            )

            context = tuple(
                raw_context
            )

            context_prediction = (
                context_model.get(
                    context,
                    data[
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

                predictions[
                    output_position
                ] = int(neural_pred)

            else:

                predictions[
                    output_position
                ] = context_prediction

        del X
        del X_tensor
        del logits
        del neural_preds

    gpu_sync()

    elapsed = (
        time.time()
        - start_time
    )

    if elapsed > 0:

        print(
            f"Hybrid GPU inference: "
            f"{len(valid_positions):,} samples "
            f"in {elapsed:.2f}s "
            f"("
            f"{len(valid_positions) / elapsed:,.0f}"
            f" samples/s)"
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


# ============================================================
# SEQUENTIAL EXACT RECONSTRUCTION
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

    if (
        len(residual_arr)
        != heldout_length
    ):

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

    # --------------------------------------------------------
    # Previous byte
    # --------------------------------------------------------

    if predictor_type == "previous":

        for i in range(
            heldout_length
        ):

            global_index = (
                train_end
                + i
            )

            if global_index in boundary_starts:

                prediction = 0

            else:

                prediction = reconstructed[
                    global_index - 1
                ]

            value = (
                prediction
                + int(
                    residual_arr[i]
                )
            ) % 256

            reconstructed.append(
                value
            )

        return bytes(
            reconstructed[train_end:]
        )

    # --------------------------------------------------------
    # Context predictor
    # --------------------------------------------------------

    if predictor_type == "context":

        for i in range(
            heldout_length
        ):

            global_index = (
                train_end
                + i
            )

            if global_index in boundary_starts:

                prediction = 0

            else:

                start = (
                    global_index
                    - CONTEXT_SIZE
                )

                if start < 0:

                    prediction = (
                        reconstructed[
                            global_index - 1
                        ]
                    )

                else:

                    context = tuple(
                        reconstructed[
                            start:
                            global_index
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

            value = (
                prediction
                + int(
                    residual_arr[i]
                )
            ) % 256

            reconstructed.append(
                value
            )

        return bytes(
            reconstructed[train_end:]
        )

    # --------------------------------------------------------
    # Neural / hybrid decoder.
    #
    # IMPORTANT:
    #
    # This is intentionally sequential.
    #
    # We CANNOT safely batch these predictions because:
    #
    # reconstructed[N]
    #       ↓
    # context[N+1]
    #       ↓
    # prediction[N+1]
    #
    # The next context depends on the byte we just decoded.
    #
    # Encoding can be massively GPU-batched.
    # Exact predictive decoding cannot.
    # --------------------------------------------------------

    if predictor_type not in (
        "neural",
        "hybrid"
    ):

        raise ValueError(
            f"Unknown predictor: "
            f"{predictor_type}"
        )

    neural_model.eval()

    total = heldout_length

    gpu_sync()

    start_time = time.time()

    # --------------------------------------------------------
    # Reuse a single GPU tensor for inference.
    # This avoids thousands of GPU allocations.
    # --------------------------------------------------------

    input_buffer = torch.empty(
        (
            1,
            CONTEXT_SIZE
        ),
        dtype=torch.long,
        device=DEVICE
    )

    decoded_count = 0

    for i in range(total):

        global_index = (
            train_end
            + i
        )

        if global_index in boundary_starts:

            prediction = 0

        else:

            start = (
                global_index
                - CONTEXT_SIZE
            )

            if start < 0:

                prediction = (
                    reconstructed[
                        global_index - 1
                    ]
                )

            else:

                # --------------------------------------------
                # Construct context from already decoded bytes.
                # --------------------------------------------

                context_bytes = (
                    reconstructed[
                        start:
                        global_index
                    ]
                )

                context_np = np.frombuffer(
                    bytes(context_bytes),
                    dtype=np.uint8
                ).astype(
                    np.int64
                )

                input_buffer[0].copy_(
                    torch.from_numpy(
                        context_np
                    )
                )

                logits = neural_model(
                    input_buffer
                )

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

                    context = tuple(
                        context_bytes
                    )

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

                del logits

        value = (
            int(prediction)
            + int(
                residual_arr[i]
            )
        ) % 256

        reconstructed.append(
            value
        )

        decoded_count += 1

        # Periodic progress without synchronizing every byte.
        if (
            decoded_count % 100_000
            == 0
        ):

            print(
                f"  Decoded "
                f"{decoded_count:,}/"
                f"{total:,}",
                end="\r",
                flush=True
            )

    gpu_sync()

    print()

    elapsed = (
        time.time()
        - start_time
    )

    if elapsed > 0:

        print(
            f"{predictor_type} "
            f"reconstruction: "
            f"{heldout_length:,} bytes "
            f"in {elapsed:.2f}s "
            f"("
            f"{heldout_length / elapsed:,.0f}"
            f" bytes/s)"
        )

    del input_buffer

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
        "residual_raw":
            len(residual),

        "residual_zlib":
            len(compressed),

        "entropy":
            stats["entropy"],

        "zero_percent":
            stats["zero_percent"],

        "mse":
            stats["mse"],

        "mae":
            stats["mae"],

        "max_error":
            stats["max_error"],
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

    if (
        len(heldout)
        <= CONTEXT_SIZE
    ):

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
    # Context model
    # --------------------------------------------------------

    print()
    print(
        "Training context model..."
    )

    context_model = (
        train_context_model(
            data,
            train_end,
            boundaries,
            CONTEXT_SIZE
        )
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

    # --------------------------------------------------------
    # Previous
    # --------------------------------------------------------

    previous_start = time.time()

    previous_residual = previous_predict(
        data,
        train_end,
        boundary_starts
    )

    print(
        f"Previous predictor: "
        f"{time.time() - previous_start:.2f}s"
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_start = time.time()

    context_residual = context_predict(
        data,
        train_end,
        boundaries,
        context_model,
        CONTEXT_SIZE
    )

    print(
        f"Context predictor:  "
        f"{time.time() - context_start:.2f}s"
    )

    # --------------------------------------------------------
    # Neural GPU path
    # --------------------------------------------------------

    neural_residual = (
        neural_predict_batched(
            neural_model,
            data,
            train_end,
            boundaries
        )
    )

    # --------------------------------------------------------
    # Hybrid GPU path
    # --------------------------------------------------------

    hybrid_residual = (
        hybrid_predict_batched(
            neural_model,
            data,
            train_end,
            boundaries,
            context_model,
            context_correct,
            neural_correct
        )
    )

    neural_size = (
        neural_model_size(
            neural_model
        )
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
    # Verify every method.
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
            total
            / len(zlib_full)
        )

        print()
        print(
            f"Verifying {method_name}..."
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

        if (
            not exact
            or not hash_match
        ):

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

            "dataset":
                dataset_name,

            "dataset_size":
                original_size,

            "training_size":
                len(training),

            "heldout_size":
                len(heldout),

            "zlib_size":
                len(zlib_full),

            "method":
                method_name,

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
                (1.0 - ratio)
                * 100.0,

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

        del reconstructed

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

    if (
        best["improvement_percent"]
        > 0
    ):

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

        "file_count":
            len(files),

        "corpus_size":
            corpus_size,

        "files":
            manifest,
    }

    with open(
        "test9_manifest.json",
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
        "TEST9 — ROCm GPU-OPTIMIZED "
        "REAL SOURCE-CODE "
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

    if gpu_available():

        print(
            f"GPU:         "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "GPU backend: "
            "ROCm / PyTorch CUDA"
        )

        props = (
            torch.cuda.get_device_properties(0)
        )

        print(
            f"GPU VRAM:    "
            f"{format_bytes(props.total_memory)}"
        )

        print(
            "GPU mode:    "
            "FP32 / conservative batches"
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
    # Each scaling point gets its own model.
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

        all_rows.extend(
            rows
        )

        del model

        if gpu_available():

            gpu_sync()

            torch.cuda.empty_cache()

    return all_rows


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(rows):

    if not rows:
        return

    csv_path = (
        "test9_results.csv"
    )

    json_path = (
        "test9_results.json"
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

        def fmt(value):

            if value is None:
                return "N/A"

            return f"{value:,}"

        print(
            f"{size:<12,}"
            f"{zlib_size:>14,}"
            f"{fmt(values['previous']):>14}"
            f"{fmt(values['context']):>14}"
            f"{fmt(values['neural']):>14}"
            f"{fmt(values['hybrid']):>14}"
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
            f" "
            f"{format_bytes(best['total_size'])}"
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
        "TEST9 — ROCm GPU-OPTIMIZED "
        "REAL SOURCE-CODE "
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

    print(
        "Neural workload GPU-optimized."
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
            "Put source files under:"
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