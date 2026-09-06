#!/usr/bin/env python3

"""
LEARNED STRUCTURE STORAGE — SELF-CONTAINED BENCHMARK

Tests three locally generated datasets:

1. Structured natural-ish text
2. Source-code-like text
3. Mathematically generated data

Tests:

    ZLIB
    Previous-byte predictor
    Context predictor
    Neural predictor
    Hybrid predictor

All representations are lossless.

The neural model is trained only on the training section.
The held-out section is never used for training.

IMPORTANT:
Every byte of residual data, model data, and metadata is counted.

This is an experimental research benchmark.
"""

import sys
import math
import time
import zlib
import hashlib
import random
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch is required.")
    print()
    print("Install with:")
    print("    pip install torch numpy")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 12345

TOTAL_SIZE = 1_000_000

TRAIN_FRACTION = 0.75

NEURAL_SAMPLES = 300_000

EPOCHS = 1500

PRINT_EVERY = 100

CONTEXT_SIZE = 4

HIDDEN_SIZE = 64

BATCH_SIZE = 4096

LEARNING_RATE = 0.001


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# BASIC UTILITIES
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def bytes_to_array(data):
    """
    IMPORTANT:

    Always use frombuffer for Python bytes.

    np.asarray(bytes, dtype=np.uint8) attempts to
    interpret the entire bytes object as a scalar.

    np.frombuffer correctly exposes each byte.
    """

    if isinstance(data, np.ndarray):
        return data.astype(
            np.uint8,
            copy=False
        )

    return np.frombuffer(
        data,
        dtype=np.uint8
    )


def entropy(data):
    """
    Shannon entropy in bits per byte.
    """

    arr = bytes_to_array(data)

    if len(arr) == 0:
        return 0.0

    counts = np.bincount(
        arr,
        minlength=256
    )

    probs = counts[counts > 0] / len(arr)

    return float(
        -np.sum(
            probs * np.log2(probs)
        )
    )


def compress_zlib(data):
    return zlib.compress(
        bytes(data),
        level=9
    )


def format_bytes(n):

    if n < 1024:
        return f"{n:,} B"

    if n < 1024 ** 2:
        return f"{n / 1024:.2f} KB"

    if n < 1024 ** 3:
        return f"{n / (1024 ** 2):.2f} MB"

    return f"{n / (1024 ** 3):.2f} GB"


# ============================================================
# DATASET GENERATORS
# ============================================================

WORDS = [
    "the",
    "and",
    "of",
    "to",
    "a",
    "in",
    "that",
    "is",
    "for",
    "with",
    "as",
    "on",
    "was",
    "this",
    "by",
    "from",
    "it",
    "an",
    "be",
    "are",
    "or",
    "not",
    "data",
    "system",
    "model",
    "information",
    "structure",
    "storage",
    "network",
    "process",
    "signal",
    "memory",
    "computer",
    "machine",
    "learning",
    "pattern",
    "state",
    "value",
    "representation",
    "function",
    "algorithm"
]


def generate_text(size):

    rng = random.Random(SEED)

    output = []

    total = 0

    while total < size:

        sentences = []

        for _ in range(
            rng.randint(5, 15)
        ):

            length = rng.randint(
                8,
                20
            )

            words = [
                rng.choice(WORDS)
                for _ in range(length)
            ]

            sentence = (
                " ".join(words).capitalize()
                + "."
            )

            sentences.append(
                sentence
            )

        paragraph = (
            " ".join(sentences)
            + "\n\n"
        )

        output.append(paragraph)

        total += len(paragraph)

    return "".join(output).encode(
        "utf-8"
    )[:size]


def generate_code(size):

    rng = random.Random(SEED)

    lines = []

    total = 0

    names = [
        "value",
        "state",
        "result",
        "data",
        "index",
        "counter",
        "buffer",
        "model",
        "node",
        "input",
        "output"
    ]

    while total < size:

        name = rng.choice(names)

        value = rng.randint(
            0,
            100000
        )

        kind = rng.randint(
            0,
            5
        )

        if kind == 0:

            line = (
                f"int {name} = {value};\n"
            )

        elif kind == 1:

            line = (
                f"{name} = "
                f"{value} * 2 + 1;\n"
            )

        elif kind == 2:

            line = (
                f"if ({name} > "
                f"{value // 2}) "
                f"{{ {name} -= 1; }}\n"
            )

        elif kind == 3:

            line = (
                f"for (int i = 0; "
                f"i < {value % 100}; "
                f"i++) "
                f"{{ {name} += i; }}\n"
            )

        elif kind == 4:

            line = (
                f"buffer[{value % 1024}] "
                f"= {name} & 0xFF;\n"
            )

        else:

            line = (
                f"return process_{name}"
                f"({name}, {value});\n"
            )

        lines.append(line)

        total += len(line)

    return "".join(lines).encode(
        "utf-8"
    )[:size]


def generate_math(size):

    """
    Deterministic mathematical signal.

    This is deliberately generated from a
    relatively small mathematical description.
    """

    output = bytearray()

    i = 0

    while len(output) < size:

        x = i / 1000.0

        value = (
            math.sin(
                x * 0.37
            )
            + 0.5 * math.sin(
                x * 1.91
            )
            + 0.25 * math.cos(
                x * 4.17
            )
            + 0.1 * math.sin(
                x * 13.7
            )
        )

        normalized = int(
            ((value + 2.0) / 4.0)
            * 255
        )

        normalized = max(
            0,
            min(
                255,
                normalized
            )
        )

        output.append(
            normalized
        )

        i += 1

    return bytes(output)


# ============================================================
# PREVIOUS BYTE PREDICTOR
# ============================================================

def previous_byte_predict(data):

    arr = bytes_to_array(data)

    predictions = np.empty_like(
        arr
    )

    predictions[0] = 0

    if len(arr) > 1:

        predictions[1:] = (
            arr[:-1]
        )

    return predictions


def previous_byte_predict_range(
    full_data,
    start,
    end
):

    """
    Predict a held-out range.

    The first held-out byte is predicted
    from the final training byte.

    This avoids incorrectly resetting the
    predictor at the train/test boundary.
    """

    arr = bytes_to_array(
        full_data
    )

    if start >= end:

        return np.empty(
            0,
            dtype=np.uint8
        )

    predictions = np.empty(
        end - start,
        dtype=np.uint8
    )

    for j, i in enumerate(
        range(start, end)
    ):

        if i == 0:

            predictions[j] = 0

        else:

            predictions[j] = (
                arr[i - 1]
            )

    return predictions


# ============================================================
# CONTEXT PREDICTOR
# ============================================================

def build_context_model(
    data,
    context_size
):

    arr = bytes_to_array(
        data
    )

    counts = {}

    if len(arr) <= context_size:
        return {}

    for i in range(
        context_size,
        len(arr)
    ):

        context = bytes(
            arr[
                i - context_size:i
            ]
        )

        value = int(
            arr[i]
        )

        if context not in counts:

            counts[context] = Counter()

        counts[context][value] += 1

    model = {}

    for context, counter in counts.items():

        model[context] = (
            counter.most_common(1)[0][0]
        )

    return model


def context_predict_range(
    full_data,
    start,
    end,
    context_model,
    context_size
):

    arr = bytes_to_array(
        full_data
    )

    predictions = np.zeros(
        end - start,
        dtype=np.uint8
    )

    for j, i in enumerate(
        range(start, end)
    ):

        if i < context_size:

            predictions[j] = 0

            continue

        context = bytes(
            arr[
                i - context_size:i
            ]
        )

        predictions[j] = (
            context_model.get(
                context,
                0
            )
        )

    return predictions


# ============================================================
# NEURAL NETWORK
# ============================================================

class BytePredictor(nn.Module):

    def __init__(
        self,
        context_size,
        hidden_size
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                context_size,
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

        return self.net(x)


# ============================================================
# NEURAL TRAINING DATA
# ============================================================

def make_neural_samples(
    data,
    start,
    end,
    context_size,
    max_samples
):

    arr = bytes_to_array(
        data
    )

    possible = (
        end
        - start
        - context_size
    )

    if possible <= 0:

        raise ValueError(
            "Dataset too small."
        )

    count = min(
        max_samples,
        possible
    )

    indices = np.linspace(
        start + context_size,
        end - 1,
        count,
        dtype=np.int64
    )

    contexts = np.empty(
        (
            len(indices),
            context_size
        ),
        dtype=np.float32
    )

    targets = np.empty(
        len(indices),
        dtype=np.int64
    )

    for j, i in enumerate(
        indices
    ):

        contexts[j] = (
            arr[
                i - context_size:i
            ].astype(
                np.float32
            )
            / 255.0
        )

        targets[j] = int(
            arr[i]
        )

    return contexts, targets


# ============================================================
# TRAIN NEURAL MODEL
# ============================================================

def train_neural_model(
    data,
    train_end
):

    print()
    print("=" * 70)
    print("NEURAL PREDICTOR")
    print("=" * 70)

    X, y = make_neural_samples(
        data,
        0,
        train_end,
        CONTEXT_SIZE,
        NEURAL_SAMPLES
    )

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.long
    )

    dataset = torch.utils.data.TensorDataset(
        X,
        y
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = BytePredictor(
        CONTEXT_SIZE,
        HIDDEN_SIZE
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Training samples:",
        len(X)
    )

    print(
        "Parameters:",
        parameter_count
    )

    start_time = time.time()

    model.train()

    for epoch in range(
        EPOCHS
    ):

        total_loss = 0.0

        batches = 0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                batch_x
            )

            loss = loss_function(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            batches += 1

        if (
            epoch % PRINT_EVERY == 0
            or epoch == EPOCHS - 1
        ):

            average_loss = (
                total_loss / batches
            )

            print(
                f"Epoch {epoch:5d} | "
                f"Loss: "
                f"{average_loss:.8f}"
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
# NEURAL PREDICTION
# ============================================================

def neural_predict_range(
    model,
    full_data,
    start,
    end
):

    arr = bytes_to_array(
        full_data
    )

    predictions = np.zeros(
        end - start,
        dtype=np.uint8
    )

    model.eval()

    with torch.no_grad():

        for batch_start in range(
            start,
            end,
            BATCH_SIZE
        ):

            batch_end = min(
                batch_start + BATCH_SIZE,
                end
            )

            indices = np.arange(
                batch_start,
                batch_end
            )

            valid_indices = (
                indices[
                    indices >= CONTEXT_SIZE
                ]
            )

            if len(valid_indices) == 0:
                continue

            X = np.empty(
                (
                    len(valid_indices),
                    CONTEXT_SIZE
                ),
                dtype=np.float32
            )

            for j, i in enumerate(
                valid_indices
            ):

                X[j] = (
                    arr[
                        i - CONTEXT_SIZE:i
                    ].astype(
                        np.float32
                    )
                    / 255.0
                )

            X_tensor = torch.tensor(
                X,
                dtype=torch.float32
            ).to(DEVICE)

            logits = model(
                X_tensor
            )

            predicted = torch.argmax(
                logits,
                dim=1
            )

            predicted = (
                predicted
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            for j, i in enumerate(
                valid_indices
            ):

                predictions[
                    i - start
                ] = predicted[j]

    return predictions


# ============================================================
# RESIDUAL
# ============================================================

def make_residual(
    actual,
    predictions
):

    actual = bytes_to_array(
        actual
    )

    predictions = bytes_to_array(
        predictions
    )

    if len(actual) != len(
        predictions
    ):

        raise ValueError(
            "Actual and prediction lengths "
            "do not match."
        )

    residual = (
        actual.astype(np.int16)
        - predictions.astype(np.int16)
    )

    residual %= 256

    return residual.astype(
        np.uint8
    )


# ============================================================
# RECONSTRUCTION
# ============================================================

def reconstruct(
    predictions,
    residual
):

    predictions = bytes_to_array(
        predictions
    )

    residual = bytes_to_array(
        residual
    )

    if len(predictions) != len(
        residual
    ):

        raise ValueError(
            "Prediction and residual "
            "lengths do not match."
        )

    reconstructed = (
        predictions.astype(np.uint16)
        + residual.astype(np.uint16)
    ) % 256

    return reconstructed.astype(
        np.uint8
    )


# ============================================================
# MODEL STORAGE
# ============================================================

def serialized_model_size(
    model
):

    raw = bytearray()

    state = model.state_dict()

    for key in sorted(
        state.keys()
    ):

        tensor = (
            state[key]
            .detach()
            .cpu()
        )

        raw.extend(
            tensor.numpy().tobytes()
        )

    compressed = zlib.compress(
        bytes(raw),
        level=9
    )

    return (
        len(raw),
        len(compressed)
    )


# ============================================================
# PREDICTOR EVALUATION
# ============================================================

def evaluate_predictor(
    name,
    actual,
    predictions
):

    actual_arr = bytes_to_array(
        actual
    )

    predictions_arr = bytes_to_array(
        predictions
    )

    residual = make_residual(
        actual_arr,
        predictions_arr
    )

    residual_bytes = residual.tobytes()

    compressed = compress_zlib(
        residual_bytes
    )

    reconstructed = reconstruct(
        predictions_arr,
        residual
    )

    exact = np.array_equal(
        actual_arr,
        reconstructed
    )

    diff = (
        actual_arr.astype(np.int16)
        - predictions_arr.astype(np.int16)
    )

    mse = float(
        np.mean(
            diff.astype(
                np.float64
            ) ** 2
        )
    )

    mae = float(
        np.mean(
            np.abs(diff)
        )
    )

    max_error = int(
        np.max(
            np.abs(diff)
        )
    )

    zero_fraction = float(
        np.mean(
            residual == 0
        )
    )

    result = {

        "name": name,

        "entropy": entropy(
            residual_bytes
        ),

        "zero_fraction":
            zero_fraction,

        "raw":
            len(residual_bytes),

        "compressed":
            len(compressed),

        "mse":
            mse,

        "mae":
            mae,

        "max_error":
            max_error,

        "exact":
            exact,

        "residual":
            residual_bytes
    }

    print()
    print(name)

    print(
        f"Entropy:              "
        f"{result['entropy']:.6f} bits/byte"
    )

    print(
        f"Zero residual:        "
        f"{zero_fraction * 100:.3f}%"
    )

    print(
        f"Residual raw:         "
        f"{result['raw']:,} bytes"
    )

    print(
        f"Residual zlib:        "
        f"{result['compressed']:,} bytes"
    )

    print(
        f"Prediction MSE:       "
        f"{mse:.6f}"
    )

    print(
        f"Prediction MAE:       "
        f"{mae:.6f}"
    )

    print(
        f"Maximum error:        "
        f"{max_error}"
    )

    print(
        "Exact reconstruction: ",
        exact
    )

    return result


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(
    name,
    data
):

    print()
    print("#" * 70)
    print("DATASET:", name)
    print("#" * 70)

    original_size = len(data)

    original_hash = sha256(
        data
    )

    print()
    print("Original size:")
    print(
        f"  {original_size:,} bytes"
    )

    print("SHA256:")
    print(
        f"  {original_hash}"
    )

    # --------------------------------------------------------
    # ZLIB
    # --------------------------------------------------------

    zlib_data = compress_zlib(
        data
    )

    print()
    print("ZLIB BASELINE")
    print(
        f"Compressed: "
        f"{len(zlib_data):,} bytes"
    )

    print(
        f"Ratio: "
        f"{len(zlib_data) / original_size:.4f}x"
    )

    print(
        f"Entropy: "
        f"{entropy(data):.6f} bits/byte"
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_end = int(
        original_size
        * TRAIN_FRACTION
    )

    test_start = train_end

    test_data = data[
        test_start:
    ]

    print()
    print("HELD-OUT TEST")
    print(
        f"Training bytes: "
        f"{train_end:,}"
    )

    print(
        f"Held-out bytes: "
        f"{len(test_data):,}"
    )

    # --------------------------------------------------------
    # PREVIOUS BYTE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREVIOUS BYTE PREDICTOR")
    print("=" * 70)

    previous_predictions = (
        previous_byte_predict_range(
            data,
            test_start,
            original_size
        )
    )

    previous_result = (
        evaluate_predictor(
            "Previous-byte",
            test_data,
            previous_predictions
        )
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONTEXT PREDICTOR")
    print("=" * 70)

    context_model = (
        build_context_model(
            data[:train_end],
            CONTEXT_SIZE
        )
    )

    print(
        "Unique learned contexts:",
        f"{len(context_model):,}"
    )

    context_predictions = (
        context_predict_range(
            data,
            test_start,
            original_size,
            context_model,
            CONTEXT_SIZE
        )
    )

    context_result = (
        evaluate_predictor(
            "Context",
            test_data,
            context_predictions
        )
    )

    # --------------------------------------------------------
    # NEURAL
    # --------------------------------------------------------

    model = train_neural_model(
        data,
        train_end
    )

    neural_predictions = (
        neural_predict_range(
            model,
            data,
            test_start,
            original_size
        )
    )

    neural_result = (
        evaluate_predictor(
            "Neural",
            test_data,
            neural_predictions
        )
    )

    raw_model_size, compressed_model_size = (
        serialized_model_size(
            model
        )
    )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HYBRID PREDICTOR")
    print("=" * 70)

    """
    Simple hybrid:

        average(previous byte, neural prediction)

    This is intentionally simple.
    """

    hybrid_predictions = (
        (
            previous_predictions.astype(
                np.uint16
            )
            +
            neural_predictions.astype(
                np.uint16
            )
        )
        // 2
    ).astype(
        np.uint8
    )

    hybrid_result = (
        evaluate_predictor(
            "Hybrid",
            test_data,
            hybrid_predictions
        )
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STORAGE ACCOUNTING")
    print("=" * 70)

    metadata_size = 1024

    neural_total = (
        neural_result["compressed"]
        + compressed_model_size
        + metadata_size
    )

    hybrid_total = (
        hybrid_result["compressed"]
        + compressed_model_size
        + metadata_size
    )

    previous_total = (
        previous_result["compressed"]
    )

    context_model_bytes = (
        len(context_model)
        * (
            CONTEXT_SIZE
            + 1
        )
    )

    context_total = (
        context_result["compressed"]
        + context_model_bytes
        + metadata_size
    )

    print(
        f"Original dataset:       "
        f"{original_size:,}"
    )

    print(
        f"Zlib entire dataset:    "
        f"{len(zlib_data):,}"
    )

    print()

    print(
        f"Previous residual:      "
        f"{previous_result['compressed']:,}"
    )

    print(
        f"Context residual:       "
        f"{context_result['compressed']:,}"
    )

    print(
        f"Neural residual:        "
        f"{neural_result['compressed']:,}"
    )

    print(
        f"Hybrid residual:        "
        f"{hybrid_result['compressed']:,}"
    )

    print()

    print(
        f"Neural model raw:       "
        f"{raw_model_size:,}"
    )

    print(
        f"Neural model compressed:"
        f" {compressed_model_size:,}"
    )

    print(
        f"Context model estimate: "
        f"{context_model_bytes:,}"
    )

    print(
        f"Metadata:               "
        f"{metadata_size:,}"
    )

    print()

    print(
        f"Previous TOTAL:         "
        f"{previous_total:,}"
    )

    print(
        f"Context TOTAL:          "
        f"{context_total:,}"
    )

    print(
        f"Neural TOTAL:           "
        f"{neural_total:,}"
    )

    print(
        f"Hybrid TOTAL:           "
        f"{hybrid_total:,}"
    )

    # --------------------------------------------------------
    # EXACT RECONSTRUCTION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXACT RECONSTRUCTION")
    print("=" * 70)

    results = [
        previous_result,
        context_result,
        neural_result,
        hybrid_result
    ]

    for result in results:

        print(
            f"{result['name']:20s}: "
            f"{result['exact']}"
        )

    # --------------------------------------------------------
    # HASH CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HELD-OUT HASH CHECK")
    print("=" * 70)

    print(
        "Held-out original SHA256:"
    )

    print(
        sha256(
            test_data
        )
    )

    for result in results:

        residual_arr = np.frombuffer(
            result["residual"],
            dtype=np.uint8
        )

        if result["name"] == "Previous-byte":

            pred = previous_predictions

        elif result["name"] == "Context":

            pred = context_predictions

        elif result["name"] == "Neural":

            pred = neural_predictions

        else:

            pred = hybrid_predictions

        reconstructed = reconstruct(
            pred,
            residual_arr
        )

        reconstructed_hash = sha256(
            reconstructed.tobytes()
        )

        print()
        print(
            f"{result['name']} SHA256:"
        )

        print(
            reconstructed_hash
        )

        print(
            "HASH MATCH:",
            reconstructed_hash
            == sha256(test_data)
        )

    # --------------------------------------------------------
    # FINAL TABLE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Method':<18}"
        f"{'Residual':>14}"
        f"{'Total':>14}"
        f"{'vs zlib':>12}"
    )

    print("-" * 60)

    methods = [

        (
            "Zlib",
            len(zlib_data),
            len(zlib_data)
        ),

        (
            "Previous",
            previous_result["compressed"],
            previous_total
        ),

        (
            "Context",
            context_result["compressed"],
            context_total
        ),

        (
            "Neural",
            neural_result["compressed"],
            neural_total
        ),

        (
            "Hybrid",
            hybrid_result["compressed"],
            hybrid_total
        )
    ]

    for method, residual_size, total in methods:

        ratio = (
            total
            / len(zlib_data)
        )

        print(
            f"{method:<18}"
            f"{residual_size:>14,}"
            f"{total:>14,}"
            f"{ratio:>11.4f}x"
        )

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best = min(
        methods,
        key=lambda x: x[2]
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Best method: "
        f"{best[0]}"
    )

    print(
        f"Best storage: "
        f"{best[2]:,} bytes"
    )

    zlib_size = len(
        zlib_data
    )

    if best[2] < zlib_size:

        improvement = (
            1.0
            - (
                best[2]
                / zlib_size
            )
        ) * 100.0

        print(
            f"Improvement over zlib: "
            f"{improvement:.4f}%"
        )

        print()
        print(
            "*** BEATS ZLIB ***"
        )

    else:

        difference = (
            best[2]
            - zlib_size
        )

        print(
            f"Difference vs zlib: "
            f"+{difference:,} bytes"
        )

        print()
        print(
            "No compression advantage."
        )

    return {
        "name": name,
        "original": original_size,
        "zlib": zlib_size,
        "neural": neural_total,
        "hybrid": hybrid_total
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LEARNED STRUCTURE STORAGE BENCHMARK")
    print("=" * 70)

    print()
    print(
        "Device:",
        DEVICE
    )

    print(
        "Dataset size:",
        format_bytes(
            TOTAL_SIZE
        )
    )

    print(
        "Training fraction:",
        TRAIN_FRACTION
    )

    print(
        "Neural samples:",
        NEURAL_SAMPLES
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Context size:",
        CONTEXT_SIZE
    )

    print(
        "Hidden size:",
        HIDDEN_SIZE
    )

    print()
    print(
        "Generating datasets..."
    )

    datasets = [

        (
            "STRUCTURED TEXT",
            generate_text(
                TOTAL_SIZE
            )
        ),

        (
            "SOURCE-CODE-LIKE",
            generate_code(
                TOTAL_SIZE
            )
        ),

        (
            "MATHEMATICAL",
            generate_math(
                TOTAL_SIZE
            )
        )
    ]

    results = []

    for name, data in datasets:

        result = run_dataset(
            name,
            data
        )

        results.append(
            result
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    print()
    print("#" * 70)
    print("GLOBAL SUMMARY")
    print("#" * 70)

    print()

    print(
        f"{'Dataset':<24}"
        f"{'Original':>14}"
        f"{'zlib':>14}"
        f"{'Neural':>14}"
        f"{'Hybrid':>14}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['name']:<24}"
            f"{result['original']:>14,}"
            f"{result['zlib']:>14,}"
            f"{result['neural']:>14,}"
            f"{result['hybrid']:>14,}"
        )

    print()
    print(
        "BENCHMARK COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()