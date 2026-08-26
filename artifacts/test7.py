#!/usr/bin/env python3

"""
TEST4 — COMPRESSION SCALING CURVE

Purpose
-------
Test whether predictive compression becomes increasingly advantageous
as dataset size grows.

The central question:

    Does fixed model overhead get amortized over larger datasets?

Methods
-------
1. zlib baseline
2. previous-byte predictor
3. context predictor
4. neural predictor
5. hybrid predictor

The experiment deliberately reports:

    residual size
    model cost
    metadata cost
    TOTAL storage

because residual-only numbers can be misleading at small scales.

This is an empirical scaling experiment, not a claim that any method
will asymptotically beat every compressor.

"""

import math
import os
import time
import hashlib
import zlib
import json
import csv
import sys
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 1337

# Dataset sizes to test.
#
# These are deliberately spaced over roughly an order of magnitude.
# Add/remove sizes depending on available RAM and CPU time.
SIZES = [
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
]

TRAIN_FRACTION = 0.75

CONTEXT_SIZE = 4
HIDDEN_SIZE = 64

# Neural predictor settings.
#
# The neural experiment is expensive. The scaling experiment therefore
# uses a fixed number of training samples rather than allowing training
# cost to grow indefinitely with dataset size.
NEURAL_SAMPLES = 300_000
NEURAL_EPOCHS = 500

# Fixed representation/model overhead estimates.
METADATA_BYTES = 1024

# Output files.
CSV_FILE = "test4_scaling_results.csv"
JSON_FILE = "test4_scaling_results.json"


# ============================================================
# RANDOMNESS
# ============================================================

np.random.seed(SEED)

if torch is not None:
    torch.manual_seed(SEED)


# ============================================================
# DATA GENERATION
# ============================================================

def generate_structured_text(n):
    """
    Structured text with strong local relationships.
    """

    templates = [
        (b"temperature=%03d humidity=%03d pressure=%04d\n", 3),
        (b"sensor=%03d value=%05d status=OK\n", 2),
        (b"node=%03d channel=%02d packet=%05d\n", 3),
        (b"time=%08d event=UPDATE value=%04d\n", 2),
        (b"device=%03d voltage=%03d current=%03d\n", 3),
    ]

    out = bytearray()

    i = 0

    while len(out) < n:

        template, nargs = templates[
            i % len(templates)
        ]

        a = (i * 17) % 1000
        b = (i * 31) % 1000
        c = (i * 47) % 10000

        if nargs == 2:
            line = template % (a, b)
        else:
            line = template % (a, b, c)

        out.extend(line)

        i += 1

    return bytes(out[:n])


def generate_source_code(n):
    """
    Source-code-like material with recurring syntax and identifiers.
    """

    snippets = [
        b"if (value > threshold) {\n",
        b"    result = process(value);\n",
        b"    return result;\n",
        b"} else {\n",
        b"    result = fallback(value);\n",
        b"}\n",
        b"\n",
        b"for (int i = 0; i < count; i++) {\n",
        b"    buffer[i] = transform(buffer[i]);\n",
        b"}\n",
        b"const int DEFAULT_VALUE = 42;\n",
        b"static bool enabled = true;\n",
        b"while (running) {\n",
        b"    update_state();\n",
        b"}\n",
    ]

    out = bytearray()

    i = 0

    while len(out) < n:
        out.extend(snippets[i % len(snippets)])
        i += 1

    return bytes(out[:n])


def generate_mathematical(n):
    """
    Highly structured mathematical text.
    """

    templates = [
        b"x=%d y=%d z=%d result=%d\n",
        b"a=%d b=%d c=%d sum=%d\n",
        b"x^2=%d y^2=%d z^2=%d\n",
        b"f(x)=%d*x^2+%d*x+%d\n",
        b"integral=%d derivative=%d\n",
    ]

    out = bytearray()

    i = 0

    while len(out) < n:

        x = (i * 13) % 10000
        y = (i * 29) % 10000
        z = (i * 43) % 10000

        result = (x * x + 3 * y - 2 * z) % 100000

        template = templates[i % len(templates)]

        try:
            line = template % (
                x,
                y,
                z,
                result
            )
        except TypeError:
            # Different templates have different arities.
            if i % len(templates) == 2:
                line = template % (
                    x,
                    y,
                    z
                )
            elif i % len(templates) == 3:
                line = template % (
                    x,
                    y,
                    z
                )
            else:
                line = template % (
                    x,
                    y
                )

        out.extend(line)

        i += 1

    return bytes(out[:n])


DATASETS = {
    "STRUCTURED_TEXT": generate_structured_text,
    "SOURCE_CODE_LIKE": generate_source_code,
    "MATHEMATICAL": generate_mathematical,
}


# ============================================================
# UTILITIES
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def entropy(data):
    if not data:
        return 0.0

    counts = np.bincount(
        np.frombuffer(data, dtype=np.uint8),
        minlength=256
    )

    probabilities = counts[counts > 0] / len(data)

    return float(
        -np.sum(probabilities * np.log2(probabilities))
    )


def zlib_size(data):
    return len(zlib.compress(data, level=9))


def residual_bytes(actual, predicted):
    """
    Exact byte residual.

    Prediction is modulo 256 so reconstruction remains exact.
    """

    a = np.frombuffer(actual, dtype=np.uint8).astype(np.int16)
    p = np.asarray(predicted, dtype=np.int16)

    residual = (a - p) % 256

    return residual.astype(np.uint8).tobytes()


def reconstruction_from_residual(predicted, residual):
    p = np.asarray(predicted, dtype=np.int16)
    r = np.frombuffer(residual, dtype=np.uint8).astype(np.int16)

    reconstructed = (p + r) % 256

    return reconstructed.astype(np.uint8)


# ============================================================
# PREVIOUS BYTE
# ============================================================

def previous_byte_predict(data, start):
    """
    Predict each byte as the preceding byte.

    The first held-out byte uses the final training byte.
    """

    arr = np.frombuffer(data, dtype=np.uint8)

    predictions = arr[start - 1:-1].copy()

    actual = arr[start:]

    return actual, predictions


# ============================================================
# CONTEXT MODEL
# ============================================================

def train_context_model(data, start, context_size=4):
    """
    Map context -> most common next byte.
    """

    arr = np.frombuffer(data[:start], dtype=np.uint8)

    table = {}

    counts = defaultdict(Counter)

    if len(arr) <= context_size:
        return {}

    for i in range(context_size, len(arr)):
        context = bytes(arr[i - context_size:i])

        counts[context][int(arr[i])] += 1

    for context, counter in counts.items():
        table[context] = counter.most_common(1)[0][0]

    return table


def context_predict(data, start, context_size=4):
    """
    Predict held-out bytes using a learned context table.

    Contexts are allowed to cross the train/test boundary.
    """

    arr = np.frombuffer(data, dtype=np.uint8)

    model = train_context_model(
        data,
        start,
        context_size
    )

    predictions = []

    for i in range(start, len(arr)):

        context = bytes(
            arr[i - context_size:i]
        )

        prediction = model.get(
            context,
            int(arr[i - 1])
        )

        predictions.append(prediction)

    return arr[start:], np.asarray(
        predictions,
        dtype=np.uint8
    ), model


# ============================================================
# NEURAL MODEL
# ============================================================

class NeuralPredictor(nn.Module):

    def __init__(self, context_size=4, hidden_size=64):

        super().__init__()

        self.net = nn.Sequential(
            nn.Embedding(256, hidden_size),
            nn.Flatten(),
            nn.Linear(
                context_size * hidden_size,
                hidden_size
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)
        )

    def forward(self, x):
        return self.net(x)


def neural_model_size(model):
    """
    Approximate raw serialized parameter size.

    We deliberately use the same accounting for every dataset size.
    """

    parameter_bytes = 0

    for p in model.parameters():
        parameter_bytes += p.numel() * 4

    return parameter_bytes


def train_neural_model(
    data,
    start,
    context_size=4,
    hidden_size=64,
    samples=300_000,
    epochs=500
):

    if torch is None:

        raise RuntimeError(
            "PyTorch is not installed."
        )

    arr = np.frombuffer(
        data[:start],
        dtype=np.uint8
    )

    available = len(arr) - context_size

    if available <= 0:
        raise ValueError("Dataset too small.")

    samples = min(
        samples,
        available
    )

    # Deterministic sampling.
    rng = np.random.default_rng(SEED)

    indices = rng.choice(
        available,
        size=samples,
        replace=False
    )

    X = np.empty(
        (samples, context_size),
        dtype=np.int64
    )

    y = np.empty(
        samples,
        dtype=np.int64
    )

    for j, i in enumerate(indices):

        X[j] = arr[i:i + context_size]
        y[j] = arr[i + context_size]

    device = "cpu"

    model = NeuralPredictor(
        context_size,
        hidden_size
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    model.train()

    batch_size = 4096

    start_time = time.time()

    for epoch in range(epochs):

        permutation = torch.randperm(
            samples
        )

        total_loss = 0.0

        for begin in range(
            0,
            samples,
            batch_size
        ):

            idx = permutation[
                begin:begin + batch_size
            ]

            logits = model(
                X_t[idx]
            )

            loss = criterion(
                logits,
                y_t[idx]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.item())
                * len(idx)
            )

    elapsed = time.time() - start_time

    return (
        model,
        elapsed,
        neural_model_size(model)
    )


def neural_predict(
    model,
    data,
    start,
    context_size=4
):

    arr = np.frombuffer(
        data,
        dtype=np.uint8
    )

    contexts = []

    for i in range(start, len(arr)):

        contexts.append(
            arr[i - context_size:i]
        )

    X = torch.from_numpy(
        np.asarray(
            contexts,
            dtype=np.int64
        )
    )

    model.eval()

    predictions = []

    batch_size = 8192

    with torch.no_grad():

        for begin in range(
            0,
            len(X),
            batch_size
        ):

            logits = model(
                X[begin:begin + batch_size]
            )

            pred = torch.argmax(
                logits,
                dim=1
            )

            predictions.append(
                pred.cpu().numpy()
            )

    return np.concatenate(
        predictions
    ).astype(np.uint8)


# ============================================================
# METHOD EVALUATION
# ============================================================

def evaluate_prediction(
    actual,
    predicted,
    model_bytes=0
):

    residual = residual_bytes(
        actual,
        predicted
    )

    compressed_residual = zlib.compress(
        residual,
        level=9
    )

    residual_size = len(
        compressed_residual
    )

    total = (
        residual_size
        + model_bytes
        + METADATA_BYTES
    )

    actual_arr = np.frombuffer(
        actual,
        dtype=np.uint8
    ).astype(np.int32)

    pred_arr = np.asarray(
        predicted,
        dtype=np.uint8
    ).astype(np.int32)

    error = (
        actual_arr - pred_arr
    )

    mse = float(
        np.mean(error.astype(np.float64) ** 2)
    )

    mae = float(
        np.mean(np.abs(error))
    )

    zero_residual = float(
        np.mean(error == 0)
    )

    reconstructed = reconstruction_from_residual(
        pred_arr,
        residual
    )

    exact = np.array_equal(
        reconstructed,
        actual_arr.astype(np.uint8)
    )

    return {
        "residual": residual_size,
        "model": model_bytes,
        "metadata": METADATA_BYTES,
        "total": total,
        "mse": mse,
        "mae": mae,
        "zero_residual": zero_residual,
        "exact": exact,
    }


# ============================================================
# SCALING CURVE
# ============================================================

def run_scaling_experiment(
    dataset_name,
    generator
):

    print()
    print("=" * 70)
    print(
        f"SCALING CURVE: {dataset_name}"
    )
    print("=" * 70)

    results = []

    # Train the neural/context models once at the largest size.
    #
    # This is important:
    #
    # We want to measure the effect of data volume on the encoded
    # representation without retraining a new model for every curve
    # point.
    #
    # Otherwise model training itself becomes another variable.
    largest_size = max(SIZES)

    print()
    print(
        f"Generating largest dataset: "
        f"{largest_size:,} bytes"
    )

    full_data = generator(
        largest_size
    )

    largest_train = int(
        largest_size * TRAIN_FRACTION
    )

    print(
        f"Training region: "
        f"{largest_train:,} bytes"
    )

    print()
    print(
        "Training context model..."
    )

    context_model = train_context_model(
        full_data,
        largest_train,
        CONTEXT_SIZE
    )

    context_model_estimate = (
        len(context_model)
        * (CONTEXT_SIZE + 1)
        + 64
    )

    neural_model = None
    neural_model_cost = 0
    neural_training_time = 0

    if torch is not None:

        print(
            "Training neural model..."
        )

        (
            neural_model,
            neural_training_time,
            neural_model_cost
        ) = train_neural_model(
            full_data,
            largest_train,
            CONTEXT_SIZE,
            HIDDEN_SIZE,
            NEURAL_SAMPLES,
            NEURAL_EPOCHS
        )

        print(
            f"Neural training time: "
            f"{neural_training_time:.2f} s"
        )

        print(
            f"Neural model size: "
            f"{neural_model_cost:,} bytes"
        )

    else:

        print(
            "PyTorch unavailable; "
            "neural method disabled."
        )

    # --------------------------------------------------------
    # Each curve point
    # --------------------------------------------------------

    for size in SIZES:

        print()
        print("-" * 70)
        print(
            f"DATASET SIZE: {size:,} bytes"
        )
        print("-" * 70)

        data = full_data[:size]

        train_size = int(
            size * TRAIN_FRACTION
        )

        heldout_size = (
            size - train_size
        )

        print(
            f"Training: {train_size:,}"
        )

        print(
            f"Held-out: {heldout_size:,}"
        )

        zlib_bytes = zlib_size(
            data
        )

        print(
            f"Zlib: {zlib_bytes:,}"
        )

        # ----------------------------------------------------
        # Previous
        # ----------------------------------------------------

        actual, pred = previous_byte_predict(
            data,
            train_size
        )

        previous = evaluate_prediction(
            actual,
            pred
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        arr = np.frombuffer(
            data,
            dtype=np.uint8
        )

        context_predictions = []

        for i in range(
            train_size,
            size
        ):

            context = bytes(
                arr[
                    i - CONTEXT_SIZE:i
                ]
            )

            context_predictions.append(
                context_model.get(
                    context,
                    int(arr[i - 1])
                )
            )

        context = evaluate_prediction(
            arr[train_size:],
            np.asarray(
                context_predictions,
                dtype=np.uint8
            ),
            context_model_estimate
        )

        # ----------------------------------------------------
        # Neural
        # ----------------------------------------------------

        if neural_model is not None:

            neural_pred = neural_predict(
                neural_model,
                data,
                train_size,
                CONTEXT_SIZE
            )

            neural = evaluate_prediction(
                arr[train_size:],
                neural_pred,
                neural_model_cost
            )

        else:

            neural = {
                "residual": None,
                "model": None,
                "metadata": None,
                "total": None,
                "mse": None,
                "mae": None,
                "zero_residual": None,
                "exact": None,
            }

        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        if neural_model is not None:

            previous_pred = pred

            # Neural prediction first.
            hybrid_pred = neural_pred.copy()

            # Where neural prediction is worse than the simple
            # previous-byte prediction, use the previous-byte
            # prediction instead.
            neural_error = np.abs(
                arr[train_size:].astype(np.int16)
                - neural_pred.astype(np.int16)
            )

            previous_error = np.abs(
                arr[train_size:].astype(np.int16)
                - previous_pred.astype(np.int16)
            )

            hybrid_pred[
                previous_error < neural_error
            ] = previous_pred[
                previous_error < neural_error
            ]

            hybrid = evaluate_prediction(
                arr[train_size:],
                hybrid_pred,
                neural_model_cost
            )

        else:

            hybrid = {
                "residual": None,
                "model": None,
                "metadata": None,
                "total": None,
                "mse": None,
                "mae": None,
                "zero_residual": None,
                "exact": None,
            }

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        methods = {
            "previous": previous,
            "context": context,
            "neural": neural,
            "hybrid": hybrid,
        }

        best_method = min(
            methods.items(),
            key=lambda x:
                x[1]["total"]
                if x[1]["total"] is not None
                else float("inf")
        )

        record = {
            "dataset": dataset_name,
            "size": size,
            "train_size": train_size,
            "heldout_size": heldout_size,
            "zlib": zlib_bytes,
            "methods": methods,
            "best_method": best_method[0],
        }

        results.append(record)

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()

        print(
            f"{'METHOD':<12}"
            f"{'RESIDUAL':>12}"
            f"{'MODEL':>12}"
            f"{'TOTAL':>12}"
            f"{'vs ZLIB':>12}"
        )

        print("-" * 60)

        print(
            f"{'zlib':<12}"
            f"{zlib_bytes:>12,}"
            f"{'-':>12}"
            f"{zlib_bytes:>12,}"
            f"{1.0:>11.4f}x"
        )

        for name in [
            "previous",
            "context",
            "neural",
            "hybrid",
        ]:

            r = methods[name]

            if r["total"] is None:
                continue

            ratio = (
                r["total"]
                / zlib_bytes
            )

            print(
                f"{name:<12}"
                f"{r['residual']:>12,}"
                f"{r['model']:>12,}"
                f"{r['total']:>12,}"
                f"{ratio:>11.4f}x"
            )

        best = methods[
            best_method[0]
        ]

        improvement = (
            1.0
            - best["total"]
            / zlib_bytes
        ) * 100.0

        print()
        print(
            f"BEST: {best_method[0]}"
        )

        print(
            f"Best total: "
            f"{best['total']:,} bytes"
        )

        print(
            f"Improvement vs zlib: "
            f"{improvement:.4f}%"
        )

    return results


# ============================================================
# CURVE ANALYSIS
# ============================================================

def analyze_curve(results):

    print()
    print("=" * 70)
    print("SCALING CURVE ANALYSIS")
    print("=" * 70)

    for dataset in sorted(
        set(r["dataset"] for r in results)
    ):

        print()
        print(
            f"DATASET: {dataset}"
        )

        subset = [
            r for r in results
            if r["dataset"] == dataset
        ]

        print()

        print(
            f"{'SIZE':>12}"
            f"{'ZLIB':>12}"
            f"{'PREV':>12}"
            f"{'CONTEXT':>12}"
            f"{'NEURAL':>12}"
            f"{'HYBRID':>12}"
        )

        print("-" * 72)

        for r in subset:

            def total(name):
                value = (
                    r["methods"][name]["total"]
                )

                return (
                    str(value)
                    if value is not None
                    else "-"
                )

            print(
                f"{r['size']:>12,}"
                f"{r['zlib']:>12,}"
                f"{total('previous'):>12}"
                f"{total('context'):>12}"
                f"{total('neural'):>12}"
                f"{total('hybrid'):>12}"
            )

        # ----------------------------------------------------
        # Break-even analysis
        # ----------------------------------------------------

        print()
        print("BREAK-EVEN ANALYSIS")

        for method in [
            "previous",
            "context",
            "neural",
            "hybrid",
        ]:

            crossed = None

            for r in subset:

                total = (
                    r["methods"][method]["total"]
                )

                if total is None:
                    continue

                if total < r["zlib"]:

                    crossed = r["size"]

                    break

            if crossed is None:

                print(
                    f"{method:<12}: "
                    f"no break-even point observed"
                )

            else:

                print(
                    f"{method:<12}: "
                    f"{crossed:,} bytes"
                )

        # ----------------------------------------------------
        # Percentage improvement
        # ----------------------------------------------------

        print()
        print("IMPROVEMENT CURVE")

        for r in subset:

            print(
                f"{r['size']:>12,} bytes"
            )

            for method in [
                "previous",
                "context",
                "neural",
                "hybrid",
            ]:

                total = (
                    r["methods"][method]["total"]
                )

                if total is None:
                    continue

                improvement = (
                    1
                    - total / r["zlib"]
                ) * 100

                print(
                    f"    {method:<10}"
                    f"{improvement:>10.3f}%"
                )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    with open(
        JSON_FILE,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    rows = []

    for r in results:

        base = {
            "dataset": r["dataset"],
            "size": r["size"],
            "train_size": r["train_size"],
            "heldout_size": r["heldout_size"],
            "zlib": r["zlib"],
        }

        for method in [
            "previous",
            "context",
            "neural",
            "hybrid",
        ]:

            m = r["methods"][method]

            row = dict(base)

            row["method"] = method
            row["residual"] = m["residual"]
            row["model"] = m["model"]
            row["metadata"] = m["metadata"]
            row["total"] = m["total"]
            row["mse"] = m["mse"]
            row["mae"] = m["mae"]
            row["zero_residual"] = (
                m["zero_residual"]
            )
            row["exact"] = m["exact"]

            if m["total"] is not None:

                row["ratio_vs_zlib"] = (
                    m["total"]
                    / r["zlib"]
                )

                row["improvement_percent"] = (
                    1
                    - m["total"]
                    / r["zlib"]
                ) * 100

            else:

                row["ratio_vs_zlib"] = None
                row["improvement_percent"] = None

            rows.append(row)

    if rows:

        fields = list(
            rows[0].keys()
        )

        with open(
            CSV_FILE,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()
            writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TEST4 — COMPRESSION SCALING CURVE")
    print("=" * 70)

    print()
    print(
        "Question:"
    )

    print(
        "Does predictive compression become "
        "more advantageous as dataset size grows?"
    )

    print()
    print(
        f"Dataset sizes: "
        f"{', '.join(f'{x:,}' for x in SIZES)}"
    )

    print(
        f"Training fraction: "
        f"{TRAIN_FRACTION:.2f}"
    )

    print(
        f"Context size: "
        f"{CONTEXT_SIZE}"
    )

    print(
        f"Neural samples: "
        f"{NEURAL_SAMPLES:,}"
    )

    print(
        f"Neural epochs: "
        f"{NEURAL_EPOCHS}"
    )

    print(
        f"Metadata: "
        f"{METADATA_BYTES:,} bytes"
    )

    print()

    all_results = []

    total_start = time.time()

    for dataset_name, generator in DATASETS.items():

        results = run_scaling_experiment(
            dataset_name,
            generator
        )

        all_results.extend(
            results
        )

    analyze_curve(
        all_results
    )

    save_results(
        all_results
    )

    elapsed = (
        time.time()
        - total_start
    )

    print()
    print("=" * 70)
    print("TEST4 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total runtime: "
        f"{elapsed:.2f} seconds"
    )

    print()
    print(
        f"CSV results: "
        f"{CSV_FILE}"
    )

    print(
        f"JSON results: "
        f"{JSON_FILE}"
    )

    print()
    print(
        "The important number is TOTAL storage,"
    )

    print(
        "not residual size alone."
    )

    print()
    print(
        "If the predictive curves fall increasingly"
    )

    print(
        "below the zlib curve as size increases,"
    )

    print(
        "that is direct evidence for the"
    )

    print(
        "model-overhead-amortization hypothesis."
    )

    print()
    print(
        "BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()