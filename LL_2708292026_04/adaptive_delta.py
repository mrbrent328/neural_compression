#!/usr/bin/env python3
"""Learn a small structural selector for lossless delta representations.

The oracle may inspect all candidate encoded sizes while producing training
labels. The selector itself sees only structural statistics of the observed
delta. After fitting, the selector is frozen and evaluated on a new seed.
"""

import json
import zlib
from pathlib import Path

import numpy as np

from structured_delta import ENCODERS, STATE_SIZE, decode_and_reconstruct

SEED_TRAIN = 1001
SEED_TEST = 2002
TRAIN_CASES_PER_CELL = 100
TEST_CASES_PER_CELL = 100
CHANGED_VALUES = (1, 4, 16, 64, 256, 512, 1024)
PATTERNS = ("random", "clustered", "similar")
REPRESENTATIONS = tuple(ENCODERS)


def delta_statistics(indices, values):
    """Statistics observable from the current delta before encoding."""
    changed = len(indices)
    density = changed / STATE_SIZE
    if changed <= 1:
        gaps = np.array([], dtype=np.float64)
    else:
        gaps = np.diff(indices).astype(np.float64)
    run_count = 0 if changed == 0 else 1 + int(np.sum(gaps > 1))
    return {
        "change_density": float(density),
        "run_count": int(run_count),
        "average_index_gap": float(np.mean(gaps)) if len(gaps) else 0.0,
        "gap_variance": float(np.var(gaps)) if len(gaps) else 0.0,
        "value_mean": float(np.mean(values)) if changed else 0.0,
        "value_std": float(np.std(values)) if changed else 0.0,
    }


def generate_change(rng, changed, pattern):
    if pattern == "clustered":
        start = int(rng.integers(0, STATE_SIZE - changed + 1))
        indices = np.arange(start, start + changed, dtype=np.int32)
        values = rng.normal(0.01, 0.01, changed).astype(np.float32)
    elif pattern == "similar":
        indices = np.sort(rng.choice(STATE_SIZE, changed, replace=False)).astype(np.int32)
        values = rng.normal(0.01, 0.0001, changed).astype(np.float32)
    elif pattern == "random":
        indices = np.sort(rng.choice(STATE_SIZE, changed, replace=False)).astype(np.int32)
        values = rng.normal(0.0, 0.01, changed).astype(np.float32)
    else:
        raise ValueError(pattern)
    return indices, values


def encode_and_measure(name, indices, values):
    raw = ENCODERS[name](indices, values)
    compressed = zlib.compress(raw, level=9)
    decoded_indices, decoded_values = decode_and_reconstruct(zlib.decompress(compressed))
    exact = bool(
        np.array_equal(indices, decoded_indices)
        and np.array_equal(values, decoded_values)
    )
    return {
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "exact": exact,
    }


def oracle_choice(indices, values):
    """Reference only: inspect every candidate's actual encoded size."""
    measurements = {
        name: encode_and_measure(name, indices, values)
        for name in ENCODERS
    }
    invalid = [name for name, result in measurements.items() if not result["exact"]]
    if invalid:
        raise RuntimeError(f"lossless representation failed: {invalid}")
    best = min(measurements, key=lambda name: measurements[name]["compressed_bytes"])
    return best, measurements


def feature_vector(stats):
    """Numerical structural features exposed to the selector."""
    return np.asarray([
        stats["change_density"],
        stats["run_count"] / max(1, STATE_SIZE),
        stats["average_index_gap"] / max(1, STATE_SIZE),
        stats["gap_variance"] / max(1, STATE_SIZE * STATE_SIZE),
        stats["value_mean"],
        stats["value_std"],
    ], dtype=np.float64)


def fit_nearest_centroid_selector(training_rows):
    """Fit an intentionally simple multiclass structural classifier.

    The oracle labels are used only during fitting. The classifier stores one
    centroid per representation in standardized feature space and chooses the
    nearest centroid at test time. No test-case encoded sizes are consulted.
    """
    X = np.asarray([feature_vector(row["stats"]) for row in training_rows])
    labels = np.asarray([row["oracle"] for row in training_rows])

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale

    centroids = {}
    for name in REPRESENTATIONS:
        members = Z[labels == name]
        if len(members):
            centroids[name] = members.mean(axis=0)

    if not centroids:
        raise RuntimeError("training produced no oracle labels")

    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "centroids": {name: centroid.tolist() for name, centroid in centroids.items()},
    }


def choose_adaptive(stats, selector):
    X = feature_vector(stats)
    mean = np.asarray(selector["feature_mean"], dtype=np.float64)
    scale = np.asarray(selector["feature_scale"], dtype=np.float64)
    Z = (X - mean) / scale
    centroids = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in selector["centroids"].items()
    }
    return min(centroids, key=lambda name: float(np.linalg.norm(Z - centroids[name])))


def collect_training_data():
    rng = np.random.default_rng(SEED_TRAIN)
    rows = []
    for pattern in PATTERNS:
        for changed in CHANGED_VALUES:
            for case_index in range(TRAIN_CASES_PER_CELL):
                indices, values = generate_change(rng, changed, pattern)
                stats = delta_statistics(indices, values)
                oracle, measurements = oracle_choice(indices, values)
                rows.append({
                    "pattern": pattern,
                    "changed_values": changed,
                    "case_index": case_index,
                    "stats": stats,
                    "oracle": oracle,
                    "oracle_bytes": measurements[oracle]["compressed_bytes"],
                })
    return rows


def evaluate_test_data(selector):
    rng = np.random.default_rng(SEED_TEST)
    rows = []
    for pattern in PATTERNS:
        for changed in CHANGED_VALUES:
            for case_index in range(TEST_CASES_PER_CELL):
                indices, values = generate_change(rng, changed, pattern)
                stats = delta_statistics(indices, values)

                # This decision occurs before any candidate is encoded.
                adaptive = choose_adaptive(stats, selector)
                adaptive_result = encode_and_measure(adaptive, indices, values)
                if not adaptive_result["exact"]:
                    raise RuntimeError(f"adaptive representation was not lossless: {adaptive}")

                # Oracle is evaluated independently after the adaptive decision.
                oracle, measurements = oracle_choice(indices, values)
                adaptive_bytes = adaptive_result["compressed_bytes"]
                oracle_bytes = measurements[oracle]["compressed_bytes"]
                baseline_bytes = measurements["absolute"]["compressed_bytes"]

                rows.append({
                    "pattern": pattern,
                    "changed_values": changed,
                    "case_index": case_index,
                    "stats": stats,
                    "baseline_representation": "absolute",
                    "baseline_bytes": baseline_bytes,
                    "adaptive_representation": adaptive,
                    "adaptive_bytes": adaptive_bytes,
                    "adaptive_exact": adaptive_result["exact"],
                    "oracle_representation": oracle,
                    "oracle_bytes": oracle_bytes,
                    "oracle_exact": measurements[oracle]["exact"],
                    "regret_bytes": adaptive_bytes - oracle_bytes,
                    "efficiency_ratio": adaptive_bytes / oracle_bytes,
                    "adaptive_vs_baseline_bytes": adaptive_bytes - baseline_bytes,
                })
    return rows


def summarize(rows):
    ratios = np.asarray([row["efficiency_ratio"] for row in rows], dtype=np.float64)
    regrets = np.asarray([row["regret_bytes"] for row in rows], dtype=np.float64)
    baseline = np.asarray([row["baseline_bytes"] for row in rows], dtype=np.float64)
    adaptive = np.asarray([row["adaptive_bytes"] for row in rows], dtype=np.float64)
    return {
        "cases": len(rows),
        "adaptive_oracle_exact_match_rate": float(np.mean(regrets == 0)),
        "baseline_oracle_exact_match_rate": float(
            np.mean(np.asarray([row["baseline_bytes"] == row["oracle_bytes"] for row in rows]))
        ),
        "mean_efficiency_ratio": float(np.mean(ratios)),
        "median_efficiency_ratio": float(np.median(ratios)),
        "worst_efficiency_ratio": float(np.max(ratios)),
        "mean_regret_bytes": float(np.mean(regrets)),
        "max_regret_bytes": int(np.max(regrets)),
        "zero_regret_cases": int(np.sum(regrets == 0)),
        "adaptive_better_than_baseline_cases": int(np.sum(adaptive < baseline)),
        "adaptive_equal_baseline_cases": int(np.sum(adaptive == baseline)),
        "adaptive_worse_than_baseline_cases": int(np.sum(adaptive > baseline)),
        "mean_baseline_bytes": float(np.mean(baseline)),
        "mean_adaptive_bytes": float(np.mean(adaptive)),
    }


def print_summary(rows, summary):
    print("Adaptive Delta")
    print(f"training seed: {SEED_TRAIN}")
    print(f"test seed:     {SEED_TEST}")
    print(f"state values:  {STATE_SIZE}")
    print()
    print("pattern       changed   oracle         oracle B   adaptive       adaptive B   ratio    regret")
    print("-" * 94)
    for row in rows:
        if row["case_index"] != 0:
            continue
        print(
            f"{row['pattern']:<13} {row['changed_values']:>7}   "
            f"{row['oracle_representation']:<13} {row['oracle_bytes']:>8}   "
            f"{row['adaptive_representation']:<13} {row['adaptive_bytes']:>10}   "
            f"{row['efficiency_ratio']:>6.3f}   {row['regret_bytes']:>6}"
        )
    print()
    print(json.dumps(summary, indent=2))


def main():
    training = collect_training_data()
    selector = fit_nearest_centroid_selector(training)
    test_rows = evaluate_test_data(selector)
    summary = summarize(test_rows)

    output = {
        "experiment": "adaptive delta representation selection",
        "state_values": STATE_SIZE,
        "seeds": {"training": SEED_TRAIN, "test": SEED_TEST},
        "cases_per_cell": {
            "training": TRAIN_CASES_PER_CELL,
            "test": TEST_CASES_PER_CELL,
        },
        "candidate_representations": list(REPRESENTATIONS),
        "selector": {
            "type": "nearest_centroid",
            "features": [
                "change_density",
                "run_count",
                "average_index_gap",
                "gap_variance",
                "value_mean",
                "value_std",
            ],
            **selector,
        },
        "training_cases": len(training),
        "summary": summary,
        "test_results": test_rows,
    }
    Path("adaptive_delta_result.json").write_text(json.dumps(output, indent=2))
    print_summary(test_rows, summary)


if __name__ == "__main__":
    main()
