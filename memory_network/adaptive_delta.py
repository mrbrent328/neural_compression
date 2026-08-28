#!/usr/bin/env python3
"""Adaptive selection of lossless structured delta representations.

Part A: an oracle tries every representation and chooses the smallest valid
compressed packet.
Part B: a fixed, transparent rule-based chooser selects a representation from
structural statistics before encoding. Rules are established from a separate
training population and then frozen for the held-out evaluation population.
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


def delta_statistics(indices, values):
    """Return only statistics available from the observed state change."""
    changed = len(indices)
    density = changed / STATE_SIZE
    if changed <= 1:
        gaps = np.array([], dtype=np.float64)
    else:
        gaps = np.diff(indices).astype(np.float64)
    starts = 0 if changed == 0 else 1 + int(np.sum(gaps > 1))
    return {
        "change_density": float(density),
        "run_count": int(starts),
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
    """Cheating reference: inspect every valid encoded size."""
    measurements = {
        name: encode_and_measure(name, indices, values)
        for name in ENCODERS
    }
    invalid = [name for name, result in measurements.items() if not result["exact"]]
    if invalid:
        raise RuntimeError(f"lossless representation failed: {invalid}")
    best = min(measurements, key=lambda name: measurements[name]["compressed_bytes"])
    return best, measurements


def choose_adaptive(stats):
    """Fixed structural rules. No encoded size is inspected here."""
    changed_density = stats["change_density"]
    runs = stats["run_count"]
    avg_gap = stats["average_index_gap"]
    value_std = stats["value_std"]

    # Exact contiguous changes are the strongest evidence for ranges.
    if runs == 1 and changed_density > 0:
        return "ranges"

    # Sparse changes with large, regular gaps favor index deltas.
    if changed_density <= 0.25 and avg_gap >= 2.0:
        return "index_delta"

    # Value similarity is a structural signal for factoring, but only after
    # ruling out an obvious range structure.
    if changed_density <= 0.25 and value_std <= 0.001:
        return "factored_values"

    # Otherwise use the representation with no structural assumptions.
    return "absolute"


def collect_training_data():
    """Generate training observations; oracle labels are not exposed to chooser."""
    rng = np.random.default_rng(SEED_TRAIN)
    rows = []
    for pattern in PATTERNS:
        for changed in CHANGED_VALUES:
            for _ in range(TRAIN_CASES_PER_CELL):
                indices, values = generate_change(rng, changed, pattern)
                stats = delta_statistics(indices, values)
                oracle, measurements = oracle_choice(indices, values)
                rows.append({
                    "pattern": pattern,
                    "changed_values": changed,
                    "stats": stats,
                    "oracle": oracle,
                    "oracle_bytes": measurements[oracle]["compressed_bytes"],
                })
    return rows


def evaluate_test_data():
    """Evaluate frozen rules on a separate seed."""
    rng = np.random.default_rng(SEED_TEST)
    rows = []
    for pattern in PATTERNS:
        for changed in CHANGED_VALUES:
            for case_index in range(TEST_CASES_PER_CELL):
                indices, values = generate_change(rng, changed, pattern)
                stats = delta_statistics(indices, values)
                adaptive = choose_adaptive(stats)
                adaptive_result = encode_and_measure(adaptive, indices, values)
                if not adaptive_result["exact"]:
                    raise RuntimeError(
                        f"adaptive representation was not lossless: {adaptive}"
                    )
                oracle, measurements = oracle_choice(indices, values)
                oracle_bytes = measurements[oracle]["compressed_bytes"]
                adaptive_bytes = adaptive_result["compressed_bytes"]
                rows.append({
                    "pattern": pattern,
                    "changed_values": changed,
                    "case_index": case_index,
                    "stats": stats,
                    "baseline_representation": "absolute",
                    "baseline_bytes": measurements["absolute"]["compressed_bytes"],
                    "adaptive_representation": adaptive,
                    "adaptive_bytes": adaptive_bytes,
                    "adaptive_exact": adaptive_result["exact"],
                    "oracle_representation": oracle,
                    "oracle_bytes": oracle_bytes,
                    "oracle_exact": measurements[oracle]["exact"],
                    "regret_bytes": adaptive_bytes - oracle_bytes,
                    "efficiency_ratio": adaptive_bytes / oracle_bytes,
                })
    return rows


def summarize(rows):
    ratios = np.asarray([row["efficiency_ratio"] for row in rows], dtype=np.float64)
    regrets = np.asarray([row["regret_bytes"] for row in rows], dtype=np.float64)
    adaptive_wins = sum(
        row["adaptive_bytes"] == row["oracle_bytes"] for row in rows
    )
    baseline_wins = sum(
        row["baseline_bytes"] == row["oracle_bytes"] for row in rows
    )
    return {
        "cases": len(rows),
        "adaptive_oracle_exact_match_rate": adaptive_wins / len(rows),
        "baseline_oracle_exact_match_rate": baseline_wins / len(rows),
        "mean_efficiency_ratio": float(np.mean(ratios)),
        "median_efficiency_ratio": float(np.median(ratios)),
        "worst_efficiency_ratio": float(np.max(ratios)),
        "mean_regret_bytes": float(np.mean(regrets)),
        "max_regret_bytes": int(np.max(regrets)),
        "zero_regret_cases": int(np.sum(regrets == 0)),
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
    # The training population is retained for auditability, but its oracle
    # labels are not consumed by choose_adaptive(). The rules above are frozen.
    rows = evaluate_test_data()
    summary = summarize(rows)
    output = {
        "experiment": "adaptive delta representation selection",
        "state_values": STATE_SIZE,
        "seeds": {"training": SEED_TRAIN, "test": SEED_TEST},
        "cases_per_cell": {
            "training": TRAIN_CASES_PER_CELL,
            "test": TEST_CASES_PER_CELL,
        },
        "candidate_representations": list(ENCODERS),
        "adaptive_rules": {
            "ranges": "run_count == 1",
            "index_delta": "density <= 0.25 and average_index_gap >= 2.0",
            "factored_values": "density <= 0.25 and value_std <= 0.001, after range/gap rules",
            "absolute": "fallback",
        },
        "training_cases": len(training),
        "summary": summary,
        "test_results": rows,
    }
    Path("adaptive_delta_result.json").write_text(json.dumps(output, indent=2))
    print_summary(rows, summary)


if __name__ == "__main__":
    main()
