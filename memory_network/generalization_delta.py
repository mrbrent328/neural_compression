#!/usr/bin/env python3
"""Test whether structural delta selection generalizes to unseen structures.

Training uses the same three pattern families as adaptive_delta.py. The
selector is then frozen and evaluated on both known-pattern controls and
novel/composed change structures. Test-time representation sizes are never
used to make the adaptive decision; the oracle is evaluated independently
afterward.
"""

import json
import zlib
from pathlib import Path

import numpy as np

from adaptive_delta import (
    CHANGED_VALUES,
    REPRESENTATIONS,
    SEED_TRAIN,
    STATE_SIZE,
    TRAIN_CASES_PER_CELL,
    delta_statistics,
    encode_and_measure,
    feature_vector,
    fit_nearest_centroid_selector,
    oracle_choice,
    choose_adaptive,
)

SEED_TEST = 2003
TEST_CASES_PER_CELL = 100
TRAIN_PATTERNS = ("random", "clustered", "similar")
KNOWN_TEST_PATTERNS = TRAIN_PATTERNS
NOVEL_TEST_PATTERNS = (
    "strided",
    "alternating",
    "multi_cluster",
    "edge_biased",
    "clustered_similar",
)


def generate_change(rng, changed, pattern):
    """Generate a delta. Novel patterns are intentionally absent from training."""
    if changed == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    if pattern == "random":
        indices = np.sort(rng.choice(STATE_SIZE, changed, replace=False)).astype(np.int32)
        values = rng.normal(0.0, 0.01, changed).astype(np.float32)
    elif pattern == "clustered":
        start = int(rng.integers(0, STATE_SIZE - changed + 1))
        indices = np.arange(start, start + changed, dtype=np.int32)
        values = rng.normal(0.01, 0.01, changed).astype(np.float32)
    elif pattern == "similar":
        indices = np.sort(rng.choice(STATE_SIZE, changed, replace=False)).astype(np.int32)
        values = rng.normal(0.01, 0.0001, changed).astype(np.float32)
    elif pattern == "strided":
        stride = max(2, min(16, STATE_SIZE // max(1, changed)))
        offsets = np.arange(changed, dtype=np.int64) * stride
        if offsets[-1] >= STATE_SIZE:
            indices = np.linspace(0, STATE_SIZE - 1, changed, dtype=np.int32)
        else:
            start = int(rng.integers(0, max(1, STATE_SIZE - int(offsets[-1]))))
            indices = (start + offsets).astype(np.int32)
        values = rng.normal(0.0, 0.01, changed).astype(np.float32)
    elif pattern == "alternating":
        parity = int(rng.integers(0, 2))
        candidates = np.arange(parity, STATE_SIZE, 2, dtype=np.int32)
        if changed <= len(candidates):
            indices = np.sort(rng.choice(candidates, changed, replace=False)).astype(np.int32)
        else:
            indices = np.sort(rng.choice(STATE_SIZE, changed, replace=False)).astype(np.int32)
        values = rng.normal(0.0, 0.01, changed).astype(np.float32)
    elif pattern == "multi_cluster":
        clusters = min(4, max(2, int(np.ceil(changed / 8))))
        points = []
        remaining = changed
        for cluster in range(clusters):
            count = remaining // (clusters - cluster)
            max_start = max(0, STATE_SIZE - max(1, count))
            start = int(rng.integers(0, max_start + 1))
            points.extend(range(start, min(STATE_SIZE, start + count)))
            remaining -= count
        indices = np.unique(np.asarray(points, dtype=np.int32))
        while len(indices) < changed:
            extra = int(rng.integers(0, STATE_SIZE))
            if extra not in indices:
                indices = np.sort(np.append(indices, extra).astype(np.int32))
        if len(indices) > changed:
            indices = indices[:changed]
        values = rng.normal(0.0, 0.01, changed).astype(np.float32)
    elif pattern == "edge_biased":
        left = min(changed // 2, changed)
        right = changed - left
        left_pool = np.arange(0, max(1, STATE_SIZE // 8), dtype=np.int32)
        right_pool = np.arange(max(0, STATE_SIZE - STATE_SIZE // 8), STATE_SIZE, dtype=np.int32)
        chosen = []
        if left:
            chosen.extend(rng.choice(left_pool, min(left, len(left_pool)), replace=False).tolist())
        if right:
            chosen.extend(rng.choice(right_pool, min(right, len(right_pool)), replace=False).tolist())
        indices = np.unique(np.asarray(chosen, dtype=np.int32))
        while len(indices) < changed:
            indices = np.unique(np.append(indices, int(rng.integers(0, STATE_SIZE))).astype(np.int32))
        if len(indices) > changed:
            indices = np.sort(indices[:changed])
        values = rng.normal(0.0, 0.01, changed).astype(np.float32)
    elif pattern == "clustered_similar":
        start = int(rng.integers(0, STATE_SIZE - changed + 1))
        indices = np.arange(start, start + changed, dtype=np.int32)
        values = rng.normal(0.01, 0.0001, changed).astype(np.float32)
    else:
        raise ValueError(pattern)

    return np.sort(indices), values


def collect_training_data():
    rng = np.random.default_rng(SEED_TRAIN)
    rows = []
    for pattern in TRAIN_PATTERNS:
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
    for pattern in KNOWN_TEST_PATTERNS + NOVEL_TEST_PATTERNS:
        for changed in CHANGED_VALUES:
            for case_index in range(TEST_CASES_PER_CELL):
                indices, values = generate_change(rng, changed, pattern)
                stats = delta_statistics(indices, values)

                # Adaptive choice happens before candidate encoding sizes are inspected.
                adaptive = choose_adaptive(stats, selector)
                adaptive_result = encode_and_measure(adaptive, indices, values)
                if not adaptive_result["exact"]:
                    raise RuntimeError(f"adaptive representation was not lossless: {adaptive}")

                oracle, measurements = oracle_choice(indices, values)
                adaptive_bytes = adaptive_result["compressed_bytes"]
                oracle_bytes = measurements[oracle]["compressed_bytes"]
                baseline_bytes = measurements["absolute"]["compressed_bytes"]

                rows.append({
                    "distribution": "known" if pattern in KNOWN_TEST_PATTERNS else "novel",
                    "pattern": pattern,
                    "changed_values": changed,
                    "case_index": case_index,
                    "stats": stats,
                    "adaptive_representation": adaptive,
                    "adaptive_bytes": adaptive_bytes,
                    "adaptive_exact": adaptive_result["exact"],
                    "oracle_representation": oracle,
                    "oracle_bytes": oracle_bytes,
                    "oracle_exact": measurements[oracle]["exact"],
                    "baseline_bytes": baseline_bytes,
                    "regret_bytes": adaptive_bytes - oracle_bytes,
                    "efficiency_ratio": adaptive_bytes / oracle_bytes,
                })
    return rows


def summarize(rows):
    ratios = np.asarray([r["efficiency_ratio"] for r in rows], dtype=np.float64)
    regrets = np.asarray([r["regret_bytes"] for r in rows], dtype=np.float64)
    adaptive = np.asarray([r["adaptive_bytes"] for r in rows], dtype=np.float64)
    baseline = np.asarray([r["baseline_bytes"] for r in rows], dtype=np.float64)
    return {
        "cases": len(rows),
        "adaptive_oracle_exact_match_rate": float(np.mean(regrets == 0)),
        "mean_efficiency_ratio": float(np.mean(ratios)),
        "median_efficiency_ratio": float(np.median(ratios)),
        "worst_efficiency_ratio": float(np.max(ratios)),
        "mean_regret_bytes": float(np.mean(regrets)),
        "max_regret_bytes": int(np.max(regrets)),
        "zero_regret_cases": int(np.sum(regrets == 0)),
        "adaptive_better_than_absolute_cases": int(np.sum(adaptive < baseline)),
        "adaptive_equal_absolute_cases": int(np.sum(adaptive == baseline)),
        "adaptive_worse_than_absolute_cases": int(np.sum(adaptive > baseline)),
        "mean_absolute_bytes": float(np.mean(baseline)),
        "mean_adaptive_bytes": float(np.mean(adaptive)),
    }


def summarize_group(rows):
    return {
        "cases": len(rows),
        "match_rate": float(np.mean([r["regret_bytes"] == 0 for r in rows])),
        "mean_ratio": float(np.mean([r["efficiency_ratio"] for r in rows])),
        "median_ratio": float(np.median([r["efficiency_ratio"] for r in rows])),
        "mean_regret_bytes": float(np.mean([r["regret_bytes"] for r in rows])),
        "max_regret_bytes": int(np.max([r["regret_bytes"] for r in rows])),
    }


def main():
    training = collect_training_data()
    selector = fit_nearest_centroid_selector(training)
    rows = evaluate_test_data(selector)

    summary = summarize(rows)
    by_distribution = {}
    for distribution in ("known", "novel"):
        group = [r for r in rows if r["distribution"] == distribution]
        by_distribution[distribution] = summarize_group(group)

    by_pattern = {}
    for pattern in KNOWN_TEST_PATTERNS + NOVEL_TEST_PATTERNS:
        group = [r for r in rows if r["pattern"] == pattern]
        by_pattern[pattern] = summarize_group(group)

    worst = sorted(rows, key=lambda r: r["regret_bytes"], reverse=True)[:20]
    output = {
        "experiment": "out-of-distribution structural delta selection generalization",
        "state_values": STATE_SIZE,
        "training_patterns": list(TRAIN_PATTERNS),
        "known_test_patterns": list(KNOWN_TEST_PATTERNS),
        "novel_test_patterns": list(NOVEL_TEST_PATTERNS),
        "seeds": {"training": SEED_TRAIN, "test": SEED_TEST},
        "cases_per_cell": {"training": TRAIN_CASES_PER_CELL, "test": TEST_CASES_PER_CELL},
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
        "by_distribution": by_distribution,
        "by_pattern": by_pattern,
        "worst_cases": worst,
        "test_results": rows,
    }
    Path("generalization_delta_result.json").write_text(json.dumps(output, indent=2))

    print("Out-of-Distribution Adaptive Delta Generalization")
    print(f"training seed: {SEED_TRAIN}")
    print(f"test seed:     {SEED_TEST}")
    print(f"training cases: {len(training)}")
    print(f"test cases:     {len(rows)}")
    print()
    print(json.dumps(by_distribution, indent=2))
    print()
    print(json.dumps(by_pattern, indent=2))
    print()
    print("Overall:")
    print(json.dumps(summary, indent=2))
    print()
    print("Worst cases:")
    for row in worst[:10]:
        print(
            f"{row['pattern']:<18} changed={row['changed_values']:>4} "
            f"adaptive={row['adaptive_representation']:<14} "
            f"oracle={row['oracle_representation']:<14} "
            f"regret={row['regret_bytes']:>4} ratio={row['efficiency_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
