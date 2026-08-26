#!/usr/bin/env python3
"""
V3: Local Structure -> Strategy Selection

V2 used only three coarse chunk statistics:
    entropy
    repetition
    unique_fraction

V3 keeps those as a control and adds cheap fine-grained structural
features describing how bytes are arranged inside the chunk.

The candidate compressors are unchanged. Only the selector's representation
changes.

Usage:
    python3 neural_local_strategy_experiment_v3_structural.py FILE --chunk 32000

Example:
    python3 neural_local_strategy_experiment_v3_structural.py \
        /home/xetal/Desktop/largecalgarycorpus/news \
        --chunk 32000 \
        --results local_strategy_results_v3.json
"""

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from neural_compress_experiment_full import compress_file
except ImportError:
    from neural_compress_opt_batch import compress_file


CANDIDATES = [
    {
        "name": "small",
        "context": 8,
        "embed": 32,
        "hidden": 64,
        "states": 1,
    },
    {
        "name": "baseline",
        "context": 16,
        "embed": 32,
        "hidden": 64,
        "states": 1,
    },
    {
        "name": "large",
        "context": 10,
        "embed": 48,
        "hidden": 300,
        "states": 3,
    },
]


# ---------------------------------------------------------------------------
# Original V2 features
# ---------------------------------------------------------------------------

def entropy(data):
    if not data:
        return 0.0

    counts = Counter(data)
    n = len(data)

    return -sum(
        (count / n) * math.log2(count / n)
        for count in counts.values()
    )


def repetition_score(data):
    if len(data) < 2:
        return 0.0

    return sum(
        1 for a, b in zip(data, data[1:]) if a == b
    ) / (len(data) - 1)


def unique_byte_fraction(data):
    if not data:
        return 0.0

    return len(set(data)) / 256.0


def signature(data):
    return (
        round(entropy(data), 1),
        round(repetition_score(data), 2),
        round(unique_byte_fraction(data), 2),
    )


# ---------------------------------------------------------------------------
# V3 fine-grained structural features
# ---------------------------------------------------------------------------

def mean_std(values):
    if not values:
        return 0.0, 0.0

    mean = sum(values) / len(values)
    std = (
        sum((x - mean) ** 2 for x in values) / len(values)
    ) ** 0.5

    return mean, std


def window_statistics(data, window_size=256):
    """
    Look at how structure changes INSIDE the chunk.

    This is deliberately cheap. We do not run a neural model here.
    """

    local_entropies = []
    local_change_rates = []

    for start in range(0, len(data), window_size):
        window = data[start:start + window_size]

        if not window:
            continue

        counts = Counter(window)
        n = len(window)

        e = -sum(
            (count / n) * math.log2(count / n)
            for count in counts.values()
        )
        local_entropies.append(e)

        if len(window) > 1:
            changes = sum(
                a != b
                for a, b in zip(window, window[1:])
            )
            local_change_rates.append(
                changes / (len(window) - 1)
            )

    entropy_mean, entropy_std = mean_std(local_entropies)
    change_mean, change_std = mean_std(local_change_rates)

    return {
        "window_entropy_mean": entropy_mean,
        "window_entropy_std": entropy_std,
        "change_rate": change_mean,
        "change_rate_std": change_std,
    }


def run_statistics(data):
    """
    Describe contiguous runs of identical bytes.

    The original repetition score only counts adjacent equality.
    Run statistics tell us how that repetition is arranged.
    """

    if not data:
        return {
            "run_mean": 0.0,
            "run_std": 0.0,
            "long_run_fraction": 0.0,
        }

    runs = []
    current = 1

    for a, b in zip(data, data[1:]):
        if a == b:
            current += 1
        else:
            runs.append(current)
            current = 1

    runs.append(current)

    run_mean, run_std = mean_std(runs)

    long_run_bytes = sum(
        length for length in runs if length >= 4
    )

    return {
        "run_mean": run_mean,
        "run_std": run_std,
        "long_run_fraction": long_run_bytes / len(data),
    }


def distribution_features(data):
    """
    Byte-distribution concentration.

    Two chunks can have the same entropy while distributing their mass
    differently. These features expose that shape.
    """

    if not data:
        return {
            "top_byte_fraction": 0.0,
            "top4_byte_fraction": 0.0,
            "top16_byte_fraction": 0.0,
            "histogram_std": 0.0,
        }

    counts = Counter(data)
    n = len(data)

    frequencies = sorted(
        (count / n for count in counts.values()),
        reverse=True,
    )

    top1 = frequencies[0]
    top4 = sum(frequencies[:4])
    top16 = sum(frequencies[:16])

    # Include zero-frequency byte values in histogram shape.
    all_freqs = [
        counts.get(i, 0) / n
        for i in range(256)
    ]
    histogram_mean = 1.0 / 256.0
    histogram_std = (
        sum((x - histogram_mean) ** 2 for x in all_freqs) / 256
    ) ** 0.5

    return {
        "top_byte_fraction": top1,
        "top4_byte_fraction": top4,
        "top16_byte_fraction": top16,
        "histogram_std": histogram_std,
    }


def pair_structure(data):
    """
    Adjacent-pair diversity.

    This adds ordering information that a plain byte histogram cannot see.
    """

    if len(data) < 2:
        return {
            "pair_unique_fraction": 0.0,
            "pair_repeat_rate": 0.0,
        }

    pairs = list(zip(data, data[1:]))
    unique_pairs = len(set(pairs))
    total = len(pairs)

    return {
        "pair_unique_fraction": unique_pairs / total,
        "pair_repeat_rate": 1.0 - (unique_pairs / total),
    }


def analyze_chunk(data):
    """
    Build the selector representation.

    IMPORTANT:
    The original three V2 features remain unchanged.
    """

    info = {
        "size": len(data),
        "entropy": entropy(data),
        "repetition": repetition_score(data),
        "unique_fraction": unique_byte_fraction(data),
        "signature": signature(data),
    }

    info.update(window_statistics(data))
    info.update(run_statistics(data))
    info.update(distribution_features(data))
    info.update(pair_structure(data))

    return info


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def make_temp_file(data, directory):
    fd, path = tempfile.mkstemp(
        prefix="neural_local_",
        suffix=".chunk",
        dir=directory,
    )

    with os.fdopen(fd, "wb") as f:
        f.write(data)

    return path


def run_candidate(data, candidate, temp_dir):
    input_path = make_temp_file(data, temp_dir)
    output_path = input_path + ".bin"

    start = time.time()

    try:
        import neural_compress_experiment_full as codec

        codec.EXPERIMENT_CONFIG["context_size"] = candidate["context"]
        codec.EXPERIMENT_CONFIG["embed_dim"] = candidate["embed"]
        codec.EXPERIMENT_CONFIG["hidden_dim"] = candidate["hidden"]
        codec.EXPERIMENT_CONFIG["num_states"] = candidate["states"]

        compress_file(
            input_path,
            output_path,
            context_size=candidate["context"],
            lr=0.04,
        )

        elapsed = time.time() - start
        compressed_size = os.path.getsize(output_path)

        return {
            "candidate": candidate["name"],
            "compressed_bytes": compressed_size,
            "elapsed": elapsed,
            "success": True,
        }

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(output_path):
            os.remove(output_path)


def choose_oracle(results):
    return min(
        results,
        key=lambda x: x["compressed_bytes"],
    )


# ---------------------------------------------------------------------------
# V3 feature space
# ---------------------------------------------------------------------------

FEATURE_KEYS = [
    # V2 control features
    "entropy",
    "repetition",
    "unique_fraction",

    # Local arrangement
    "window_entropy_mean",
    "window_entropy_std",
    "change_rate",
    "change_rate_std",

    # Run structure
    "run_mean",
    "run_std",
    "long_run_fraction",

    # Distribution shape
    "top_byte_fraction",
    "top4_byte_fraction",
    "top16_byte_fraction",
    "histogram_std",

    # Adjacent ordering
    "pair_unique_fraction",
    "pair_repeat_rate",
]


def feature_vector(info):
    """
    Normalize heterogeneous features into a comparable Euclidean space.

    The normalization is deliberately fixed and interpretable rather than
    learned from the test set.
    """

    # Run lengths can have a very different scale, so use log1p.
    run_mean = math.log1p(max(0.0, info["run_mean"])) / math.log1p(256.0)
    run_std = math.log1p(max(0.0, info["run_std"])) / math.log1p(256.0)

    return [
        float(info["entropy"]) / 8.0,
        float(info["repetition"]),
        float(info["unique_fraction"]),

        float(info["window_entropy_mean"]) / 8.0,
        float(info["window_entropy_std"]) / 8.0,
        float(info["change_rate"]),
        float(info["change_rate_std"]),

        run_mean,
        run_std,
        float(info["long_run_fraction"]),

        float(info["top_byte_fraction"]),
        float(info["top4_byte_fraction"]),
        float(info["top16_byte_fraction"]),
        float(info["histogram_std"]) * 256.0,

        float(info["pair_unique_fraction"]),
        float(info["pair_repeat_rate"]),
    ]


def feature_distance(a, b):
    av = feature_vector(a)
    bv = feature_vector(b)

    return math.sqrt(
        sum((x - y) ** 2 for x, y in zip(av, bv))
    )


def predict_knn(training_records, info, k=5):
    """
    KNN with the V3 structural feature vector.

    The sort uses an explicit key so equal distances cannot cause Python
    to compare the record dictionaries.
    """

    neighbors = sorted(
        (
            (feature_distance(info, record), record)
            for record in training_records
        ),
        key=lambda x: x[0],
    )[:max(1, min(k, len(training_records)))]

    votes = defaultdict(float)

    for distance, record in neighbors:
        strategy = record["oracle"]["candidate"]
        votes[strategy] += 1.0 / max(distance, 1e-9)

    predicted = max(votes, key=votes.get)

    return predicted, [
        {
            "distance": distance,
            "strategy": record["oracle"]["candidate"],
            "index": record["index"],
            "signature": record["signature"],
        }
        for distance, record in neighbors
    ]


# ---------------------------------------------------------------------------
# Advantage diagnostics
# ---------------------------------------------------------------------------

def candidate_size(results, name):
    for result in results:
        if result["candidate"] == name:
            return result["compressed_bytes"]
    return None


def advantage_diagnostics(results):
    small = candidate_size(results, "small")
    large = candidate_size(results, "large")
    baseline = candidate_size(results, "baseline")

    return {
        "small_minus_large_bytes": (
            small - large
            if small is not None and large is not None
            else None
        ),
        "small_minus_baseline_bytes": (
            small - baseline
            if small is not None and baseline is not None
            else None
        ),
        "large_minus_baseline_bytes": (
            large - baseline
            if large is not None and baseline is not None
            else None
        ),
    }


def weighted_neighbor_advantage(training_records, neighbors):
    """
    Estimate the expected small-vs-large advantage from nearby examples.

    Positive means the nearby training examples favor large.
    Negative means they favor small.
    """

    weighted_sum = 0.0
    weight_total = 0.0

    by_index = {
        record["index"]: record
        for record in training_records
    }

    for neighbor in neighbors:
        record = by_index[neighbor["index"]]
        diagnostics = record.get("advantage", {})
        advantage = diagnostics.get("small_minus_large_bytes")

        if advantage is None:
            continue

        weight = 1.0 / max(neighbor["distance"], 1e-9)
        weighted_sum += weight * advantage
        weight_total += weight

    if weight_total == 0:
        return None

    return weighted_sum / weight_total


def strategy_regret(learner, oracle):
    return learner["compressed_bytes"] - oracle["compressed_bytes"]



# ---------------------------------------------------------------------------
# V4: Adaptive representation allocation
# ---------------------------------------------------------------------------
#
# V3 chooses among fixed codec strategies for each fixed-size chunk.
# V4 keeps that control structure, but adds a lightweight adaptive allocator:
# each chunk is subdivided into fixed candidate regions, each region receives
# a structural complexity score, and the allocator assigns a variable number
# of representation slots under a global budget.
#
# This first experiment intentionally keeps boundaries fixed. That isolates
# the question: "Does variable representation capacity help?" before we also
# allow the system to move chunk boundaries.
# ---------------------------------------------------------------------------

ADAPTIVE_MIN_SLOTS = 1
ADAPTIVE_MAX_SLOTS = 8
ADAPTIVE_TOTAL_SLOTS = 32
ADAPTIVE_REGION_SIZE = 1024


def region_complexity(data):
    """Cheap structural complexity score used only for allocation."""
    if not data:
        return 0.0

    info = analyze_chunk(data)

    # These terms intentionally emphasize local variation and distribution
    # complexity rather than raw size.
    entropy_term = info["entropy"] / 8.0
    local_variation = info["window_entropy_std"]
    change_term = info["change_rate"]
    pair_term = info["pair_unique_fraction"]

    return (
        0.30 * entropy_term
        + 0.25 * local_variation / 4.0
        + 0.25 * change_term
        + 0.20 * pair_term
    )


def adaptive_regions(data, region_size=ADAPTIVE_REGION_SIZE):
    """Split a fixed chunk into candidate regions."""
    return [
        data[i:i + region_size]
        for i in range(0, len(data), region_size)
    ]


def allocate_slots(data,
                   region_size=ADAPTIVE_REGION_SIZE,
                   total_slots=ADAPTIVE_TOTAL_SLOTS,
                   min_slots=ADAPTIVE_MIN_SLOTS,
                   max_slots=ADAPTIVE_MAX_SLOTS):
    """
    Allocate a variable number of representation slots to each region.

    The allocation is deterministic and budgeted. Every region gets a minimum
    allocation; remaining slots are distributed toward structurally complex
    regions. This is intentionally a transparent baseline before replacing
    the allocator with a learned policy.
    """
    regions = adaptive_regions(data, region_size)

    if not regions:
        return []

    scores = [region_complexity(region) for region in regions]
    allocations = [min_slots for _ in regions]

    budget = max(0, total_slots - sum(allocations))

    # Greedy marginal allocation. The diminishing term prevents one region
    # from consuming the entire budget.
    while budget > 0:
        candidates = [
            i for i, allocation in enumerate(allocations)
            if allocation < max_slots
        ]

        if not candidates:
            break

        best = max(
            candidates,
            key=lambda i: scores[i] / (allocations[i] ** 0.5),
        )
        allocations[best] += 1
        budget -= 1

    return [
        {
            "region_index": i,
            "bytes": len(region),
            "complexity": scores[i],
            "slots": allocations[i],
        }
        for i, region in enumerate(regions)
    ]


def adaptive_capacity_summary(data):
    allocations = allocate_slots(data)

    if not allocations:
        return {
            "regions": 0,
            "total_slots": 0,
            "mean_slots": 0.0,
            "min_slots": 0,
            "max_slots": 0,
            "slot_distribution": [],
            "regions_detail": [],
        }

    slots = [x["slots"] for x in allocations]

    return {
        "regions": len(allocations),
        "total_slots": sum(slots),
        "mean_slots": sum(slots) / len(slots),
        "min_slots": min(slots),
        "max_slots": max(slots),
        "slot_distribution": slots,
        "regions_detail": allocations,
    }


def adaptive_proxy_score(data):
    """
    V4 proxy for representation usefulness.

    We do not change the underlying neural codec yet. Instead, this produces
    an explicit adaptive representation plan so the first experiment can
    measure whether structural complexity predicts where capacity should go.
    """
    summary = adaptive_capacity_summary(data)

    if not summary["regions_detail"]:
        return 0.0

    weighted = sum(
        r["complexity"] * r["slots"]
        for r in summary["regions_detail"]
    )

    return weighted / summary["total_slots"]


# ---------------------------------------------------------------------------
# V4 Main
# ---------------------------------------------------------------------------

def main_v4():
    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument(
        "--results",
        default="local_strategy_results_v4_adaptive.json",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        default=ADAPTIVE_REGION_SIZE,
    )
    parser.add_argument(
        "--total-slots",
        type=int,
        default=ADAPTIVE_TOTAL_SLOTS,
    )
    parser.add_argument(
        "--min-slots",
        type=int,
        default=ADAPTIVE_MIN_SLOTS,
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=ADAPTIVE_MAX_SLOTS,
    )

    args = parser.parse_args()

    if args.min_slots < 1:
        raise SystemExit("--min-slots must be >= 1")

    if args.max_slots < args.min_slots:
        raise SystemExit("--max-slots must be >= --min-slots")

    input_path = Path(args.input)

    with open(input_path, "rb") as f:
        data = f.read()

    chunks = [
        data[i:i + args.chunk]
        for i in range(0, len(data), args.chunk)
    ]

    if len(chunks) < 2:
        raise SystemExit(
            "Need at least two chunks for train/test comparison."
        )

    split = max(
        1,
        min(
            len(chunks) - 1,
            int(len(chunks) * args.train_fraction),
        ),
    )

    training_chunks = chunks[:split]
    test_chunks = chunks[split:]

    print("=" * 78)
    print("LOCAL STRUCTURE -> ADAPTIVE REPRESENTATION V4")
    print("=" * 78)
    print(f"File:             {input_path}")
    print(f"Raw size:         {len(data):,} B")
    print(f"Chunk size:       {args.chunk:,} B")
    print(f"Region size:      {args.region_size:,} B")
    print(f"Chunks:           {len(chunks)}")
    print(f"Training chunks:  {len(training_chunks)}")
    print(f"Test chunks:      {len(test_chunks)}")
    print()
    print("Adaptive budget:")
    print(f"  total slots:    {args.total_slots}")
    print(f"  min/region:     {args.min_slots}")
    print(f"  max/region:     {args.max_slots}")
    print()
    print("IMPORTANT: V4 keeps region boundaries fixed.")
    print("Only representation allocation is adaptive.")
    print()

    training_plans = []

    for index, chunk in enumerate(training_chunks, 1):
        plan = adaptive_capacity_summary(chunk)

        training_plans.append({
            "index": index,
            "plan": plan,
            "proxy_score": adaptive_proxy_score(chunk),
        })

        print("-" * 78)
        print(f"TRAIN [{index}/{len(training_chunks)}]")
        print(
            f"regions={plan['regions']} "
            f"slots={plan['total_slots']} "
            f"mean={plan['mean_slots']:.2f} "
            f"range={plan['min_slots']}..{plan['max_slots']}"
        )
        print(f"  allocation: {plan['slot_distribution']}")

    print()
    print("=" * 78)
    print("UNSEEN TEST CHUNKS")
    print("=" * 78)

    test_records = []

    for local_index, chunk in enumerate(test_chunks, 1):
        plan = adaptive_capacity_summary(chunk)
        proxy = adaptive_proxy_score(chunk)

        record = {
            "index": local_index,
            "size": len(chunk),
            "adaptive_plan": plan,
            "proxy_score": proxy,
        }

        test_records.append(record)

        print(
            f"[{local_index}/{len(test_chunks)}] "
            f"regions={plan['regions']} "
            f"slots={plan['total_slots']} "
            f"allocation={plan['slot_distribution']} "
            f"proxy={proxy:.5f}"
        )

    # Distribution diagnostics.
    all_allocations = [
        slot
        for record in test_records
        for slot in record["adaptive_plan"]["slot_distribution"]
    ]

    if all_allocations:
        mean_allocation = sum(all_allocations) / len(all_allocations)
        allocation_histogram = dict(
            sorted(
                Counter(all_allocations).items()
            )
        )
    else:
        mean_allocation = 0.0
        allocation_histogram = {}

    result = {
        "experiment": "local_structure_adaptive_representation_v4",
        "input": str(input_path),
        "chunk_size": args.chunk,
        "region_size": args.region_size,
        "train_fraction": args.train_fraction,
        "adaptive_budget": {
            "total_slots": args.total_slots,
            "min_slots": args.min_slots,
            "max_slots": args.max_slots,
        },
        "design": {
            "fixed_boundaries": True,
            "adaptive_dimension": "representation_count_per_region",
            "purpose": (
                "Test whether variable representation capacity should "
                "depend on local structural complexity."
            ),
        },
        "training": training_plans,
        "test": test_records,
        "diagnostics": {
            "mean_slots_per_region": mean_allocation,
            "slot_histogram": allocation_histogram,
        },
    }

    with open(args.results, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"Mean slots/region: {mean_allocation:.3f}")
    print(f"Slot histogram:    {allocation_histogram}")
    print(f"Results saved:     {args.results}")
    print("=" * 78)


if __name__ == "__main__":
    main_v4()
