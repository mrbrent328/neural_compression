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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument(
        "--results",
        default="local_strategy_results_v3.json",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="number of nearest training chunks used by KNN",
    )

    args = parser.parse_args()

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
    print("LOCAL STRUCTURE -> STRATEGY EXPERIMENT V3")
    print("=" * 78)
    print(f"File:             {input_path}")
    print(f"Raw size:         {len(data):,} B")
    print(f"Chunk size:       {args.chunk:,} B")
    print(f"Chunks:           {len(chunks)}")
    print(f"Training chunks:  {len(training_chunks)}")
    print(f"Test chunks:      {len(test_chunks)}")
    print()
    print("Selector features:")
    print("  V2 control: entropy, repetition, unique_fraction")
    print("  local:      window entropy/change statistics")
    print("  runs:       run mean/std/long-run fraction")
    print("  histogram:  top-byte concentration")
    print("  ordering:   adjacent-pair diversity")
    print()
    print("Candidates:")
    for candidate in CANDIDATES:
        print(" ", candidate)
    print()

    with tempfile.TemporaryDirectory(
        prefix="neural_local_exp_v3_"
    ) as temp_dir:

        training_records = []

        # ------------------------------------------------------------
        # Training
        # ------------------------------------------------------------

        for index, chunk in enumerate(training_chunks, 1):
            info = analyze_chunk(chunk)

            print("-" * 78)
            print(f"TRAIN [{index}/{len(training_chunks)}]")
            print(
                f"entropy={info['entropy']:.3f} "
                f"repeat={info['repetition']:.3f} "
                f"unique={info['unique_fraction']:.3f} "
                f"change={info['change_rate']:.3f} "
                f"win_entropy_std={info['window_entropy_std']:.3f}"
            )

            candidate_results = []

            for candidate in CANDIDATES:
                result = run_candidate(
                    chunk,
                    candidate,
                    temp_dir,
                )
                candidate_results.append(result)

                print(
                    f"  {candidate['name']:10s} "
                    f"{result['compressed_bytes']:7,d} B "
                    f"{result['elapsed']:7.2f}s"
                )

            oracle = choose_oracle(candidate_results)
            advantage = advantage_diagnostics(candidate_results)

            training_records.append({
                "index": index,
                **info,
                "candidates": candidate_results,
                "oracle": oracle,
                "advantage": advantage,
            })

            print(
                f"  ORACLE -> {oracle['candidate']} "
                f"({oracle['compressed_bytes']:,} B)"
            )

        # ------------------------------------------------------------
        # Test
        # ------------------------------------------------------------

        print()
        print("=" * 78)
        print("KNN SELECTOR V3")
        print("=" * 78)
        print(f"k = {args.k}")
        print("Using coarse + fine structural features.")

        fixed_strategy = CANDIDATES[1]

        fixed_total = 0
        oracle_total = 0
        learner_total = 0

        test_records = []

        print()
        print("=" * 78)
        print("UNSEEN TEST CHUNKS")
        print("=" * 78)

        for local_index, chunk in enumerate(test_chunks, 1):
            info = analyze_chunk(chunk)

            candidate_results = [
                run_candidate(
                    chunk,
                    candidate,
                    temp_dir,
                )
                for candidate in CANDIDATES
            ]

            by_name = {
                result["candidate"]: result
                for result in candidate_results
            }

            oracle = choose_oracle(candidate_results)

            learned_name, neighbors = predict_knn(
                training_records,
                info,
                k=args.k,
            )

            learned = by_name[learned_name]
            fixed = by_name[fixed_strategy["name"]]

            fixed_total += fixed["compressed_bytes"]
            oracle_total += oracle["compressed_bytes"]
            learner_total += learned["compressed_bytes"]

            regret = strategy_regret(learned, oracle)
            matched_oracle = learned_name == oracle["candidate"]

            advantage = advantage_diagnostics(candidate_results)
            predicted_advantage = weighted_neighbor_advantage(
                training_records,
                neighbors,
            )

            record = {
                "index": local_index,
                **info,
                "fixed": fixed,
                "learner": learned,
                "oracle": oracle,
                "learned_strategy": learned_name,
                "matched_oracle": matched_oracle,
                "regret_bytes": regret,
                "advantage": advantage,
                "predicted_small_minus_large_bytes": predicted_advantage,
                "knn_neighbors": neighbors,
            }

            test_records.append(record)

            print(
                f"[{local_index}/{len(test_chunks)}] "
                f"sig={info['signature']} "
                f"change={info['change_rate']:.3f} "
                f"win_std={info['window_entropy_std']:.3f} "
                f"fixed={fixed['compressed_bytes']:,} "
                f"learner={learned['compressed_bytes']:,} "
                f"oracle={oracle['compressed_bytes']:,} "
                f"strategy={learned_name} "
                f"match={matched_oracle} "
                f"regret={regret:,} B"
            )

            if predicted_advantage is not None:
                print(
                    f"    predicted small-large advantage: "
                    f"{predicted_advantage:+.1f} B"
                )

            for neighbor in neighbors:
                print(
                    f"    neighbor {neighbor['strategy']:10s} "
                    f"dist={neighbor['distance']:.5f} "
                    f"train={neighbor['index']}"
                )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    raw_test_size = sum(len(c) for c in test_chunks)

    def reduction(size):
        if raw_test_size == 0:
            return 0.0
        return (1.0 - size / raw_test_size) * 100.0

    learner_choices = [
        r["learned_strategy"]
        for r in test_records
    ]

    matched_count = sum(
        1 for r in test_records
        if r["matched_oracle"]
    )

    total_regret = sum(
        r["regret_bytes"]
        for r in test_records
    )

    average_regret = (
        total_regret / len(test_records)
        if test_records
        else 0.0
    )

    result = {
        "experiment": "local_structure_strategy_selection_v3_structural",
        "input": str(input_path),
        "raw_test_bytes": raw_test_size,
        "chunk_size": args.chunk,
        "train_fraction": args.train_fraction,
        "candidates": CANDIDATES,
        "feature_set": {
            "control_v2": [
                "entropy",
                "repetition",
                "unique_fraction",
            ],
            "structural_v3": [
                "window_entropy_mean",
                "window_entropy_std",
                "change_rate",
                "change_rate_std",
                "run_mean",
                "run_std",
                "long_run_fraction",
                "top_byte_fraction",
                "top4_byte_fraction",
                "top16_byte_fraction",
                "histogram_std",
                "pair_unique_fraction",
                "pair_repeat_rate",
            ],
        },
        "knn": {
            "k": args.k,
            "selector_accuracy": (
                matched_count / len(test_records)
                if test_records
                else 0.0
            ),
            "matched_oracle": matched_count,
            "test_count": len(test_records),
            "total_regret_bytes": total_regret,
            "average_regret_bytes": average_regret,
        },
        "test_totals": {
            "fixed_bytes": fixed_total,
            "learner_bytes": learner_total,
            "oracle_bytes": oracle_total,
            "fixed_reduction": reduction(fixed_total),
            "learner_reduction": reduction(learner_total),
            "oracle_reduction": reduction(oracle_total),
        },
        "learner_choices": learner_choices,
        "training": training_records,
        "test": test_records,
    }

    with open(args.results, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"Test raw size:     {raw_test_size:,} B")
    print(
        f"Fixed baseline:    {fixed_total:,} B "
        f"({reduction(fixed_total):.2f}% reduction)"
    )
    print(
        f"Learned selector:  {learner_total:,} B "
        f"({reduction(learner_total):.2f}% reduction)"
    )
    print(
        f"Oracle:            {oracle_total:,} B "
        f"({reduction(oracle_total):.2f}% reduction)"
    )

    if oracle_total:
        print(
            f"Selector/oracle gap: "
            f"{(learner_total / oracle_total - 1) * 100:.4f}%"
        )

    if test_records:
        print(
            f"Selector accuracy:  "
            f"{matched_count}/{len(test_records)} "
            f"({matched_count / len(test_records) * 100:.2f}%)"
        )
    else:
        print("Selector accuracy:  n/a")

    print(f"Total regret:       {total_regret:,} B")
    print(f"Average regret:     {average_regret:.2f} B")
    print(f"Results saved:      {args.results}")
    print("=" * 78)


if __name__ == "__main__":
    main()
