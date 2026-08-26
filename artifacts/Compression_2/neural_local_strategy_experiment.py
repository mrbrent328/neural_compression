#!/usr/bin/env python3
"""
neural_local_strategy_experiment.py

Experiment: fixed vs oracle vs learned local strategy selection.

This is deliberately a research harness, not a new codec.

Idea:
    file -> chunks -> local structural signature
                         |
                         +-> candidate strategies
                         |
                         +-> measured coding cost
                         |
                         +-> winner remembered as signature -> strategy

The first version uses the existing neural compressor as the candidate
engine. It does NOT change the known-good synchronized codec.

Usage:
    python3 neural_local_strategy_experiment.py /path/to/file

Optional:
    --chunk 8192
    --train-fraction 0.5
    --results local_strategy_results.json
    --keep-artifacts
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

# Import the existing compressor implementation.
# The experiment script should live beside neural_compress_experiment_full.py.
try:
    from neural_compress_experiment_full import compress_file
except ImportError:
    from neural_compress_opt_batch import compress_file


# ------------------------------------------------------------
# Candidate strategies
# ------------------------------------------------------------
#
# Start small. We can expand this list after seeing whether
# local selection actually produces a useful signal.
#
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


def entropy(data):
    """Shannon entropy in bits/byte."""
    if not data:
        return 0.0

    counts = Counter(data)
    n = len(data)

    return -sum(
        (count / n) * math.log2(count / n)
        for count in counts.values()
    )


def repetition_score(data):
    """
    Cheap local structure signal.

    Measures the fraction of adjacent byte pairs that repeat.
    This is intentionally simple for the first experiment.
    """
    if len(data) < 2:
        return 0.0

    repeats = sum(
        1 for a, b in zip(data, data[1:]) if a == b
    )

    return repeats / (len(data) - 1)


def unique_byte_fraction(data):
    """Number of unique byte values divided by 256."""
    if not data:
        return 0.0

    return len(set(data)) / 256.0


def signature(data):
    """
    Structural signature.

    We intentionally use coarse statistics rather than the literal
    bytes so the learner has a chance to generalize to unseen chunks.
    """
    e = entropy(data)
    r = repetition_score(data)
    u = unique_byte_fraction(data)

    # Quantize so similar chunks land in the same bucket.
    return (
        round(e, 1),
        round(r, 2),
        round(u, 2),
    )


def analyze_chunk(data):
    return {
        "size": len(data),
        "entropy": entropy(data),
        "repetition": repetition_score(data),
        "unique_fraction": unique_byte_fraction(data),
        "signature": signature(data),
    }


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
    """
    Measure a candidate on one chunk using the existing compressor.

    The output is intentionally discarded after measurement.
    The existing codec remains the source of truth.
    """
    input_path = make_temp_file(data, temp_dir)
    output_path = input_path + ".bin"

    start = time.time()

    try:
        # The experiment harness accepts CLI configuration through globals,
        # so set the configuration explicitly before invoking compress_file.
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


def learn_strategy_map(training_records):
    """
    Learn signature -> strategy using only training chunks.

    If a signature has appeared multiple times, choose the strategy
    with the best average measured size.
    """
    buckets = defaultdict(lambda: defaultdict(list))

    for record in training_records:
        sig = tuple(record["signature"])
        winner = record["oracle"]["candidate"]
        buckets[sig][winner].append(
            record["oracle"]["compressed_bytes"]
        )

    strategy_map = {}

    for sig, strategies in buckets.items():
        averages = {
            strategy: statistics.mean(values)
            for strategy, values in strategies.items()
        }

        strategy_map[sig] = min(
            averages,
            key=averages.get,
        )

    return strategy_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument(
        "--results",
        default="local_strategy_results.json",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="reserved for future versions",
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
    print("LOCAL STRUCTURE → STRATEGY EXPERIMENT")
    print("=" * 78)
    print(f"File:             {input_path}")
    print(f"Raw size:         {len(data):,} B")
    print(f"Chunk size:       {args.chunk:,} B")
    print(f"Chunks:           {len(chunks)}")
    print(f"Training chunks:  {len(training_chunks)}")
    print(f"Test chunks:      {len(test_chunks)}")
    print()
    print("Candidates:")
    for candidate in CANDIDATES:
        print(" ", candidate)
    print()

    with tempfile.TemporaryDirectory(prefix="neural_local_exp_") as temp_dir:
        # --------------------------------------------------------
        # Phase 1: measure candidates on training chunks
        # --------------------------------------------------------
        training_records = []

        for index, chunk in enumerate(training_chunks, 1):
            info = analyze_chunk(chunk)

            print("-" * 78)
            print(f"TRAIN [{index}/{len(training_chunks)}]")
            print(
                f"entropy={info['entropy']:.3f} "
                f"repeat={info['repetition']:.3f} "
                f"unique={info['unique_fraction']:.3f}"
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

            training_records.append({
                "index": index,
                **info,
                "candidates": candidate_results,
                "oracle": oracle,
            })

            print(
                f"  ORACLE → {oracle['candidate']} "
                f"({oracle['compressed_bytes']:,} B)"
            )

        strategy_map = learn_strategy_map(training_records)

        print()
        print("=" * 78)
        print("LEARNED STRATEGY MAP")
        print("=" * 78)

        for sig, strategy in strategy_map.items():
            print(f"{sig} -> {strategy}")

        # --------------------------------------------------------
        # Phase 2: unseen test chunks
        # --------------------------------------------------------
        fixed_strategy = CANDIDATES[1]  # known baseline

        fixed_total = 0
        oracle_total = 0
        learner_total = 0

        test_records = []

        print()
        print("=" * 78)
        print("UNSEEN TEST CHUNKS")
        print("=" * 78)

        name_to_candidate = {
            c["name"]: c
            for c in CANDIDATES
        }

        for local_index, chunk in enumerate(test_chunks, 1):
            info = analyze_chunk(chunk)

            candidate_results = []

            for candidate in CANDIDATES:
                candidate_results.append(
                    run_candidate(
                        chunk,
                        candidate,
                        temp_dir,
                    )
                )

            by_name = {
                result["candidate"]: result
                for result in candidate_results
            }

            oracle = choose_oracle(candidate_results)

            sig = tuple(info["signature"])
            learned_name = strategy_map.get(
                sig,
                fixed_strategy["name"],
            )

            learned = by_name[learned_name]
            fixed = by_name[fixed_strategy["name"]]

            fixed_total += fixed["compressed_bytes"]
            oracle_total += oracle["compressed_bytes"]
            learner_total += learned["compressed_bytes"]

            record = {
                "index": local_index,
                **info,
                "fixed": fixed,
                "learner": learned,
                "oracle": oracle,
                "learned_strategy": learned_name,
            }

            test_records.append(record)

            print(
                f"[{local_index}/{len(test_chunks)}] "
                f"sig={sig} "
                f"fixed={fixed['compressed_bytes']:,} "
                f"learner={learned['compressed_bytes']:,} "
                f"oracle={oracle['compressed_bytes']:,} "
                f"strategy={learned_name}"
            )

    raw_test_size = sum(len(c) for c in test_chunks)

    def reduction(size):
        if raw_test_size == 0:
            return 0.0
        return (1.0 - size / raw_test_size) * 100.0

    learner_choices = [
        r["learned_strategy"]
        for r in test_records
    ]

    result = {
        "experiment": "local_structure_strategy_selection_v1",
        "input": str(input_path),
        "raw_test_bytes": raw_test_size,
        "chunk_size": args.chunk,
        "train_fraction": args.train_fraction,
        "candidates": CANDIDATES,
        "training_chunks": len(training_chunks),
        "test_chunks": len(test_chunks),
        "strategy_map": {
            str(k): v
            for k, v in strategy_map.items()
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

    if fixed_total:
        print(
            f"Selector vs fixed: "
            f"{(1 - learner_total / fixed_total) * 100:.2f}% smaller"
        )

    if oracle_total:
        print(
            f"Selector/oracle gap: "
            f"{(learner_total / oracle_total - 1) * 100:.2f}%"
        )

    print(f"Results saved:     {args.results}")
    print("=" * 78)


if __name__ == "__main__":
    main()
