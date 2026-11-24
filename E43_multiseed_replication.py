#!/usr/bin/env python3
"""Run the E43 Phase 1 synthetic protocol across multiple independent seeds.

This is a replication harness for E43_latent_communication.py. It launches the
unchanged single-seed experiment in a fresh process for each seed, preserves
each raw JSON result, and writes an aggregate summary. It does not change the
underlying task, model, optimizer, split, or channel.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

METRICS = (
    "learned_quantized_latent",
    "shuffled_quantized_latent",
    "random_quantized_latent",
    "zero_quantized_latent",
    "no_message",
)


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(43, 53)),
        help="Independent random seeds (default: 43 through 52).",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--script", type=Path,
        default=Path(__file__).with_name("E43_latent_communication.py"),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/E43/multiseed"),
        help="Directory for the per-seed raw JSON results.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("runs/E43/e43_multiseed_summary.json"),
        help="Aggregate summary JSON path.",
    )
    args = parser.parse_args()

    if not args.script.is_file():
        raise SystemExit(f"Experiment script not found: {args.script}")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must contain at least one unique seed")
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for seed in args.seeds:
        raw_path = args.run_dir / f"e43_phase1_seed_{seed}.json"
        command = [
            sys.executable, str(args.script),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--out", str(raw_path),
        ]
        print(f"Running seed {seed}: {' '.join(command)}", flush=True)
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Seed {seed} failed with exit code {completed.returncode}.\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if not raw_path.is_file():
            raise RuntimeError(f"Seed {seed} did not produce {raw_path}")

        result = json.loads(raw_path.read_text())
        if result.get("experiment") != "E43_phase1_synthetic_protocol_smoke_test_v2":
            raise RuntimeError(f"Unexpected experiment output for seed {seed}")
        if result.get("seed") != seed:
            raise RuntimeError(f"Seed mismatch in output file {raw_path}")
        missing = [name for name in METRICS if name not in result["results"]]
        if missing:
            raise RuntimeError(f"Seed {seed} is missing metrics: {missing}")

        runs.append({
            "seed": seed,
            "raw_result": str(raw_path),
            "device": result.get("device"),
            "results": {
                name: result["results"][name]
                for name in METRICS
            },
            "learning_diagnostics": result.get("learning_diagnostics", {}),
            "accounting": result.get("accounting", {}),
        })
        learned = result["results"]["learned_quantized_latent"]["accuracy"]
        print(f"Seed {seed}: learned accuracy={learned:.4%}", flush=True)

    aggregate = {}
    for metric in METRICS:
        aggregate[metric] = summarize([
            run["results"][metric]["accuracy"] for run in runs
        ])

    paired_differences = {}
    for control in METRICS[1:]:
        differences = [
            run["results"]["learned_quantized_latent"]["accuracy"]
            - run["results"][control]["accuracy"]
            for run in runs
        ]
        paired_differences[f"learned_minus_{control}"] = {
            **summarize(differences),
            "seeds_learned_higher": sum(value > 0 for value in differences),
            "seeds_tied": sum(value == 0 for value in differences),
            "seeds_learned_lower": sum(value < 0 for value in differences),
        }

    summary = {
        "experiment": "E43_phase1_multiseed_replication",
        "status": (
            "synthetic protocol replication only; not a pretrained-LM test "
            "and not evidence of a compression advantage"
        ),
        "protocol": {
            "single_seed_script": str(args.script),
            "epochs_per_seed": args.epochs,
            "seeds": args.seeds,
            "independent_process_per_seed": True,
            "task_architecture_optimizer_split_and_channel": (
                "unchanged from E43_latent_communication.py"
            ),
            "chance_accuracy_for_64_balanced_classes": 1.0 / 64,
        },
        "aggregate_accuracy": aggregate,
        "paired_accuracy_differences": paired_differences,
        "runs": runs,
        "interpretation_warning": (
            "Report all seeds and dispersion. This harness is descriptive; "
            "do not infer generalization beyond this synthetic task, claim "
            "statistical significance from the summary alone, or compare "
            "against the symbolic text-oracle diagnostic as a matched baseline."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "summary_path": str(args.out),
        "seed_count": len(runs),
        "aggregate_accuracy": aggregate,
        "paired_accuracy_differences": paired_differences,
    }, indent=2))


if __name__ == "__main__":
    main()
