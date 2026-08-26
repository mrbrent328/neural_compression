#!/usr/bin/env python3
"""
E28 — GEOMETRY TO BEHAVIOR SENSITIVITY

Goal:
Test whether Hessian geometry predicts functional sensitivity.

Question:
Do low-curvature latent directions change behavior less than high-curvature
directions for the same latent displacement?

Inputs expected:
E27_l2_geometry_replication_results/
    seed_XX/
        geometry_summary.json
        behavior_probes.txt

Outputs:
E28_geometry_behavior_results/
    seed_summary.csv
    geometry_behavior_report.json

This is intentionally an analysis pass. It does NOT retrain models.
"""

from pathlib import Path
import json
import csv
import re
import math

ROOT = Path("E27_l2_geometry_replication_results")
OUT = Path("E28_geometry_behavior_results")
OUT.mkdir(exist_ok=True)


def extract_probe_points(path):
    """
    Parse behavior probes.

    Expected examples:
    eig_low_-0.500 ... exact=14/24 ... logit_RMSE=...
    """
    rows = []

    text = path.read_text(errors="ignore")

    pattern = re.compile(
        r"(eig_(?:low|high)_[+-]?\d+\.\d+).*?"
        r"exact=(\d+)/(\d+).*?"
        r"logit_RMSE=([0-9.eE+-]+)",
        re.S,
    )

    for name, exact, total, rmse in pattern.findall(text):
        direction, offset = name.rsplit("_", 1)

        rows.append({
            "probe": name,
            "direction": direction,
            "offset": float(offset),
            "exact": int(exact),
            "total": int(total),
            "logit_rmse": float(rmse),
        })

    return rows


def analyze_seed(seed_dir):
    geometry = json.loads(
        (seed_dir / "geometry_summary.json").read_text().split(
            '"axis_profile_csv"'
        )[0].rstrip().rstrip(",") + "}"
    )

    probes = extract_probe_points(
        seed_dir / "behavior_probes.txt"
    )

    center = [
        float(x) for x in geometry["center"]
    ]

    results = []

    for p in probes:
        distance = abs(p["offset"])

        results.append({
            **p,
            "latent_distance": distance,
            "rmse_per_distance": (
                abs(p["logit_rmse"] - next(
                    (
                        x["logit_rmse"]
                        for x in probes
                        if x["offset"] == 0
                        and x["direction"] == p["direction"]
                    ),
                    p["logit_rmse"]
                )) / distance
                if distance > 0 else 0.0
            )
        })

    low = [
        x for x in results
        if x["direction"] == "eig_low"
        and x["offset"] != 0
    ]

    high = [
        x for x in results
        if x["direction"] == "eig_high"
        and x["offset"] != 0
    ]

    return {
        "seed": int(seed_dir.name.split("_")[1]),
        "eigenvalues": geometry.get("eigenvalues"),
        "low_mean_sensitivity": (
            sum(x["rmse_per_distance"] for x in low) / len(low)
            if low else None
        ),
        "high_mean_sensitivity": (
            sum(x["rmse_per_distance"] for x in high) / len(high)
            if high else None
        ),
        "low_points": low,
        "high_points": high,
    }


def main():
    summaries = []

    for seed_dir in sorted(ROOT.glob("seed_*")):
        if (seed_dir / "geometry_summary.json").exists() and (
            seed_dir / "behavior_probes.txt"
        ).exists():
            try:
                summaries.append(analyze_seed(seed_dir))
                print("processed", seed_dir.name)
            except Exception as e:
                print("failed", seed_dir.name, e)

    with (OUT / "geometry_behavior_report.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    with (OUT / "seed_summary.csv").open("w", newline="") as f:
        fields = [
            "seed",
            "low_mean_sensitivity",
            "high_mean_sensitivity",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                k: row[k] for k in fields
            })

    print()
    print("E28 COMPLETE")
    print("Seeds:", len(summaries))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
