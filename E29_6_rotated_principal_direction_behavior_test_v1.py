#!/usr/bin/env python3
"""
E29.6 — Rotated Principal Direction Behavior Test

Goal:
Test whether behavioral effects follow intrinsic latent geometry.

Question:
If we rotate the latent coordinate system, do the principal directions
still correspond to the same behavioral sensitivity?

This version prepares the comparison using existing E27 artifacts.

Inputs:
    E27_l2_geometry_replication_results/
        seed_XX/
            geometry_summary.json
            behavior_probes.txt

Outputs:
    E29_6_rotated_behavior_results/
"""

from pathlib import Path
import json
import csv
import math
import re


ROOT = Path("E27_l2_geometry_replication_results")
OUT = Path("E29_6_rotated_behavior_results")
OUT.mkdir(exist_ok=True)

ANGLES = [0, 15, 30, 45, 60, 75, 90]


def rotate_vector(v, degrees):
    theta = math.radians(degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    return [
        c * v[0] - s * v[1],
        s * v[0] + c * v[1],
    ]


def parse_geometry(path):
    text = path.read_text()
    text = text.split('"axis_profile_csv"')[0]
    text = text.rstrip().rstrip(",") + "}"
    return json.loads(text)


def parse_behavior(path):
    """
    Extract probe behavior values from saved probe logs.

    Keeps this intentionally tolerant because probe formatting changed
    during earlier experiments.
    """

    results = []

    for line in path.read_text(errors="ignore").splitlines():
        if "logit_RMSE=" in line:
            match = re.search(
                r"logit_RMSE=([0-9.eE+-]+)",
                line
            )

            if match:
                results.append(float(match.group(1)))

    return results


def analyze_seed(seed_dir):
    geometry = parse_geometry(seed_dir / "geometry_summary.json")

    vectors = geometry["eigenvectors_columns"]

    behavior = parse_behavior(
        seed_dir / "behavior_probes.txt"
    )

    rows = []

    for angle in ANGLES:
        low = rotate_vector(vectors[0], angle)
        high = rotate_vector(vectors[1], angle)

        rows.append({
            "seed": geometry["seed"],
            "angle": angle,
            "rotated_low_x": low[0],
            "rotated_low_y": low[1],
            "rotated_high_x": high[0],
            "rotated_high_y": high[1],
            "behavior_samples_found": len(behavior),
        })

    return rows


def main():
    rows = []

    for seed in sorted(ROOT.glob("seed_*")):
        if (
            (seed / "geometry_summary.json").exists()
            and (seed / "behavior_probes.txt").exists()
        ):
            print("Processing", seed.name)
            rows.extend(analyze_seed(seed))

    csv_path = OUT / "rotated_behavior_alignment.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment": "E29.6 rotated principal direction behavior test",
        "seeds": len(set(r["seed"] for r in rows)),
        "angles": ANGLES,
        "status": "rotation mapping prepared; behavior comparison pending refinement"
    }

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("=" * 80)
    print("E29.6 COMPLETE")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
