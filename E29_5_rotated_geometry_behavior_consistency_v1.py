#!/usr/bin/env python3
"""
E29.5 — Rotated Geometry Behavior Consistency Test

Question:
Does the behavioral effect follow the intrinsic geometry, or only the
original coordinate axes?

This is an analysis framework for comparing behavior along principal
directions after coordinate rotation.

NOTE:
This version operates on saved probe data. It does not retrain models.

Inputs:
    E27_l2_geometry_replication_results/
        seed_XX/
            geometry_summary.json
            behavior_probes.txt

Outputs:
    E29_5_rotation_behavior_results/
"""

from pathlib import Path
import json
import csv
import math


ROOT = Path("E27_l2_geometry_replication_results")
OUT = Path("E29_5_rotation_behavior_results")
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

    # Handles existing summaries with trailing metadata
    text = text.split('"axis_profile_csv"')[0]
    text = text.rstrip().rstrip(",") + "}"

    return json.loads(text)


def load_probe_summary(path):
    """
    Extract center/high-level probe behavior.

    Detailed probe comparison can be expanded once probe format is finalized.
    """

    values = []

    for line in path.read_text(errors="ignore").splitlines():
        if "logit_RMSE" in line:
            try:
                value = float(
                    line.split("logit_RMSE=")[1].split()[0]
                )
                values.append(value)
            except Exception:
                pass

    return values


def analyze_seed(seed_dir):
    geometry = parse_geometry(seed_dir / "geometry_summary.json")

    eig = geometry["eigenvectors_columns"]

    rows = []

    for angle in ANGLES:
        low_rot = rotate_vector(eig[0], angle)
        high_rot = rotate_vector(eig[1], angle)

        rows.append({
            "seed": geometry["seed"],
            "angle": angle,
            "low_x": low_rot[0],
            "low_y": low_rot[1],
            "high_x": high_rot[0],
            "high_y": high_rot[1],
            "low_norm": math.sqrt(low_rot[0]**2 + low_rot[1]**2),
            "high_norm": math.sqrt(high_rot[0]**2 + high_rot[1]**2),
        })

    return rows


def main():
    rows = []

    for seed in sorted(ROOT.glob("seed_*")):
        if (seed / "geometry_summary.json").exists():
            print("Processing", seed.name)
            rows.extend(analyze_seed(seed))

    with (OUT / "rotation_behavior_geometry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment": "E29.5 rotated geometry behavior consistency",
        "seeds": len(set(r["seed"] for r in rows)),
        "angles": ANGLES,
        "note": "Initial geometry rotation preparation. Behavioral probe alignment follows."
    }

    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    print("=" * 80)
    print("E29.5 COMPLETE")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
