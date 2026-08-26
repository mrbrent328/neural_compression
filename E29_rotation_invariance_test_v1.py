#!/usr/bin/env python3
"""
E29 — Latent Coordinate Rotation Invariance Test

Question:
Is the discovered L2 geometry intrinsic to the latent representation,
or dependent on the arbitrary coordinate axes?

Method:
For each seed:
1. Load the learned center and Hessian.
2. Rotate the coordinate system by several angles.
3. Transform the Hessian into the rotated basis:
       H' = R^T H R
4. Compare eigenvalues and principal structure.
5. Measure whether the geometry survives rotation.

This is an analysis-only experiment.
No model retraining required.

Inputs:
    E27_l2_geometry_replication_results/seed_XX/geometry_summary.json

Outputs:
    E29_rotation_invariance_results/
"""

from pathlib import Path
import json
import csv
import math


ROOT = Path("E27_l2_geometry_replication_results")
OUT = Path("E29_rotation_invariance_results")
OUT.mkdir(exist_ok=True)

ANGLES = [
    0, 15, 30, 45, 60, 75, 90,
    120, 135, 150
]


def rotate_hessian(H, degrees):
    theta = math.radians(degrees)

    c = math.cos(theta)
    s = math.sin(theta)

    R = [
        [c, -s],
        [s, c]
    ]

    # R^T H R
    RtHR = [[0.0, 0.0], [0.0, 0.0]]

    for i in range(2):
        for j in range(2):
            total = 0.0
            for a in range(2):
                for b in range(2):
                    total += R[a][i] * H[a][b] * R[b][j]
            RtHR[i][j] = total

    return RtHR


def eigenvalues_2x2(H):
    a = H[0][0]
    b = H[0][1]
    c = H[1][1]

    trace = a + c
    det = a*c - b*b

    disc = max(0.0, trace*trace - 4*det)

    return sorted([
        (trace - math.sqrt(disc))/2,
        (trace + math.sqrt(disc))/2
    ])


def analyze_seed(seed_dir):
    data = json.loads(
        (seed_dir / "geometry_summary.json")
        .read_text()
        .split('"axis_profile_csv"')[0]
        .rstrip()
        .rstrip(",") + "}"
    )

    H = data["hessian"]
    original = eigenvalues_2x2(H)

    rows = []

    for angle in ANGLES:
        rotated = rotate_hessian(H, angle)
        eig = eigenvalues_2x2(rotated)

        rows.append({
            "seed": data["seed"],
            "angle": angle,
            "original_low": original[0],
            "original_high": original[1],
            "rotated_low": eig[0],
            "rotated_high": eig[1],
            "low_error": abs(eig[0]-original[0]),
            "high_error": abs(eig[1]-original[1]),
        })

    return rows


def main():
    all_rows = []

    for seed in sorted(ROOT.glob("seed_*")):
        if (seed / "geometry_summary.json").exists():
            print("Processing", seed.name)
            all_rows.extend(analyze_seed(seed))

    csv_path = OUT / "rotation_results.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=all_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "experiment": "E29 coordinate rotation invariance",
        "seeds": len(set(r["seed"] for r in all_rows)),
        "angles": ANGLES,
        "max_low_error": max(r["low_error"] for r in all_rows),
        "max_high_error": max(r["high_error"] for r in all_rows),
    }

    (OUT / "rotation_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("="*80)
    print("E29 COMPLETE")
    print("CSV:", csv_path)
    print("Report:", OUT / "rotation_report.json")


if __name__ == "__main__":
    main()
