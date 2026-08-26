#!/usr/bin/env python3
"""
E28.5 — Cross Seed Geometry/Behavior Summary

Reads E28 output and creates a compact summary table.

Goal:
Determine whether stronger latent anisotropy corresponds to stronger
behavioral separation between low/high curvature directions.

Inputs:
    E28_geometry_behavior_results/geometry_behavior_report.json

Outputs:
    E28_geometry_behavior_results/e28_cross_seed_summary.csv
    E28_geometry_behavior_results/e28_cross_seed_summary.json
"""

from pathlib import Path
import json
import csv
import math


INPUT = Path("E28_geometry_behavior_results/geometry_behavior_report.json")
OUT_DIR = Path("E28_geometry_behavior_results")

CSV_OUT = OUT_DIR / "e28_cross_seed_summary.csv"
JSON_OUT = OUT_DIR / "e28_cross_seed_summary.json"


def safe_ratio(a, b):
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return a / b


def correlation(xs, ys):
    if len(xs) < 2:
        return None

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)

    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))

    if dx == 0 or dy == 0:
        return None

    return num / (dx * dy)


def main():
    data = json.loads(INPUT.read_text())

    rows = []
    curvature = []
    separation = []

    for item in data:
        eigen = item.get("eigenvalues", [])

        if len(eigen) >= 2:
            low_curv = eigen[0]
            high_curv = eigen[1]
            curvature_ratio = safe_ratio(low_curv, high_curv)
        else:
            low_curv = None
            high_curv = None
            curvature_ratio = None

        low_s = item.get("low_mean_sensitivity")
        high_s = item.get("high_mean_sensitivity")

        sensitivity_ratio = safe_ratio(high_s, low_s)

        row = {
            "seed": item.get("seed"),
            "low_curvature": low_curv,
            "high_curvature": high_curv,
            "curvature_ratio_low_high": curvature_ratio,
            "low_sensitivity": low_s,
            "high_sensitivity": high_s,
            "behavior_separation_high_over_low": sensitivity_ratio,
        }

        rows.append(row)

        if curvature_ratio is not None and sensitivity_ratio is not None:
            curvature.append(curvature_ratio)
            separation.append(sensitivity_ratio)

    report = {
        "seeds_analyzed": len(rows),
        "correlation_curvature_vs_behavior_separation":
            correlation(curvature, separation),
        "rows": rows,
    }

    CSV_OUT.write_text("")
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    JSON_OUT.write_text(json.dumps(report, indent=2))

    print("=" * 80)
    print("E28.5 COMPLETE")
    print("=" * 80)
    print("Seeds:", len(rows))
    print("Correlation:", report["correlation_curvature_vs_behavior_separation"])
    print("CSV:", CSV_OUT)
    print("JSON:", JSON_OUT)


if __name__ == "__main__":
    main()
