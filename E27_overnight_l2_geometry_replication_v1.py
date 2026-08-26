
#!/usr/bin/env python3
"""
E27 — OVERNIGHT L2 GEOMETRY REPLICATION + CONSISTENCY TEST

Purpose
-------
Use a long unattended CPU window to test whether the L2 geometry observed in
E25 (seed 18) and E26 (seed 21) repeats across fresh seeds.

IMPORTANT:
- This script is intentionally conservative.
- It does NOT change the model architecture or objectives.
- It runs the same single-seed geometry experiment repeatedly.
- Seeds 21 is NOT rerun by default because it already exists from E26.
- Fresh seeds default to 22..30 (9 new seeds).
- Each seed:
    1. trains the shared constructor
    2. trains the relational latent objective
    3. computes the local Hessian
    4. extracts the two principal curvature directions
    5. sweeps both directions
    6. evaluates behavior at every sweep point
    7. writes a compact per-seed result
- At the end, it computes cross-seed statistics and a simple "geometry
  consistency" report.

The script is written as a self-contained experiment driver, but it expects
the same project/model-loading helpers used by E25/E26 to be available.
This file deliberately does not silently invent model paths or architecture
parameters that are not known here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = list(range(22, 31))   # E26 seed 21 already completed
DEFAULT_OFFSETS = [-1.0, -0.75, -0.5, -0.375, -0.25, -0.125,
                   0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0]

# The two reference runs we already have.
REFERENCE_RUNS = {
    18: {
        "low_curvature": 0.014798400923609734,
        "high_curvature": 0.23460958898067474,
        "low_high_ratio": 0.014798400923609734 / 0.23460958898067474,
    },
    21: {
        "low_curvature": 0.012773018330335617,
        "high_curvature": 0.14245086908340454,
        "low_high_ratio": 0.012773018330335617 / 0.14245086908340454,
    },
}

OUT_ROOT = Path("E27_overnight_l2_geometry_replication_results")


# ---------------------------------------------------------------------------
# SMALL UTILITIES
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def finite(x: float) -> bool:
    return math.isfinite(float(x))


def safe_mean(xs: list[float]) -> float:
    ys = [float(x) for x in xs if finite(x)]
    return sum(ys) / len(ys) if ys else float("nan")


def safe_median(xs: list[float]) -> float:
    ys = sorted(float(x) for x in xs if finite(x))
    if not ys:
        return float("nan")
    n = len(ys)
    return ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def safe_std(xs: list[float]) -> float:
    ys = [float(x) for x in xs if finite(x)]
    if len(ys) < 2:
        return float("nan")
    m = sum(ys) / len(ys)
    return math.sqrt(sum((x - m) ** 2 for x in ys) / (len(ys) - 1))


def fmt(x: float) -> str:
    return "nan" if not finite(x) else f"{x:.6f}"


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CORE TEST CONTRACT
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    seed: int
    direction: str
    offset: float
    z1: float
    z2: float
    weight_error: float
    greedy: float
    exact: int
    total: int
    logit_rmse: float


@dataclass
class SeedGeometry:
    seed: int
    center_z1: float
    center_z2: float
    grad_norm: float
    eig_low: float
    eig_high: float
    curvature_ratio: float
    low_direction_x: float
    low_direction_y: float
    high_direction_x: float
    high_direction_y: float
    low_best_exact: int
    low_center_exact: int
    high_best_exact: int
    high_center_exact: int
    low_best_greedy: float
    high_best_greedy: float
    low_center_greedy: float
    high_center_greedy: float
    low_plateau_width_exact_ge_center: float
    high_drop_at_plus_1: int
    high_drop_at_minus_1: int


# ---------------------------------------------------------------------------
# ADAPTER
# ---------------------------------------------------------------------------
#
# The experiment-specific functions below are intentionally isolated.
# Copy the corresponding implementations from E25/E26 into this section
# if they are not already importable from the project.
#
# This prevents accidental "creative" changes to the scientific machinery.
# ---------------------------------------------------------------------------

def load_experiment_module():
    """
    Try to import E25/E26 helpers from the current directory.

    If your E25/E26 code is a standalone script with functions defined in
    __main__, this raises a clear error rather than pretending the overnight
    run succeeded.
    """
    candidates = [
        "E26_l2_geometry_replication_v1",
        "E25_l2_eigenvector_sweep_v1",
        "E24_l2_coordinate_geometry_v1",
    ]

    for name in candidates:
        try:
            return __import__(name)
        except Exception:
            continue

    raise RuntimeError(
        "Could not import the existing E25/E26 experiment helpers. "
        "Copy the experiment-specific training/evaluation functions into "
        "the ADAPTER section of this script before starting the overnight run."
    )


def run_one_seed(seed: int, offsets: list[float]) -> tuple[SeedGeometry, list[SweepPoint]]:
    """
    Adapter around the existing E25/E26 experiment.

    Expected module-level API:
        run_geometry_for_seed(seed, offsets)

    returning:
        {
          "center": [z1,z2],
          "gradient": [g1,g2],
          "eigenvalues": [low,high],
          "eigenvectors": [[low_x,high_x],[low_y,high_y]],
          "behavior": [
             {"direction": ..., "offset": ..., "z": [...],
              "weight_error": ..., "greedy": ..., "exact": ...,
              "total": 24, "logit_rmse": ...},
             ...
          ]
        }

    This adapter deliberately requires the existing machinery instead of
    recreating it from memory.
    """
    mod = load_experiment_module()

    if not hasattr(mod, "run_geometry_for_seed"):
        raise RuntimeError(
            f"{mod.__name__} does not expose run_geometry_for_seed(). "
            "Refusing to run a modified/guessed version of the experiment."
        )

    result = mod.run_geometry_for_seed(seed=seed, offsets=offsets)

    center = result["center"]
    grad = result["gradient"]
    eigvals = result["eigenvalues"]
    eigvecs = result["eigenvectors"]

    low, high = sorted([float(eigvals[0]), float(eigvals[1])])
    ratio = low / high if high != 0 else float("nan")

    # Normalize eigenvector interpretation to low/high by matching eigenvalue
    # order. The source experiment is expected to return columns in
    # eigenvalue order.
    low_vec = (float(eigvecs[0][0]), float(eigvecs[1][0]))
    high_vec = (float(eigvecs[0][1]), float(eigvecs[1][1]))

    points: list[SweepPoint] = []
    for p in result["behavior"]:
        points.append(
            SweepPoint(
                seed=seed,
                direction=str(p["direction"]),
                offset=float(p["offset"]),
                z1=float(p["z"][0]),
                z2=float(p["z"][1]),
                weight_error=float(p["weight_error"]),
                greedy=float(p["greedy"]),
                exact=int(p["exact"]),
                total=int(p.get("total", 24)),
                logit_rmse=float(p["logit_rmse"]),
            )
        )

    low_pts = [p for p in points if p.direction == "eig_low"]
    high_pts = [p for p in points if p.direction == "eig_high"]

    def exact_at(points_: list[SweepPoint], offset: float) -> int:
        for p in points_:
            if abs(p.offset - offset) < 1e-9:
                return p.exact
        return -1

    def greedy_at(points_: list[SweepPoint], offset: float) -> float:
        for p in points_:
            if abs(p.offset - offset) < 1e-9:
                return p.greedy
        return float("nan")

    low_center = exact_at(low_pts, 0.0)
    high_center = exact_at(high_pts, 0.0)
    low_center_g = greedy_at(low_pts, 0.0)
    high_center_g = greedy_at(high_pts, 0.0)

    low_best = max((p.exact for p in low_pts), default=-1)
    high_best = max((p.exact for p in high_pts), default=-1)
    low_best_g = max((p.greedy for p in low_pts), default=float("nan"))
    high_best_g = max((p.greedy for p in high_pts), default=float("nan"))

    # Width of contiguous sampled low-curvature region whose exact score is
    # at least the center score. This is intentionally simple and descriptive,
    # not a statistical confidence interval.
    good_low = [p.offset for p in low_pts if p.exact >= low_center]
    plateau_width = (
        max(good_low) - min(good_low) if good_low else float("nan")
    )

    return (
        SeedGeometry(
            seed=seed,
            center_z1=float(center[0]),
            center_z2=float(center[1]),
            grad_norm=math.sqrt(sum(float(x) ** 2 for x in grad)),
            eig_low=low,
            eig_high=high,
            curvature_ratio=ratio,
            low_direction_x=low_vec[0],
            low_direction_y=low_vec[1],
            high_direction_x=high_vec[0],
            high_direction_y=high_vec[1],
            low_best_exact=low_best,
            low_center_exact=low_center,
            high_best_exact=high_best,
            high_center_exact=high_center,
            low_best_greedy=low_best_g,
            high_best_greedy=high_best_g,
            low_center_greedy=low_center_g,
            high_center_greedy=high_center_g,
            low_plateau_width_exact_ge_center=plateau_width,
            high_drop_at_plus_1=(
                high_center - exact_at(high_pts, +1.0)
                if high_center >= 0 and exact_at(high_pts, +1.0) >= 0
                else -1
            ),
            high_drop_at_minus_1=(
                high_center - exact_at(high_pts, -1.0)
                if high_center >= 0 and exact_at(high_pts, -1.0) >= 0
                else -1
            ),
        ),
        points,
    )


# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------

def summarize(rows: list[SeedGeometry]) -> dict:
    ratios = [r.curvature_ratio for r in rows]
    eig_low = [r.eig_low for r in rows]
    eig_high = [r.eig_high for r in rows]

    # "same kind of terrain" indicators:
    # 1) low curvature substantially below high curvature
    # 2) low-direction plateau at least one sampled step wide
    # 3) high direction causes measurable degradation at an extreme
    terrain_flags = []
    for r in rows:
        terrain_flags.append(
            {
                "seed": r.seed,
                "low_flatter_than_high": r.eig_low < r.eig_high,
                "curvature_ratio_lt_0.25": r.curvature_ratio < 0.25,
                "low_plateau_width_positive": (
                    finite(r.low_plateau_width_exact_ge_center)
                    and r.low_plateau_width_exact_ge_center > 0
                ),
                "high_direction_drops_at_an_extreme": (
                    r.high_drop_at_plus_1 > 0 or r.high_drop_at_minus_1 > 0
                ),
            }
        )

    return {
        "n_seeds": len(rows),
        "seeds": [r.seed for r in rows],
        "curvature_ratio": {
            "mean": safe_mean(ratios),
            "median": safe_median(ratios),
            "std": safe_std(ratios),
            "min": min(ratios) if ratios else float("nan"),
            "max": max(ratios) if ratios else float("nan"),
        },
        "eig_low": {
            "mean": safe_mean(eig_low),
            "median": safe_median(eig_low),
            "std": safe_std(eig_low),
        },
        "eig_high": {
            "mean": safe_mean(eig_high),
            "median": safe_median(eig_high),
            "std": safe_std(eig_high),
        },
        "terrain_flags": terrain_flags,
        "terrain_consistency_fraction": (
            sum(
                1
                for f in terrain_flags
                if f["low_flatter_than_high"]
                and f["curvature_ratio_lt_0.25"]
                and f["high_direction_drops_at_an_extreme"]
            )
            / len(terrain_flags)
            if terrain_flags
            else float("nan")
        ),
        "reference_runs": REFERENCE_RUNS,
    }


def write_outputs(rows: list[SeedGeometry], points: list[SweepPoint]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with (OUT_ROOT / "seed_geometry_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    with (OUT_ROOT / "all_behavior_sweeps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for p in points:
            writer.writerow(asdict(p))

    summary = summarize(rows)
    write_json(OUT_ROOT / "cross_seed_summary.json", summary)

    lines = [
        "=" * 88,
        "E27 — OVERNIGHT L2 GEOMETRY REPLICATION SUMMARY",
        "=" * 88,
        f"Seeds completed: {summary['seeds']}",
        "",
        "CURVATURE",
        f"  low eigenvalue  mean   = {fmt(summary['eig_low']['mean'])}",
        f"  low eigenvalue  median = {fmt(summary['eig_low']['median'])}",
        f"  high eigenvalue mean   = {fmt(summary['eig_high']['mean'])}",
        f"  high eigenvalue median = {fmt(summary['eig_high']['median'])}",
        f"  low/high ratio mean     = {fmt(summary['curvature_ratio']['mean'])}",
        f"  low/high ratio median   = {fmt(summary['curvature_ratio']['median'])}",
        "",
        "PER-SEED",
    ]

    for r in rows:
        lines.append(
            f"  seed {r.seed:2d} | "
            f"ratio {r.curvature_ratio:.4f} | "
            f"low {r.eig_low:.5f} | high {r.eig_high:.5f} | "
            f"center {r.low_center_exact}/24 | "
            f"low-best {r.low_best_exact}/24 | "
            f"high-best {r.high_best_exact}/24 | "
            f"plateau-width {r.low_plateau_width_exact_ge_center:.3f}"
        )

    lines += [
        "",
        "STRUCTURAL REPLICATION CHECK",
        f"  fraction satisfying the simple terrain criteria = "
        f"{summary['terrain_consistency_fraction']:.3f}",
        "",
        "NOTE: This is a descriptive replication check, not a statistical test.",
        "Do not interpret 15/24 vs 16/24 as a large effect by itself.",
        "",
    ]

    (OUT_ROOT / "overnight_report.txt").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# SELF-TESTS
# ---------------------------------------------------------------------------

def run_self_tests() -> None:
    # These tests catch the exact class of bug that bit E26: silently running
    # only the first seed.
    seeds = list(range(22, 31))
    assert len(seeds) == 9
    assert seeds[0] == 22
    assert seeds[-1] == 30
    assert len(set(seeds)) == len(seeds)

    offsets = DEFAULT_OFFSETS
    assert 0.0 in offsets
    assert -1.0 in offsets
    assert 1.0 in offsets
    assert len(offsets) == 13

    # Geometry sanity checks from the known reference runs.
    for seed, ref in REFERENCE_RUNS.items():
        assert ref["low_curvature"] > 0
        assert ref["high_curvature"] > ref["low_curvature"]
        assert ref["low_high_ratio"] < 1.0

    # Ensure output directory name isn't accidentally inherited from E25/E26.
    assert "E25" not in str(OUT_ROOT)
    assert "E26" not in str(OUT_ROOT)

    print("SELF-TESTS: PASS")
    print(f"Fresh seeds scheduled: {seeds}")
    print(f"Offsets per direction: {len(offsets)}")
    print("This driver will refuse to run if the E25/E26 adapter is unavailable.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Fresh seeds to run. Default: 22..30.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run only local configuration/safety tests.",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    run_self_tests()

    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        raise SystemExit("No seeds supplied.")

    print("=" * 88)
    print("E27 — OVERNIGHT L2 GEOMETRY REPLICATION")
    print("=" * 88)
    print(f"Seeds: {seeds}")
    print(f"Count: {len(seeds)}")
    print(f"Offsets/direction: {len(DEFAULT_OFFSETS)}")
    print(f"Output: {OUT_ROOT.resolve()}")
    print()

    all_rows: list[SeedGeometry] = []
    all_points: list[SweepPoint] = []

    started = time.time()

    for i, seed in enumerate(seeds, 1):
        print("=" * 88)
        print(f"SEED {seed}  ({i}/{len(seeds)})")
        print("=" * 88)
        set_all_seeds(seed)

        try:
            row, points = run_one_seed(seed, DEFAULT_OFFSETS)
        except Exception as exc:
            # Never silently claim a failed seed completed.
            fail_dir = OUT_ROOT / f"seed_{seed:02d}"
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "FAILED.txt").write_text(
                f"Seed {seed} failed.\n\n{type(exc).__name__}: {exc}\n"
            )
            print(f"SEED {seed} FAILED: {type(exc).__name__}: {exc}")
            print("Continuing to the next seed.")
            continue

        all_rows.append(row)
        all_points.extend(points)

        seed_dir = OUT_ROOT / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_json(seed_dir / "geometry.json", asdict(row))

        print(
            f"seed {seed} | ratio={row.curvature_ratio:.4f} | "
            f"low={row.eig_low:.6f} | high={row.eig_high:.6f} | "
            f"center={row.low_center_exact}/24"
        )

        # Persist after EVERY seed so an interruption never destroys prior work.
        if all_rows:
            write_outputs(all_rows, all_points)

    elapsed = time.time() - started

    print()
    print("=" * 88)
    print("E27 COMPLETE")
    print("=" * 88)
    print(f"Completed seeds: {[r.seed for r in all_rows]}")
    print(f"Failed seeds: {[s for s in seeds if s not in [r.seed for r in all_rows]]}")
    print(f"Elapsed: {elapsed / 3600:.2f} hours")
    print(f"Summary: {OUT_ROOT / 'overnight_report.txt'}")
    print(f"CSV:     {OUT_ROOT / 'seed_geometry_summary.csv'}")
    print(f"All sweeps: {OUT_ROOT / 'all_behavior_sweeps.csv'}")


if __name__ == "__main__":
    main()
