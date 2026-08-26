#!/usr/bin/env python3
"""
E30 — Seed Geometry Regime Clustering

Question
--------
E28 showed that latent geometry/behavior agreement is NOT a single clean
relationship across seeds. Some seeds look like strong ridge systems, some
are weak/round, and some invert the expected behavioral relationship.

E30 asks:
    Are there reproducible geometric/behavioral REGIMES across seeds?

This is an analysis-only experiment. It does NOT retrain any model.

Expected directory layout (run from the experiment root):

    seed_22/
        geometry_summary.json
        behavior_probes.txt
        ...
    seed_23/
        ...
    ...
    seed_30/

If an E28 cross-seed summary is present, the script will use it when possible.
Otherwise it extracts the geometry and behavior information directly.

Outputs:
    e30_seed_regime_summary.json
    e30_seed_regime_table.csv
    e30_regime_scatter.png
    e30_regime_features.png

Dependencies:
    Python 3
    numpy
    matplotlib
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = list(range(22, 31))
ROOT = Path(".")
OUTPUT_JSON = ROOT / "e30_seed_regime_summary.json"
OUTPUT_CSV = ROOT / "e30_seed_regime_table.csv"
PLOT_SCATTER = ROOT / "e30_regime_scatter.png"
PLOT_FEATURES = ROOT / "e30_regime_features.png"


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom < 1e-12:
        return 0.0

    return float(np.sum(x * y) / denom)


def rank_corr(x, y):
    """Spearman correlation without scipy."""
    rx = np.argsort(np.argsort(np.asarray(x)))
    ry = np.argsort(np.argsort(np.asarray(y)))
    return pearson(rx, ry)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------

def find_number(obj, keys):
    """
    Recursively search a JSON object for the first numeric value associated
    with one of the requested keys.
    """
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                value = safe_float(obj[key])
                if value is not None:
                    return value

        for value in obj.values():
            found = find_number(value, keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = find_number(value, keys)
            if found is not None:
                return found

    return None


def load_geometry(seed_dir):
    path = seed_dir / "geometry_summary.json"

    if not path.exists():
        return None

    data = json.loads(path.read_text())

    low = find_number(
        data,
        [
            "low_curvature",
            "low_eigenvalue",
            "lambda_low",
            "eig_low",
            "eigenvalue_low",
            "curvature_low",
        ],
    )

    high = find_number(
        data,
        [
            "high_curvature",
            "high_eigenvalue",
            "lambda_high",
            "eig_high",
            "eigenvalue_high",
            "curvature_high",
        ],
    )

    ratio = find_number(
        data,
        [
            "curvature_ratio",
            "low_high_ratio",
            "eigenvalue_ratio",
            "ratio",
        ],
    )

    if ratio is None and low is not None and high is not None:
        if abs(high) > 1e-12:
            ratio = low / high

    return {
        "low_curvature": low,
        "high_curvature": high,
        "curvature_ratio": ratio,
        "raw": data,
    }


# ---------------------------------------------------------------------
# Behavior extraction
# ---------------------------------------------------------------------

def extract_behavior_from_text(seed_dir):
    """
    Best-effort parser for behavior_probes.txt.

    It accepts common formats such as:

        -0.500 13/24
        offset -0.500 score 13/24
        +0.250 -> 15/24

    The parser intentionally keeps the raw lines available if the format
    differs, rather than silently inventing values.
    """
    path_candidates = [
        seed_dir / "behavior_probes.txt",
        seed_dir / "behavior_probes(1).txt",
        seed_dir / "behavior_probes(2).txt",
    ]

    path = next((p for p in path_candidates if p.exists()), None)

    if path is None:
        return None

    rows = []

    # Matches an offset followed somewhere nearby by "score/total".
    pattern = re.compile(
        r"""
        (?P<offset>[+-]?\d+(?:\.\d+)?)
        .*?
        (?P<score>\d+)\s*/\s*(?P<total>\d+)
        """,
        re.VERBOSE,
    )

    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue

        offset = float(match.group("offset"))
        score = float(match.group("score"))
        total = float(match.group("total"))

        if total <= 0:
            continue

        rows.append((offset, score / total))

    if not rows:
        return None

    rows.sort(key=lambda x: x[0])

    offsets = np.array([x[0] for x in rows], dtype=float)
    scores = np.array([x[1] for x in rows], dtype=float)

    # Near-zero point.
    center_idx = int(np.argmin(np.abs(offsets)))
    center = scores[center_idx]

    # Compare behavior variation on each side of the local direction.
    # The important E28 quantity was essentially:
    #     behavioral sensitivity along steep / flat direction.
    #
    # Here we provide a robust range-based proxy.
    negative = scores[offsets < -1e-8]
    positive = scores[offsets > 1e-8]

    if len(negative) and len(positive):
        span = max(
            float(np.max(negative) - np.min(negative)),
            float(np.max(positive) - np.min(positive)),
        )
    else:
        span = float(np.max(scores) - np.min(scores))

    return {
        "n_points": int(len(scores)),
        "center_score": float(center),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_span": span,
        "offsets": offsets.tolist(),
        "scores": scores.tolist(),
    }


# ---------------------------------------------------------------------
# E28 summary fallback / enrichment
# ---------------------------------------------------------------------

def find_e28_summary():
    candidates = [
        ROOT / "e28_cross_seed_summary.json",
        ROOT / "e28_cross_seed_summary" / "e28_cross_seed_summary.json",
        ROOT / "E28" / "e28_cross_seed_summary.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Search a little deeper without depending on a fixed project name.
    for path in ROOT.glob("**/e28_cross_seed_summary.json"):
        return path

    return None


def load_e28_summary():
    path = find_e28_summary()

    if path is None:
        return None

    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def flatten_seed_records(obj):
    """
    Find dicts that look like per-seed E28 records.
    """
    records = []

    def visit(node):
        if isinstance(node, dict):
            seed = node.get("seed")

            if isinstance(seed, (int, float, str)):
                try:
                    seed_int = int(seed)
                except Exception:
                    seed_int = None

                if seed_int in SEEDS:
                    records.append((seed_int, node))

            for value in node.values():
                visit(value)

        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(obj)
    return records


# ---------------------------------------------------------------------
# Regime construction
# ---------------------------------------------------------------------

def build_records():
    e28 = load_e28_summary()
    e28_by_seed = {}

    if e28 is not None:
        for seed, record in flatten_seed_records(e28):
            e28_by_seed.setdefault(seed, record)

    records = []

    for seed in SEEDS:
        seed_dir = ROOT / f"seed_{seed}"

        geometry = load_geometry(seed_dir)
        behavior = extract_behavior_from_text(seed_dir)

        e28_record = e28_by_seed.get(seed, {})

        # Prefer explicit E28 values when present.
        curvature_ratio = find_number(
            e28_record,
            [
                "curvature_ratio",
                "low_high_ratio",
                "eigenvalue_ratio",
            ],
        )

        behavior_separation = find_number(
            e28_record,
            [
                "behavior_separation",
                "sensitivity_ratio",
                "separation",
                "behavior_ratio",
            ],
        )

        low = geometry["low_curvature"] if geometry else None
        high = geometry["high_curvature"] if geometry else None

        if curvature_ratio is None and geometry:
            curvature_ratio = geometry["curvature_ratio"]

        # If E28's behavior separation is unavailable, use the raw probe
        # span as a descriptive fallback. Do NOT pretend it is the same
        # metric; label it separately.
        fallback_behavior_span = (
            behavior["score_span"] if behavior else None
        )

        records.append(
            {
                "seed": seed,
                "geometry_found": geometry is not None,
                "behavior_found": behavior is not None,
                "low_curvature": low,
                "high_curvature": high,
                "curvature_ratio": curvature_ratio,
                "behavior_separation": behavior_separation,
                "behavior_span_fallback": fallback_behavior_span,
                "e28_record_found": seed in e28_by_seed,
            }
        )

    return records


# ---------------------------------------------------------------------
# Simple deterministic regime clustering
# ---------------------------------------------------------------------

def standardize_matrix(X):
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)

    std[std < 1e-12] = 1.0

    return (X - mean) / std


def kmeans(X, k, iterations=100):
    """
    Tiny dependency-free k-means implementation.

    Initialization is deterministic and uses evenly spaced observations.
    """
    n = len(X)

    if n < k:
        raise ValueError("Not enough records for requested k.")

    initial = np.linspace(0, n - 1, k, dtype=int)
    centers = X[initial].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(iterations):
        distances = (
            (X[:, None, :] - centers[None, :, :]) ** 2
        ).sum(axis=2)

        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        for j in range(k):
            members = X[labels == j]

            if len(members):
                centers[j] = members.mean(axis=0)

    return labels, centers


def classify_regimes(records):
    """
    Primary clustering uses the two quantities that motivated E30:

        curvature ratio
        behavior separation

    We additionally retain signs so inverted regimes remain visible.
    """
    usable = [
        r for r in records
        if r["curvature_ratio"] is not None
        and r["behavior_separation"] is not None
    ]

    if len(usable) < 3:
        return records, {
            "status": "insufficient_data",
            "reason": (
                "Need at least three seeds with both curvature ratio "
                "and E28 behavior separation."
            ),
        }

    raw = np.array(
        [
            [
                float(r["curvature_ratio"]),
                float(r["behavior_separation"]),
            ]
            for r in usable
        ]
    )

    X = standardize_matrix(raw)

    # Try k=2 and k=3. We prefer k=3 if it produces three populated
    # regimes, because the hypothesis explicitly predicts:
    # strong / weak / inverted-or-weird.
    candidates = []

    for k in (2, 3):
        if len(usable) >= k:
            labels, centers = kmeans(X, k)

            counts = np.bincount(labels, minlength=k)
            within = 0.0

            for i in range(k):
                members = X[labels == i]
                if len(members):
                    within += float(
                        ((members - centers[i]) ** 2).sum()
                    )

            candidates.append(
                {
                    "k": k,
                    "labels": labels,
                    "centers": centers,
                    "counts": counts,
                    "within_ss": within,
                }
            )

    # With only 9 seeds, avoid pretending model selection is definitive.
    # Prefer k=3 when every cluster has >=2 members; otherwise k=2.
    selected = None

    for candidate in candidates:
        if candidate["k"] == 3 and np.all(candidate["counts"] >= 2):
            selected = candidate
            break

    if selected is None:
        selected = min(
            candidates,
            key=lambda c: c["within_ss"],
        )

    label_map = {}

    # Name clusters by their mean behavior/geometry, not arbitrary k-means ID.
    cluster_stats = []

    for cluster_id in range(selected["k"]):
        members = [
            usable[i]
            for i, label in enumerate(selected["labels"])
            if label == cluster_id
        ]

        ratios = np.array(
            [m["curvature_ratio"] for m in members],
            dtype=float,
        )
        separations = np.array(
            [m["behavior_separation"] for m in members],
            dtype=float,
        )

        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "seeds": [m["seed"] for m in members],
                "mean_curvature_ratio": float(ratios.mean()),
                "mean_behavior_separation": float(separations.mean()),
            }
        )

    # Descriptive naming:
    # "strong" = smaller curvature ratio + larger behavior separation
    # "weak" = larger ratio + behavior near 1
    # "inverted" = behavior separation < 1
    for stat in cluster_stats:
        ratio = stat["mean_curvature_ratio"]
        separation = stat["mean_behavior_separation"]

        if separation < 1.0:
            name = "inverted_or_weak"
        elif ratio < 0.15 and separation > 2.0:
            name = "strong_ridge"
        elif ratio > 0.5:
            name = "round_or_weak"
        else:
            name = "intermediate"

        label_map[stat["cluster_id"]] = name
        stat["regime"] = name

    for i, record in enumerate(usable):
        cluster_id = int(selected["labels"][i])
        record["cluster_id"] = cluster_id
        record["regime"] = label_map[cluster_id]

    return records, {
        "status": "ok",
        "selected_k": int(selected["k"]),
        "clusters": cluster_stats,
        "warning": (
            "With nine seeds, clustering is descriptive rather than "
            "statistically definitive."
        ),
    }


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def make_plots(records):
    usable = [
        r for r in records
        if r.get("curvature_ratio") is not None
        and r.get("behavior_separation") is not None
    ]

    if not usable:
        return

    x = np.array([r["curvature_ratio"] for r in usable])
    y = np.array([r["behavior_separation"] for r in usable])
    seeds = [r["seed"] for r in usable]

    plt.figure(figsize=(9, 7))

    for r in usable:
        label = r.get("regime", "unclassified")
        plt.scatter(
            r["curvature_ratio"],
            r["behavior_separation"],
            s=80,
        )
        plt.annotate(
            f"{r['seed']} ({label})",
            (r["curvature_ratio"], r["behavior_separation"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.axhline(1.0, linewidth=0.8)
    plt.axvline(0.15, linewidth=0.8)

    plt.xlabel("curvature ratio (low / high)")
    plt.ylabel("behavior separation")
    plt.title("E30 — Seed geometry / behavior regimes")
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER, dpi=160)
    plt.close()

    # Feature table visualization.
    features = np.column_stack(
        [
            x,
            y,
            np.array(
                [
                    r["low_curvature"]
                    if r["low_curvature"] is not None
                    else np.nan
                    for r in usable
                ]
            ),
            np.array(
                [
                    r["high_curvature"]
                    if r["high_curvature"] is not None
                    else np.nan
                    for r in usable
                ]
            ),
        ]
    )

    # Normalize each column independently for visual comparison.
    means = np.nanmean(features, axis=0)
    stds = np.nanstd(features, axis=0)
    stds[stds < 1e-12] = 1.0
    normalized = (features - means) / stds

    plt.figure(figsize=(11, 6))
    for j, name in enumerate(
        [
            "curvature ratio",
            "behavior separation",
            "low curvature",
            "high curvature",
        ]
    ):
        plt.plot(
            seeds,
            normalized[:, j],
            marker="o",
            label=name,
        )

    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("seed")
    plt.ylabel("standardized value")
    plt.title("E30 — Geometry/behavior feature profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FEATURES, dpi=160)
    plt.close()


def main():
    print("=" * 88)
    print("E30 — SEED GEOMETRY REGIME CLUSTERING")
    print("=" * 88)
    print()

    records = build_records()

    print("Seed inventory:")
    for r in records:
        print(
            f"  seed {r['seed']:2d} | "
            f"geometry={'yes' if r['geometry_found'] else 'NO ':3s} | "
            f"behavior={'yes' if r['behavior_found'] else 'NO ':3s} | "
            f"E28={'yes' if r['e28_record_found'] else 'no '}"
        )

    print()

    records, clustering = classify_regimes(records)

    usable = [
        r for r in records
        if r.get("curvature_ratio") is not None
        and r.get("behavior_separation") is not None
    ]

    print("Per-seed results:")
    for r in usable:
        print(
            f"  seed {r['seed']:2d} | "
            f"ratio={r['curvature_ratio']:+.4f} | "
            f"behavior={r['behavior_separation']:+.4f} | "
            f"regime={r.get('regime', 'unclassified')}"
        )

    print()

    if len(usable) >= 3:
        ratios = np.array(
            [r["curvature_ratio"] for r in usable]
        )
        separations = np.array(
            [r["behavior_separation"] for r in usable]
        )

        print("Cross-seed relationship:")
        print(
            f"  Pearson  = {pearson(ratios, separations):+.4f}"
        )
        print(
            f"  Spearman = {rank_corr(ratios, separations):+.4f}"
        )
        print()

    print("Clustering:")
    print(json.dumps(clustering, indent=2))
    print()

    report = {
        "experiment": "E30_seed_geometry_regime_clustering",
        "question": (
            "Are the different E28 geometry/behavior outcomes "
            "separable into recurring seed-level regimes?"
        ),
        "seeds": SEEDS,
        "records": records,
        "clustering": clustering,
        "cross_seed_relationship": (
            {
                "pearson": pearson(
                    np.array(
                        [
                            r["curvature_ratio"]
                            for r in usable
                        ]
                    ),
                    np.array(
                        [
                            r["behavior_separation"]
                            for r in usable
                        ]
                    ),
                ),
                "spearman": rank_corr(
                    np.array(
                        [
                            r["curvature_ratio"]
                            for r in usable
                        ]
                    ),
                    np.array(
                        [
                            r["behavior_separation"]
                            for r in usable
                        ]
                    ),
                ),
            }
            if len(usable) >= 3
            else None
        ),
        "interpretation_guardrail": (
            "This experiment is descriptive. With nine seeds, a cluster "
            "is evidence of a possible regime, not proof of a population "
            "level phenomenon."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2)
    )

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "low_curvature",
                "high_curvature",
                "curvature_ratio",
                "behavior_separation",
                "behavior_span_fallback",
                "cluster_id",
                "regime",
            ],
        )
        writer.writeheader()

        for r in records:
            writer.writerow(
                {
                    key: r.get(key)
                    for key in writer.fieldnames
                }
            )

    make_plots(records)

    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"Report : {OUTPUT_JSON}")
    print(f"CSV    : {OUTPUT_CSV}")
    print(f"Plot   : {PLOT_SCATTER}")
    print(f"Plot   : {PLOT_FEATURES}")


if __name__ == "__main__":
    main()
