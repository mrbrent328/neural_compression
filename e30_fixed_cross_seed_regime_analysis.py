#!/usr/bin/env python3
"""
E30 — Fixed cross-seed geometry/behavior regime analysis

This version is designed to consume the ACTUAL E27/E28 summary files rather
than assuming seed_22/seed_23/... directories.

It searches recursively for JSON files whose names contain:
    e27
    e28

and recursively extracts per-seed records from them.

It is intentionally conservative:
- it will NOT invent missing metrics
- it reports exactly which fields it found
- it can use several likely field names for curvature and behavior
- it writes an extraction audit so parser problems are visible

Run from the experiment project root:

    python3 e30_seed_geometry_regime_clustering_fixed.py

Outputs:
    e30_fixed_summary.json
    e30_fixed_table.csv
    e30_fixed_regime_scatter.png
    e30_fixed_feature_profiles.png
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(".")
SEEDS = list(range(22, 31))

OUT_JSON = ROOT / "e30_fixed_summary.json"
OUT_CSV = ROOT / "e30_fixed_table.csv"
OUT_SCATTER = ROOT / "e30_fixed_regime_scatter.png"
OUT_FEATURES = ROOT / "e30_fixed_feature_profiles.png"


def number(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def seed_from_obj(obj):
    if not isinstance(obj, dict):
        return None

    for key in ("seed", "seed_id", "random_seed"):
        value = number(obj.get(key))
        if value is not None and int(value) in SEEDS:
            return int(value)

    return None


def recursive_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from recursive_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from recursive_dicts(value)


def first_number(obj, names):
    wanted = {x.lower() for x in names}

    for d in recursive_dicts(obj):
        for key, value in d.items():
            if str(key).lower() in wanted:
                v = number(value)
                if v is not None:
                    return v, key

    return None, None


def load_json_files():
    files = []

    for path in ROOT.rglob("*.json"):
        name = path.name.lower()
        if "e27" in name or "e28" in name:
            files.append(path)

    return sorted(set(files))


def extract_records():
    files = load_json_files()

    records = {
        seed: {
            "seed": seed,
            "source_files": [],
            "found_fields": {},
        }
        for seed in SEEDS
    }

    # Candidate names intentionally include the terminology used in our
    # earlier E27/E28 reports.
    curvature_names = [
        "curvature_ratio",
        "eigenvalue_ratio",
        "lambda_ratio",
        "low_high_ratio",
        "anisotropy_ratio",
        "ridge_ratio",
        "ridge_strength",
    ]

    low_names = [
        "low_curvature",
        "low_eigenvalue",
        "lambda_low",
        "eig_low",
        "eigenvalue_low",
        "curvature_low",
        "lambda1",
    ]

    high_names = [
        "high_curvature",
        "high_eigenvalue",
        "lambda_high",
        "eig_high",
        "eigenvalue_high",
        "curvature_high",
        "lambda2",
    ]

    behavior_names = [
        "behavior_separation",
        "behaviour_separation",
        "behavior_ratio",
        "behaviour_ratio",
        "sensitivity_ratio",
        "ridge_behavior_ratio",
        "output_separation",
        "behavioral_separation",
    ]

    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
            continue

        # Find every dict that carries a recognizable seed.
        for obj in recursive_dicts(data):
            seed = seed_from_obj(obj)
            if seed is None:
                continue

            rec = records[seed]

            if str(path) not in rec["source_files"]:
                rec["source_files"].append(str(path))

            # Search this seed-containing object first.
            ratio, ratio_key = first_number(obj, curvature_names)
            low, low_key = first_number(obj, low_names)
            high, high_key = first_number(obj, high_names)
            behavior, behavior_key = first_number(obj, behavior_names)

            if ratio is not None:
                rec["curvature_ratio"] = ratio
                rec["found_fields"]["curvature_ratio"] = ratio_key

            if low is not None:
                rec["low_curvature"] = low
                rec["found_fields"]["low_curvature"] = low_key

            if high is not None:
                rec["high_curvature"] = high
                rec["found_fields"]["high_curvature"] = high_key

            if behavior is not None:
                rec["behavior_separation"] = behavior
                rec["found_fields"]["behavior_separation"] = behavior_key

    # Derive curvature ratio only if both eigen/curvature values exist.
    for rec in records.values():
        if rec.get("curvature_ratio") is None:
            low = rec.get("low_curvature")
            high = rec.get("high_curvature")

            if low is not None and high is not None and abs(high) > 1e-12:
                rec["curvature_ratio"] = low / high
                rec["found_fields"]["curvature_ratio"] = "derived(low/high)"

    return list(records.values()), files


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom < 1e-12:
        return 0.0

    return float((x * y).sum() / denom)


def spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))

    return pearson(rx, ry)


def standardize(x):
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std


def kmeans(X, k, iterations=100):
    # Deterministic initialization.
    indices = np.linspace(0, len(X) - 1, k, dtype=int)
    centers = X[indices].copy()

    labels = np.zeros(len(X), dtype=int)

    for _ in range(iterations):
        distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        for i in range(k):
            members = X[labels == i]
            if len(members):
                centers[i] = members.mean(axis=0)

    return labels, centers


def classify(records):
    usable = [
        r for r in records
        if r.get("curvature_ratio") is not None
        and r.get("behavior_separation") is not None
    ]

    if len(usable) < 3:
        return {
            "status": "insufficient_data",
            "usable_seeds": [r["seed"] for r in usable],
            "reason": (
                "Fewer than three seeds have both curvature_ratio and "
                "behavior_separation. The parser did not invent missing "
                "values."
            ),
        }

    raw = np.array([
        [
            r["curvature_ratio"],
            r["behavior_separation"],
        ]
        for r in usable
    ], dtype=float)

    X = standardize(raw)

    # With only 9 seeds, clustering is descriptive. Prefer 3 only when
    # every cluster gets at least two seeds.
    chosen = None

    for k in (3, 2):
        if len(usable) < k:
            continue

        labels, centers = kmeans(X, k)
        counts = np.bincount(labels, minlength=k)

        if k == 3 and np.all(counts >= 2):
            chosen = (k, labels, centers)
            break

        if k == 2 and chosen is None:
            chosen = (k, labels, centers)

    k, labels, centers = chosen

    for rec, label in zip(usable, labels):
        rec["cluster_id"] = int(label)

    clusters = []

    for cid in range(k):
        members = [r for r in usable if r["cluster_id"] == cid]

        ratio = np.mean([r["curvature_ratio"] for r in members])
        behavior = np.mean([r["behavior_separation"] for r in members])

        if behavior < 1.0:
            regime = "inverted_or_weak"
        elif ratio < 0.15 and behavior > 2.0:
            regime = "strong_ridge"
        elif ratio > 0.5:
            regime = "round_or_weak"
        else:
            regime = "intermediate"

        clusters.append({
            "cluster_id": cid,
            "regime": regime,
            "seeds": [r["seed"] for r in members],
            "mean_curvature_ratio": float(ratio),
            "mean_behavior_separation": float(behavior),
        })

        for r in members:
            r["regime"] = regime

    ratios = np.array([r["curvature_ratio"] for r in usable])
    behavior = np.array([r["behavior_separation"] for r in usable])

    return {
        "status": "ok",
        "usable_seeds": [r["seed"] for r in usable],
        "selected_k": k,
        "clusters": clusters,
        "pearson": pearson(ratios, behavior),
        "spearman": spearman(ratios, behavior),
        "warning": (
            "Nine seeds are enough to inspect the pattern but not enough "
            "to establish population-level cluster statistics."
        ),
    }


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

    plt.figure(figsize=(9, 7))

    for r in usable:
        plt.scatter(r["curvature_ratio"], r["behavior_separation"], s=90)
        plt.annotate(
            f"seed {r['seed']}",
            (r["curvature_ratio"], r["behavior_separation"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.axhline(1.0, linewidth=0.8)
    plt.axvline(0.15, linewidth=0.8)
    plt.xlabel("curvature ratio (low / high)")
    plt.ylabel("E28 behavior separation")
    plt.title("E30 — Seed geometry vs behavioral regime")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=160)
    plt.close()

    features = []
    labels = []

    for r in usable:
        features.append([
            r["curvature_ratio"],
            r["behavior_separation"],
            r.get("low_curvature", np.nan),
            r.get("high_curvature", np.nan),
        ])
        labels.append(r["seed"])

    A = np.asarray(features, dtype=float)
    A = standardize(A)

    plt.figure(figsize=(11, 6))

    names = [
        "curvature ratio",
        "behavior separation",
        "low curvature",
        "high curvature",
    ]

    for j, name in enumerate(names):
        plt.plot(labels, A[:, j], marker="o", label=name)

    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("seed")
    plt.ylabel("standardized value")
    plt.title("E30 — Cross-seed feature profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FEATURES, dpi=160)
    plt.close()


def main():
    print("=" * 90)
    print("E30 — FIXED CROSS-SEED GEOMETRY / BEHAVIOR ANALYSIS")
    print("=" * 90)
    print()

    records, files = extract_records()

    print("Found E27/E28 JSON files:")
    for path in files:
        print(f"  {path}")

    print()
    print("Extraction audit:")

    for r in records:
        fields = ", ".join(
            f"{k}={v}"
            for k, v in r["found_fields"].items()
        )

        print(
            f"  seed {r['seed']:2d} | "
            f"curvature={r.get('curvature_ratio')} | "
            f"behavior={r.get('behavior_separation')} | "
            f"{fields or 'NO MATCHED FIELDS'}"
        )

    print()

    classification = classify(records)

    print("Classification:")
    print(json.dumps(classification, indent=2))
    print()

    # Save a flat CSV for easy inspection.
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "curvature_ratio",
                "low_curvature",
                "high_curvature",
                "behavior_separation",
                "cluster_id",
                "regime",
                "source_files",
                "found_fields",
            ],
        )
        writer.writeheader()

        for r in records:
            writer.writerow({
                "seed": r["seed"],
                "curvature_ratio": r.get("curvature_ratio"),
                "low_curvature": r.get("low_curvature"),
                "high_curvature": r.get("high_curvature"),
                "behavior_separation": r.get("behavior_separation"),
                "cluster_id": r.get("cluster_id"),
                "regime": r.get("regime"),
                "source_files": "; ".join(r["source_files"]),
                "found_fields": json.dumps(r["found_fields"]),
            })

    report = {
        "experiment": "E30_fixed_cross_seed_regime_analysis",
        "question": (
            "Do the seed-to-seed geometry/behavior differences form "
            "recurring regimes?"
        ),
        "source_files": [str(p) for p in files],
        "records": records,
        "classification": classification,
        "guardrail": (
            "This is analysis of existing E27/E28 outputs. No model "
            "was retrained and missing values were not fabricated."
        ),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    make_plots(records)

    print("=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"JSON: {OUT_JSON}")
    print(f"CSV : {OUT_CSV}")
    print(f"PNG : {OUT_SCATTER}")
    print(f"PNG : {OUT_FEATURES}")


if __name__ == "__main__":
    main()
