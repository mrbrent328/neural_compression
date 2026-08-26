#!/usr/bin/env python3
import json
import csv
import re
from pathlib import Path

# Run this from /home/xetal/Desktop/latent_model_test, or change this path.
ROOT = Path("/home/xetal/Desktop/latent_model_test/E27_l2_geometry_replication_results/")

OUT_CSV = ROOT / "recovered_cross_seed_summary.csv"
OUT_JSON = ROOT / "recovered_cross_seed_summary.json"

rows = []

for seed_dir in sorted(ROOT.glob("seed_*")):
    if not seed_dir.is_dir():
        continue

    m = re.fullmatch(r"seed_(\d+)", seed_dir.name)
    if not m:
        continue

    seed = int(m.group(1))
    summary_path = seed_dir / "geometry_summary.json"

    if not summary_path.exists():
        rows.append({
            "seed": seed,
            "status": "MISSING geometry_summary.json",
        })
        continue

    try:
        data = json.loads(summary_path.read_text())

        row = {
            "seed": seed,
            "status": "complete",
            "center": json.dumps(data.get("center")),
            "gradient": json.dumps(data.get("gradient")),
            "eigenvalues": json.dumps(data.get("eigenvalues")),
            "eigenvectors": json.dumps(data.get("eigenvectors_columns")),
        }

        ev = data.get("eigenvalues")
        if isinstance(ev, list) and len(ev) >= 2 and float(ev[1]) != 0:
            row["low_high_ratio"] = float(ev[0]) / float(ev[1])
        else:
            row["low_high_ratio"] = None

        rows.append(row)

    except Exception as exc:
        rows.append({
            "seed": seed,
            "status": f"ERROR: {type(exc).__name__}: {exc}",
        })

# Write the recovered summary.
fieldnames = [
    "seed",
    "status",
    "center",
    "gradient",
    "eigenvalues",
    "eigenvectors",
    "low_high_ratio",
]

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

OUT_JSON.write_text(json.dumps(rows, indent=2))

print("=" * 80)
print("E27 RECOVERED CROSS-SEED SUMMARY")
print("=" * 80)
print(f"Seed folders found: {len(rows)}")
print()

for row in rows:
    print(
        f"seed {row['seed']:2d} | "
        f"{row['status']:30s} | "
        f"low/high = {row.get('low_high_ratio', 'n/a')}"
    )

print()
print(f"CSV : {OUT_CSV}")
print(f"JSON: {OUT_JSON}")
