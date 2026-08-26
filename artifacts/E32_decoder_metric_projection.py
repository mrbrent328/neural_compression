#!/usr/bin/env python3
"""
E32 — Decoder-Metric Projection of V5 Latent Response Vectors

Goal
----
E31 showed that the decoder induces a strongly non-uniform local metric
G = J^T J over the 2-D latent space.

V5 gives local latent response vectors for the known factors:
    A, B, C, interaction

E32 asks:

    Does measuring those response vectors with the decoder's metric
    change our interpretation of their size/direction?

For each probe and factor response vector v:

    Euclidean magnitude = ||v||
    decoder-effective magnitude = sqrt(v^T G v)

The second quantity is the first-order output-space movement produced
by that latent response.

We also compute:
    - decoder gain = metric_mag / euclidean_mag
    - alignment with decoder principal directions
    - correlations between Euclidean and decoder-weighted magnitudes
    - top-10% overlap between the two rankings

This uses the REAL V1 checkpoint and the existing V5 CSV.
It does not invent a new probe distribution.
"""

from __future__ import annotations

import csv
import json
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR
V1_PATH = ROOT.parent / "known_factor_synthetic_latent_v1.py"
V5_CSV = ROOT / "synthetic_latent_v5" / "probe_vector_geometry.csv"
MODEL_PATH = ROOT / "best_model.pt"
OUT = ROOT / "synthetic_latent_e32_decoder_metric"
DELTA = 0.01

FACTORS = ["A", "B", "C", "interaction"]
VECTOR_COLUMNS = {
    "A": ("dA_z0", "dA_z1"),
    "B": ("dB_z0", "dB_z1"),
    "C": ("dC_z0", "dC_z1"),
    "interaction": ("dinteraction_z0", "dinteraction_z1"),
}


def load_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model():
    v1 = load_module_from_path("real_latent_v1_e32", V1_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
    else:
        raise RuntimeError("best_model.pt is not a dictionary checkpoint.")

    # Real V1 architecture used by the checkpoint.
    model = v1.TinyAutoencoder(v1.ROWS * v1.COLS, 2)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def load_v5_rows():
    if not V5_CSV.exists():
        raise FileNotFoundError(f"Missing V5 CSV: {V5_CSV}")

    required = [
        "probe",
        "A", "B", "C", "interaction",
        "dA_z0", "dA_z1",
        "dB_z0", "dB_z1",
        "dC_z0", "dC_z1",
        "dinteraction_z0", "dinteraction_z1",
    ]

    rows = []
    with V5_CSV.open(newline="") as f:
        reader = csv.DictReader(f)

        missing = [x for x in required if x not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"V5 CSV is missing required columns: {missing}"
            )

        for raw in reader:
            row = {k: raw[k] for k in required}
            row["probe"] = int(row["probe"])

            for k in required:
                if k != "probe":
                    row[k] = float(row[k])

            rows.append(row)

    return rows


@torch.no_grad()
def decoder_jacobian(model, z):
    """
    Finite-difference decoder Jacobian at one latent point.

    J shape = [4096, 2].
    """
    cols = []

    for axis in range(2):
        zp = z.clone()
        zm = z.clone()

        zp[:, axis] += DELTA
        zm[:, axis] -= DELTA

        yp = model.decode(zp)
        ym = model.decode(zm)

        dz = (yp - ym) / (2.0 * DELTA)
        cols.append(dz.reshape(-1).cpu().numpy())

    return np.stack(cols, axis=1)


def metric_from_jacobian(J):
    G = J.T @ J

    eigvals, eigvecs = np.linalg.eigh(G)
    # columns of eigvecs correspond to ascending eigenvalues
    low = eigvecs[:, 0]
    high = eigvecs[:, 1]

    return G, eigvals, low, high


def metric_mag(v, G):
    value = float(v @ G @ v)
    return float(np.sqrt(max(value, 0.0)))


def metric_cosine(a, b, G):
    aa = float(a @ G @ a)
    bb = float(b @ G @ b)

    if aa <= 1e-20 or bb <= 1e-20:
        return np.nan

    ab = float(a @ G @ b)
    return float(
        np.clip(
            ab / np.sqrt(aa * bb),
            -1.0,
            1.0,
        )
    )


def angle_deg_from_cos(c):
    if not np.isfinite(c):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def analyze():
    OUT.mkdir(exist_ok=True)

    print("=" * 90)
    print("E32 — DECODER-METRIC PROJECTION OF V5 LATENT RESPONSE VECTORS")
    print("=" * 90)
    print()
    print(f"V1:       {V1_PATH}")
    print(f"Model:    {MODEL_PATH}")
    print(f"V5 CSV:   {V5_CSV}")
    print(f"Delta:    {DELTA}")
    print()

    model = load_model()
    rows = load_v5_rows()

    print(f"Loaded {len(rows)} V5 probes.")
    print()

    results = []

    for n, row in enumerate(rows):
        z = torch.tensor(
            [[row.get("z0", 0.0), row.get("z1", 0.0)]],
            dtype=torch.float32,
        )

        # V5 CSV does not need to expose z because E32 can recover it
        # from the factor coordinates using the real model + V5 generator.
        # Therefore z0/z1 must be reconstructed below if absent.
        #
        # We do this once outside the loop if needed.
        if n == 0:
            break

    # ------------------------------------------------------------------
    # Recover the exact latent coordinates used by V5 from its known
    # factor probes. We deliberately reuse V5's own generator rather than
    # treating the four scalar factor columns as the model input.
    # ------------------------------------------------------------------
    v5 = load_module_from_path(
        "real_latent_v5_e32",
        ROOT / "known_factor_synthetic_latent_v5_vector_field.py",
    )

    if not hasattr(v5, "generate_exact_v1_dataset"):
        raise RuntimeError("V5 does not expose generate_exact_v1_dataset().")
    if not hasattr(v5, "generate_from_factors"):
        raise RuntimeError("V5 does not expose generate_from_factors().")
    if not hasattr(v5, "encode_factors"):
        raise RuntimeError("V5 does not expose encode_factors().")

    _, factors, patterns, norm_mean, norm_std = (
        v5.generate_exact_v1_dataset()
    )

    # V5 uses the held-out 20% exactly as its vector-field experiment.
    split = int(len(factors) * 0.8)
    heldout = factors[split:]

    if len(heldout) != len(rows):
        raise RuntimeError(
            "Probe count mismatch:\n"
            f"  V5 CSV probes: {len(rows)}\n"
            f"  V5 held-out factors: {len(heldout)}\n"
            "Refusing to silently misalign probes."
        )

    # Encode the exact held-out probes through V5's machinery.
    device = getattr(v5, "device", torch.device("cpu"))
    try:
        z_tensor = v5.encode_factors(
            model,
            device,
            heldout,
            patterns,
            norm_mean,
            norm_std,
        )
    except Exception:
        # V5 may expect model/device tensors in a slightly different form.
        z_tensor = v5.encode_factors(
            model,
            torch.device("cpu"),
            heldout,
            patterns,
            norm_mean,
            norm_std,
        )

    z_np = np.asarray(z_tensor)

    if z_np.ndim != 2 or z_np.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected latent shape from V5: {z_np.shape}"
        )

    print("Recovered exact V5 latent coordinates.")
    print(f"Latent shape: {z_np.shape}")
    print()

    for i, row in enumerate(rows):
        z = torch.tensor(
            z_np[i:i + 1],
            dtype=torch.float32,
        )

        J = decoder_jacobian(model, z)
        G, eigvals, low, high = metric_from_jacobian(J)

        record = {
            "probe": row["probe"],
            "z0": float(z_np[i, 0]),
            "z1": float(z_np[i, 1]),
            "eig_low": float(eigvals[0]),
            "eig_high": float(eigvals[1]),
            "anisotropy": float(
                eigvals[1] / max(eigvals[0], 1e-12)
            ),
        }

        for factor in FACTORS:
            c0, c1 = VECTOR_COLUMNS[factor]
            v = np.array(
                [row[c0], row[c1]],
                dtype=float,
            )

            euc = float(np.linalg.norm(v))
            dec = metric_mag(v, G)
            gain = dec / max(euc, 1e-12)

            cos_high = float(
                np.clip(
                    np.dot(v, high) /
                    max(np.linalg.norm(v), 1e-12),
                    -1.0,
                    1.0,
                )
            )
            cos_low = float(
                np.clip(
                    np.dot(v, low) /
                    max(np.linalg.norm(v), 1e-12),
                    -1.0,
                    1.0,
                )
            )

            record[f"{factor}_euclidean_mag"] = euc
            record[f"{factor}_decoder_mag"] = dec
            record[f"{factor}_decoder_gain"] = gain
            record[f"{factor}_angle_high_deg"] = angle_deg_from_cos(
                cos_high
            )
            record[f"{factor}_angle_low_deg"] = angle_deg_from_cos(
                cos_low
            )

        # Pairwise decoder-metric angles.
        vectors = {}
        for factor in FACTORS:
            c0, c1 = VECTOR_COLUMNS[factor]
            vectors[factor] = np.array(
                [row[c0], row[c1]],
                dtype=float,
            )

        for a_idx in range(len(FACTORS)):
            for b_idx in range(a_idx + 1, len(FACTORS)):
                a = FACTORS[a_idx]
                b = FACTORS[b_idx]
                c = metric_cosine(
                    vectors[a],
                    vectors[b],
                    G,
                )
                record[f"{a}_vs_{b}_decoder_angle_deg"] = (
                    angle_deg_from_cos(c)
                )

        results.append(record)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = OUT / "e32_decoder_metric.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {
        "n_probes": len(results),
        "delta": DELTA,
        "factors": {},
        "global": {},
    }

    for factor in FACTORS:
        e = np.array(
            [r[f"{factor}_euclidean_mag"] for r in results]
        )
        d = np.array(
            [r[f"{factor}_decoder_mag"] for r in results]
        )
        g = np.array(
            [r[f"{factor}_decoder_gain"] for r in results]
        )

        top_e = set(
            np.argsort(e)[-max(1, len(e) // 10):]
        )
        top_d = set(
            np.argsort(d)[-max(1, len(d) // 10):]
        )

        summary["factors"][factor] = {
            "euclidean_magnitude": {
                "median": float(np.median(e)),
                "mean": float(np.mean(e)),
                "p95": float(np.percentile(e, 95)),
                "max": float(np.max(e)),
            },
            "decoder_metric_magnitude": {
                "median": float(np.median(d)),
                "mean": float(np.mean(d)),
                "p95": float(np.percentile(d, 95)),
                "max": float(np.max(d)),
            },
            "decoder_gain": {
                "median": float(np.median(g)),
                "mean": float(np.mean(g)),
                "p95": float(np.percentile(g, 95)),
                "max": float(np.max(g)),
            },
            "euclidean_vs_decoder_rank_correlation": safe_corr(e, d),
            "top10_overlap_fraction": float(
                len(top_e & top_d) / max(len(top_e), 1)
            ),
        }

    anis = np.array([r["anisotropy"] for r in results])

    summary["global"] = {
        "decoder_anisotropy": {
            "median": float(np.median(anis)),
            "mean": float(np.mean(anis)),
            "p95": float(np.percentile(anis, 95)),
            "max": float(np.max(anis)),
        }
    }

    # Correlate local decoder anisotropy with how much decoder weighting
    # changes each factor's apparent magnitude.
    for factor in FACTORS:
        gain = np.array(
            [r[f"{factor}_decoder_gain"] for r in results]
        )
        summary["global"][
            f"anisotropy_vs_{factor}_gain_corr"
        ] = safe_corr(anis, gain)

    json_path = OUT / "e32_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # 1. Euclidean vs decoder metric magnitude.
    for factor in FACTORS:
        e = np.array(
            [r[f"{factor}_euclidean_mag"] for r in results]
        )
        d = np.array(
            [r[f"{factor}_decoder_mag"] for r in results]
        )

        plt.figure(figsize=(8, 7))
        plt.scatter(e, d, s=16, alpha=0.65)
        lo = min(np.min(e), np.min(d))
        hi = max(np.max(e), np.max(d))
        plt.plot([lo, hi], [lo, hi], linewidth=1)
        plt.xlabel("Euclidean latent response magnitude")
        plt.ylabel("Decoder-metric response magnitude")
        plt.title(f"E32 — {factor}: latent vs decoder geometry")
        plt.tight_layout()
        plt.savefig(
            OUT / f"e32_{factor}_euclidean_vs_decoder.png",
            dpi=160,
        )
        plt.close()

    # 2. Decoder gain over latent space.
    for factor in FACTORS:
        gain = np.array(
            [r[f"{factor}_decoder_gain"] for r in results]
        )
        z0 = np.array([r["z0"] for r in results])
        z1 = np.array([r["z1"] for r in results])

        plt.figure(figsize=(8, 7))
        sc = plt.scatter(z0, z1, c=gain, s=18)
        plt.xlabel("latent z0")
        plt.ylabel("latent z1")
        plt.title(f"E32 — decoder gain for {factor}")
        plt.colorbar(sc, label="decoder metric / Euclidean")
        plt.tight_layout()
        plt.savefig(
            OUT / f"e32_{factor}_decoder_gain_map.png",
            dpi=160,
        )
        plt.close()

    # 3. Global anisotropy vs decoder gain.
    plt.figure(figsize=(8, 7))
    for factor in FACTORS:
        gain = np.array(
            [r[f"{factor}_decoder_gain"] for r in results]
        )
        plt.scatter(
            anis,
            gain,
            s=14,
            alpha=0.45,
            label=factor,
        )
    plt.xscale("log")
    plt.xlabel("decoder anisotropy")
    plt.ylabel("decoder gain")
    plt.title("E32 — decoder anisotropy vs response gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT / "e32_anisotropy_vs_gain.png",
        dpi=160,
    )
    plt.close()

    # 4. Decoder-aware pairwise angles.
    pair_names = []
    pair_arrays = []
    for a_idx in range(len(FACTORS)):
        for b_idx in range(a_idx + 1, len(FACTORS)):
            a = FACTORS[a_idx]
            b = FACTORS[b_idx]
            key = f"{a}_vs_{b}_decoder_angle_deg"
            pair_names.append(key)
            pair_arrays.append(
                np.array([r[key] for r in results])
            )

    plt.figure(figsize=(11, 6))
    plt.boxplot(
        [x[np.isfinite(x)] for x in pair_arrays],
        tick_labels=pair_names,
    )
    plt.ylabel("decoder-metric angle (degrees)")
    plt.title("E32 — factor response angles under decoder geometry")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUT / "e32_decoder_pairwise_angles.png",
        dpi=160,
    )
    plt.close()

    print("=" * 90)
    print("E32 COMPLETE")
    print("=" * 90)
    print()
    print(json.dumps(summary, indent=2))
    print()
    print(f"CSV:     {csv_path}")
    print(f"SUMMARY: {json_path}")
    print(f"OUTPUT:  {OUT}")


if __name__ == "__main__":
    analyze()
