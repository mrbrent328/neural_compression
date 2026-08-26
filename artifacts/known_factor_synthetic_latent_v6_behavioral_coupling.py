#!/usr/bin/env python3
"""
V6 — Behavioral Coupling of Latent Geometry

Goal
----
Take the trained small synthetic autoencoder used by V5 and measure, at the
same probe points:

    input intervention
        -> latent movement
        -> reconstruction/output movement

This is the first experiment that directly tests whether the striking V5
latent geometry is behaviorally meaningful.

For each probe and factor, V6 computes central finite differences:

    dZ/dx  = (Z(x + delta) - Z(x - delta)) / (2 delta)

and

    dY/dx  = (Y(x + delta) - Y(x - delta)) / (2 delta)

where Y is the reconstructed input.

Then it reports:

    latent magnitude
    output magnitude
    output/latent gain
    reconstruction error
    output directional agreement with the expected input perturbation

It also compares the top-10% latent-sensitivity probes against the rest.

No training is performed.

Expected usage
--------------
Run from the directory containing the V5 model/checkpoint and dataset, e.g.

    cd /home/xetal/Desktop/latent_model_test/synthetic_latent_v1

    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
      python3 known_factor_synthetic_latent_v6_behavioral_coupling.py

The script searches nearby directories for:
    best_model.pt
    dataset.pt

and can also reuse V5's probe CSV if found.

Outputs:
    synthetic_latent_v6/
        v6_behavioral_coupling.csv
        v6_behavioral_coupling_results.json
        v6_latent_vs_output.png
        v6_top10_vs_rest.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(".")
OUT_DIR = ROOT / "synthetic_latent_v6"
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = ["best_model.pt", "model.pt", "tiny_autoencoder.pt"]
DATA_NAMES = ["dataset.pt", "data.pt"]

DELTA = 0.01
TOP_PERCENT = 0.10

# Factor order follows the synthetic dataset used in the V5 experiments.
FACTORS = ["A", "B", "C", "interaction"]


# ---------------------------------------------------------------------
# Model discovery / loading
# ---------------------------------------------------------------------

def find_file(names):
    # Prefer files close to the current experiment.
    for name in names:
        direct = ROOT / name
        if direct.exists():
            return direct

    # Then search a few levels down.
    for depth in range(1, 5):
        pattern = "/".join(["*"] * depth + [""])
        # Simpler recursive search, with shallow filtering.
        for name in names:
            matches = list(ROOT.rglob(name))
            if matches:
                # Prefer a path containing synthetic_latent_v1 or v5.
                matches.sort(
                    key=lambda p: (
                        "synthetic_latent_v5" not in str(p),
                        len(p.parts),
                    )
                )
                return matches[0]

    return None


def load_model(model_path):
    """
    Supports either:
      - a full nn.Module checkpoint
      - a state_dict checkpoint
      - a checkpoint dict containing model_state_dict/state_dict
    """
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        return checkpoint

    if isinstance(checkpoint, dict):
        state = None

        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break

        if state is not None:
            # Reconstruct the known tiny autoencoder architecture locally.
            model = TinyAutoencoder()
            model.load_state_dict(state)
            return model

        # Some checkpoints are state_dicts directly.
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            model = TinyAutoencoder()
            model.load_state_dict(checkpoint)
            return model

    raise RuntimeError(
        f"Could not interpret checkpoint: {model_path}"
    )


# ---------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------

class TinyAutoencoder(nn.Module):
    """
    Compatibility implementation of the small synthetic autoencoder used
    in the V5 experiment.

    If the checkpoint architecture differs, loading will fail loudly rather
    than silently analyzing the wrong model.
    """

    def __init__(self, input_dim=4, latent_dim=2, hidden_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y, z


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------

def load_dataset(data_path):
    obj = torch.load(data_path, map_location="cpu")

    # Common formats.
    if isinstance(obj, torch.Tensor):
        X = obj.float()
        return X

    if isinstance(obj, dict):
        for key in (
            "data",
            "X",
            "inputs",
            "samples",
            "dataset",
        ):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].float()

    if isinstance(obj, (tuple, list)):
        for item in obj:
            if isinstance(item, torch.Tensor):
                return item.float()

    raise RuntimeError(
        f"Could not find a tensor dataset inside {data_path}"
    )


def normalize_dataset(X):
    """
    The synthetic V5 probes are represented by four normalized factor
    channels. We preserve the dataset as-is unless it clearly has another
    shape.
    """
    if X.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D dataset tensor, got shape {tuple(X.shape)}"
        )

    if X.shape[1] < 4:
        raise RuntimeError(
            f"Expected at least 4 factor columns, got {X.shape}"
        )

    return X[:, :4].contiguous()


# ---------------------------------------------------------------------
# Synthetic factor reconstruction
# ---------------------------------------------------------------------

def make_probe(A, B, C, interaction):
    """
    Construct the known-factor representation used in the synthetic
    experiment.

    The four channels are:
        A
        B
        C
        interaction

    V5's CSV confirms these are the probe coordinates exposed to the
    geometry analysis.
    """
    return torch.tensor(
        [[float(A), float(B), float(C), float(interaction)]],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------
# Probe generation
# ---------------------------------------------------------------------

def load_v5_probes():
    candidates = [
        ROOT / "synthetic_latent_v5" / "probe_vector_geometry.csv",
        ROOT / "probe_vector_geometry.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = list(ROOT.rglob("probe_vector_geometry.csv"))
    if matches:
        return matches[0]

    return None


def read_v5_probes(path):
    import csv

    rows = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = ["A", "B", "C", "interaction"]
        missing = [x for x in required if x not in reader.fieldnames]

        if missing:
            raise RuntimeError(
                f"V5 CSV is missing required columns: {missing}"
            )

        for row in reader:
            rows.append({
                "probe": int(row["probe"]),
                "A": float(row["A"]),
                "B": float(row["B"]),
                "C": float(row["C"]),
                "interaction": float(row["interaction"]),
            })

    return rows


def generate_probes_from_dataset(X, n=2000, seed=1234):
    rng = np.random.default_rng(seed)

    n = min(n, len(X))

    indices = rng.choice(len(X), size=n, replace=False)

    rows = []

    for i, idx in enumerate(indices):
        values = X[idx].numpy()

        rows.append({
            "probe": i,
            "A": float(values[0]),
            "B": float(values[1]),
            "C": float(values[2]),
            "interaction": float(values[3]),
        })

    return rows


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, x):
    y, z = model(x)
    return y.detach().cpu().numpy()[0], z.detach().cpu().numpy()[0]


def norm(v):
    return float(np.linalg.norm(v))


def cosine(a, b):
    na = norm(a)
    nb = norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def angle_deg(a, b):
    c = np.clip(cosine(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# ---------------------------------------------------------------------
# Behavioral coupling
# ---------------------------------------------------------------------

def analyze_probe(model, row, delta):
    base_values = np.array(
        [
            row["A"],
            row["B"],
            row["C"],
            row["interaction"],
        ],
        dtype=np.float32,
    )

    base_x = torch.tensor(
        base_values[None, :],
        dtype=torch.float32,
    )

    recon_base, z_base = evaluate(model, base_x)

    base_error = norm(recon_base - base_values)

    result = {
        "probe": row["probe"],
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "interaction": row["interaction"],
        "base_reconstruction_error": base_error,
    }

    total_latent = 0.0

    for j, factor in enumerate(FACTORS):
        plus_values = base_values.copy()
        minus_values = base_values.copy()

        plus_values[j] += delta
        minus_values[j] -= delta

        x_plus = torch.tensor(
            plus_values[None, :],
            dtype=torch.float32,
        )
        x_minus = torch.tensor(
            minus_values[None, :],
            dtype=torch.float32,
        )

        recon_plus, z_plus = evaluate(model, x_plus)
        recon_minus, z_minus = evaluate(model, x_minus)

        dz = (z_plus - z_minus) / (2.0 * delta)
        dy = (recon_plus - recon_minus) / (2.0 * delta)

        latent_mag = norm(dz)
        output_mag = norm(dy)

        # This is the key behavioral coupling quantity:
        #
        # How much output movement is produced per unit latent movement?
        gain = output_mag / max(latent_mag, 1e-12)

        # For an identity-like autoencoder, changing input factor j should
        # change output mostly along basis vector j. This lets us distinguish
        # output movement from merely arbitrary reconstruction drift.
        expected = np.zeros(4, dtype=np.float32)
        expected[j] = 1.0

        output_alignment = cosine(dy, expected)
        output_angle = angle_deg(dy, expected)

        # Reconstruction error at each side.
        err_plus = norm(recon_plus - plus_values)
        err_minus = norm(recon_minus - minus_values)

        # How much the reconstruction itself moves.
        midpoint_output = (recon_plus + recon_minus) / 2.0
        output_asymmetry = norm(
            midpoint_output - recon_base
        )

        result[f"latent_mag_{factor}"] = latent_mag
        result[f"output_mag_{factor}"] = output_mag
        result[f"output_per_latent_{factor}"] = gain
        result[f"output_alignment_{factor}"] = output_alignment
        result[f"output_angle_{factor}"] = output_angle
        result[f"recon_error_plus_{factor}"] = err_plus
        result[f"recon_error_minus_{factor}"] = err_minus
        result[f"output_asymmetry_{factor}"] = output_asymmetry

        total_latent += latent_mag

    result["total_latent_sensitivity"] = total_latent

    return result


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------

def pearson_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return float("nan")

    x = x[mask]
    y = y[mask]

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows):
    summary = {
        "n_probes": len(rows),
        "delta": DELTA,
        "top_percent": TOP_PERCENT,
        "factors": {},
    }

    for factor in FACTORS:
        latent = np.array(
            [r[f"latent_mag_{factor}"] for r in rows]
        )
        output = np.array(
            [r[f"output_mag_{factor}"] for r in rows]
        )
        gain = np.array(
            [r[f"output_per_latent_{factor}"] for r in rows]
        )
        alignment = np.array(
            [r[f"output_alignment_{factor}"] for r in rows]
        )
        angle = np.array(
            [r[f"output_angle_{factor}"] for r in rows]
        )

        threshold = np.quantile(
            np.array([r["total_latent_sensitivity"] for r in rows]),
            1.0 - TOP_PERCENT,
        )

        top = np.array([
            r["total_latent_sensitivity"] >= threshold
            for r in rows
        ])

        summary["factors"][factor] = {
            "latent_median": float(np.median(latent)),
            "latent_mean": float(np.mean(latent)),
            "latent_max": float(np.max(latent)),
            "output_median": float(np.median(output)),
            "output_mean": float(np.mean(output)),
            "output_max": float(np.max(output)),
            "gain_median": float(np.median(gain)),
            "gain_mean": float(np.mean(gain)),
            "correlation_latent_output": pearson_safe(
                latent,
                output,
            ),
            "correlation_latent_gain": pearson_safe(
                latent,
                gain,
            ),
            "alignment_median": float(np.median(alignment)),
            "angle_median": float(np.median(angle)),
            "top10_latent_median": float(
                np.median(latent[top])
            ),
            "rest_latent_median": float(
                np.median(latent[~top])
            ),
            "top10_output_median": float(
                np.median(output[top])
            ),
            "rest_output_median": float(
                np.median(output[~top])
            ),
            "top10_gain_median": float(
                np.median(gain[top])
            ),
            "rest_gain_median": float(
                np.median(gain[~top])
            ),
            "top10_alignment_median": float(
                np.median(alignment[top])
            ),
            "rest_alignment_median": float(
                np.median(alignment[~top])
            ),
            "top10_angle_median": float(
                np.median(angle[top])
            ),
            "rest_angle_median": float(
                np.median(angle[~top])
            ),
        }

    return summary


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def make_plots(rows):
    total = np.array(
        [r["total_latent_sensitivity"] for r in rows]
    )

    top_threshold = np.quantile(
        total,
        1.0 - TOP_PERCENT,
    )

    top = total >= top_threshold

    # One plot per factor, not subplots, to keep visual interpretation clean.
    for factor in FACTORS:
        latent = np.array([
            r[f"latent_mag_{factor}"] for r in rows
        ])
        output = np.array([
            r[f"output_mag_{factor}"] for r in rows
        ])

        plt.figure(figsize=(8, 7))
        plt.scatter(
            latent[~top],
            output[~top],
            s=12,
            alpha=0.45,
            label="other 90%",
        )
        plt.scatter(
            latent[top],
            output[top],
            s=24,
            alpha=0.8,
            label="top 10% total sensitivity",
        )

        # Reference diagonal is only visual; scales are not assumed equal.
        plt.xlabel(f"latent sensitivity |dZ/d{factor}|")
        plt.ylabel(f"output sensitivity |dY/d{factor}|")
        plt.title(
            f"V6 — {factor}: latent sensitivity vs output sensitivity"
        )
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUT_DIR / f"v6_{factor}_latent_vs_output.png",
            dpi=160,
        )
        plt.close()

    # Gain versus total latent sensitivity.
    plt.figure(figsize=(9, 7))

    for factor in FACTORS:
        x = np.array([
            r[f"latent_mag_{factor}"] for r in rows
        ])
        y = np.array([
            r[f"output_per_latent_{factor}"] for r in rows
        ])

        plt.scatter(
            x,
            y,
            s=18,
            alpha=0.45,
            label=factor,
        )

    plt.xlabel("latent sensitivity")
    plt.ylabel("output movement / latent movement")
    plt.title("V6 — Behavioral gain across the latent field")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "v6_behavioral_gain.png",
        dpi=160,
    )
    plt.close()

    # Top 10% vs rest.
    top_means = []
    rest_means = []

    for factor in FACTORS:
        output = np.array([
            r[f"output_mag_{factor}"] for r in rows
        ])

        top_means.append(float(np.median(output[top])))
        rest_means.append(float(np.median(output[~top])))

    x = np.arange(len(FACTORS))

    plt.figure(figsize=(9, 6))
    width = 0.35

    plt.bar(
        x - width / 2,
        rest_means,
        width,
        label="rest 90%",
    )
    plt.bar(
        x + width / 2,
        top_means,
        width,
        label="top 10%",
    )

    plt.xticks(x, FACTORS)
    plt.ylabel("median output sensitivity")
    plt.title("V6 — Output sensitivity: top latent-sensitivity probes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "v6_top10_vs_rest.png",
        dpi=160,
    )
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 90)
    print("V6 — BEHAVIORAL COUPLING OF LATENT GEOMETRY")
    print("=" * 90)
    print()

    model_path = find_file(MODEL_NAMES)

    if model_path is None:
        raise FileNotFoundError(
            "Could not find best_model.pt/model.pt/tiny_autoencoder.pt "
            "under the current project directory."
        )

    print(f"Model: {model_path}")

    data_path = find_file(DATA_NAMES)

    if data_path is None:
        raise FileNotFoundError(
            "Could not find dataset.pt/data.pt under the current project."
        )

    print(f"Dataset: {data_path}")

    print(f"Delta: {DELTA}")
    print()

    model = load_model(model_path)
    model.eval()

    X = normalize_dataset(load_dataset(data_path))

    v5_path = load_v5_probes()

    if v5_path is not None:
        print(f"Reusing V5 probes: {v5_path}")
        probes = read_v5_probes(v5_path)
    else:
        print("V5 probe CSV not found; generating deterministic probes.")
        probes = generate_probes_from_dataset(X)

    print(f"Probes: {len(probes)}")
    print()

    rows = []

    for i, row in enumerate(probes):
        result = analyze_probe(
            model,
            row,
            DELTA,
        )

        rows.append(result)

        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"probe {i + 1:5d}/{len(probes)}"
            )

    summary = summarize(rows)

    # Mark top 10% exactly as a per-row field.
    total = np.array([
        r["total_latent_sensitivity"]
        for r in rows
    ])

    threshold = float(
        np.quantile(total, 1.0 - TOP_PERCENT)
    )

    for r in rows:
        r["top_10pct"] = (
            r["total_latent_sensitivity"] >= threshold
        )

    # CSV.
    csv_path = OUT_DIR / "v6_behavioral_coupling.csv"

    import csv

    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment": "V6_behavioral_coupling",
        "model": str(model_path),
        "dataset": str(data_path),
        "v5_probe_source": str(v5_path) if v5_path else None,
        "delta": DELTA,
        "top_percent": TOP_PERCENT,
        "n_probes": len(rows),
        "top10_threshold_total_latent_sensitivity": threshold,
        "summary": summary,
        "interpretation_guardrails": [
            (
                "A correlation between latent and output sensitivity "
                "supports behavioral coupling but does not establish causality."
            ),
            (
                "A large latent response with small output response would "
                "suggest representational movement that the decoder compensates."
            ),
            (
                "A large output response per unit latent movement indicates "
                "that local latent movement is behaviorally amplified."
            ),
            (
                "The output alignment metric checks whether reconstructed "
                "movement follows the known input intervention direction."
            ),
        ],
    }

    json_path = OUT_DIR / "v6_behavioral_coupling_results.json"
    json_path.write_text(
        json.dumps(report, indent=2)
    )

    make_plots(rows)

    print()
    print("=" * 90)
    print("V6 COMPLETE")
    print("=" * 90)
    print()
    print("Summary:")

    for factor in FACTORS:
        s = summary["factors"][factor]

        print(
            f"{factor:12s} "
            f"latent/output r={s['correlation_latent_output']:+.4f} | "
            f"gain median={s['gain_median']:.5f} | "
            f"top10 output={s['top10_output_median']:.5f} | "
            f"rest output={s['rest_output_median']:.5f}"
        )

    print()
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"DIR : {OUT_DIR}")


if __name__ == "__main__":
    main()
