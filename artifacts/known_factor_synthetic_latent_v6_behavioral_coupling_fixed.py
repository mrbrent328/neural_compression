#!/usr/bin/env python3
"""
V6 — Behavioral Coupling of the REAL V5 Image Autoencoder

This version deliberately imports the actual TinyAutoencoder from
known_factor_synthetic_latent_v1 instead of recreating an architecture.

It reuses V5's probe coordinates and measures, for each intervention:

    input perturbation -> latent movement -> reconstruction movement

The model checkpoint shown by the previous run has the architecture:

    4096 -> 256 -> 256 -> 64 -> 2
     2   -> 64  -> 256 -> 4096

so the reconstruction is treated as a 4096-dimensional output vector.

Run from:
    /home/xetal/Desktop/latent_model_test/synthetic_latent_v1

Example:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    python3 known_factor_synthetic_latent_v6_behavioral_coupling_fixed.py

Outputs:
    synthetic_latent_v6/
        v6_behavioral_coupling.csv
        v6_behavioral_coupling_results.json
        v6_*_latent_vs_output.png
        v6_behavioral_gain.png
        v6_top10_vs_rest.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from known_factor_synthetic_latent_v1 import ...

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "synthetic_latent_v6"
OUT_DIR.mkdir(exist_ok=True)

DELTA = 0.01
TOP_PERCENT = 0.10

FACTORS = ["A", "B", "C", "interaction"]


def import_real_model():
    """
    Import the architecture that produced the V5 checkpoint.

    This is intentionally not duplicated here. That prevents another
    architecture mismatch if V1 changes.
    """
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from known_factor_synthetic_latent_v1 import TinyAutoencoder

    return TinyAutoencoder


def load_real_model():
    TinyAutoencoder = import_real_model()

    model_path = ROOT / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )

    # The previous traceback shows this is a state_dict checkpoint.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        raise RuntimeError(
            "best_model.pt is not a dictionary/state_dict checkpoint."
        )

    model = TinyAutoencoder()

    # Load strictly. If V1 and the checkpoint disagree, we want to know.
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, model_path


def load_v5_probes():
    path = ROOT / "synthetic_latent_v5" / "probe_vector_geometry.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find V5 probe CSV:\n{path}"
        )

    rows = []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = ["probe", "A", "B", "C", "interaction"]
        missing = [x for x in required if x not in reader.fieldnames]

        if missing:
            raise RuntimeError(
                f"V5 CSV is missing columns: {missing}"
            )

        for row in reader:
            rows.append(
                {
                    "probe": int(row["probe"]),
                    "A": float(row["A"]),
                    "B": float(row["B"]),
                    "C": float(row["C"]),
                    "interaction": float(row["interaction"]),
                }
            )

    return path, rows


def model_forward(model, x):
    """
    Accommodate the real V1 model whether forward() returns:
        reconstruction, latent
    or another tuple-like form.

    Prefer the public encode/decode methods when available.
    """
    with torch.no_grad():
        if hasattr(model, "encode") and hasattr(model, "decode"):
            z = model.encode(x)
            y = model.decode(z)
            return y, z

        out = model(x)

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]

    raise RuntimeError(
        "Could not obtain reconstruction and latent from TinyAutoencoder."
    )


def evaluate(model, values):
    x = torch.tensor(
        np.asarray(values, dtype=np.float32)[None, :],
        dtype=torch.float32,
    )

    with torch.no_grad():
        y, z = model_forward(model, x)

    y = y.detach().cpu().numpy().reshape(-1)
    z = z.detach().cpu().numpy().reshape(-1)

    return y, z


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


def analyze_probe(model, row):
    """
    IMPORTANT:
    The V5 CSV gives A/B/C/interaction as the probe coordinates used by
    the geometry experiment. Those are the coordinates passed through the
    actual model's input representation by the V5 experiment.

    We therefore use those exact probe vectors here and do not invent a
    new probe distribution.
    """
    base_values = np.array(
        [
            row["A"],
            row["B"],
            row["C"],
            row["interaction"],
        ],
        dtype=np.float32,
    )

    recon_base, z_base = evaluate(model, base_values)

    # If the real model expects 4096 inputs, fail with a useful message
    # instead of producing nonsense.
    if base_values.size != 4096:
        raise RuntimeError(
            "The V5 CSV contains four factor coordinates, but the real "
            "TinyAutoencoder expects a 4096-dimensional image input. "
            "V6 needs the original V5 probe-to-image construction, not "
            "the four scalar coordinates alone."
        )

    result = {
        "probe": row["probe"],
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "interaction": row["interaction"],
        "base_reconstruction_error": norm(
            recon_base - base_values
        ),
    }

    total_latent = 0.0

    for j, factor in enumerate(FACTORS):
        plus = base_values.copy()
        minus = base_values.copy()

        plus[j] += DELTA
        minus[j] -= DELTA

        recon_plus, z_plus = evaluate(model, plus)
        recon_minus, z_minus = evaluate(model, minus)

        dz = (z_plus - z_minus) / (2.0 * DELTA)
        dy = (recon_plus - recon_minus) / (2.0 * DELTA)

        latent_mag = norm(dz)
        output_mag = norm(dy)

        gain = output_mag / max(latent_mag, 1e-12)

        result[f"latent_mag_{factor}"] = latent_mag
        result[f"output_mag_{factor}"] = output_mag
        result[f"output_per_latent_{factor}"] = gain

        total_latent += latent_mag

    result["total_latent_sensitivity"] = total_latent
    return result


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
    total = np.array(
        [r["total_latent_sensitivity"] for r in rows]
    )

    threshold = float(
        np.quantile(total, 1.0 - TOP_PERCENT)
    )

    top = total >= threshold

    result = {
        "n_probes": len(rows),
        "delta": DELTA,
        "top_percent": TOP_PERCENT,
        "top10_threshold": threshold,
        "factors": {},
    }

    for factor in FACTORS:
        latent = np.array([
            r[f"latent_mag_{factor}"] for r in rows
        ])

        output = np.array([
            r[f"output_mag_{factor}"] for r in rows
        ])

        gain = np.array([
            r[f"output_per_latent_{factor}"] for r in rows
        ])

        result["factors"][factor] = {
            "latent_mean": float(np.mean(latent)),
            "latent_median": float(np.median(latent)),
            "latent_max": float(np.max(latent)),
            "output_mean": float(np.mean(output)),
            "output_median": float(np.median(output)),
            "output_max": float(np.max(output)),
            "gain_mean": float(np.mean(gain)),
            "gain_median": float(np.median(gain)),
            "correlation_latent_output": pearson_safe(
                latent,
                output,
            ),
            "correlation_latent_gain": pearson_safe(
                latent,
                gain,
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
        }

    return result


def make_plots(rows):
    total = np.array([
        r["total_latent_sensitivity"]
        for r in rows
    ])

    threshold = np.quantile(
        total,
        1.0 - TOP_PERCENT,
    )

    top = total >= threshold

    for factor in FACTORS:
        latent = np.array([
            r[f"latent_mag_{factor}"]
            for r in rows
        ])

        output = np.array([
            r[f"output_mag_{factor}"]
            for r in rows
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
            label="top 10%",
        )

        plt.xlabel(f"|dZ/d{factor}|")
        plt.ylabel(f"|dY/d{factor}|")
        plt.title(
            f"V6 — {factor}: latent vs reconstruction sensitivity"
        )
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUT_DIR / f"v6_{factor}_latent_vs_output.png",
            dpi=160,
        )
        plt.close()

    plt.figure(figsize=(9, 7))

    for factor in FACTORS:
        x = np.array([
            r[f"latent_mag_{factor}"]
            for r in rows
        ])

        y = np.array([
            r[f"output_per_latent_{factor}"]
            for r in rows
        ])

        plt.scatter(
            x,
            y,
            s=18,
            alpha=0.45,
            label=factor,
        )

    plt.xlabel("latent sensitivity")
    plt.ylabel("reconstruction movement / latent movement")
    plt.title("V6 — Behavioral gain")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "v6_behavioral_gain.png",
        dpi=160,
    )
    plt.close()

    top_outputs = []
    rest_outputs = []

    for factor in FACTORS:
        output = np.array([
            r[f"output_mag_{factor}"]
            for r in rows
        ])

        top_outputs.append(float(np.median(output[top])))
        rest_outputs.append(float(np.median(output[~top])))

    x = np.arange(len(FACTORS))
    width = 0.35

    plt.figure(figsize=(9, 6))

    plt.bar(
        x - width / 2,
        rest_outputs,
        width,
        label="rest 90%",
    )

    plt.bar(
        x + width / 2,
        top_outputs,
        width,
        label="top 10%",
    )

    plt.xticks(x, FACTORS)
    plt.ylabel("median reconstruction sensitivity")
    plt.title("V6 — Top latent-sensitivity probes vs rest")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "v6_top10_vs_rest.png",
        dpi=160,
    )
    plt.close()


def write_csv(rows):
    path = OUT_DIR / "v6_behavioral_coupling.csv"

    fields = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def main():
    print("=" * 90)
    print("V6 — BEHAVIORAL COUPLING OF REAL V5 AUTOENCODER")
    print("=" * 90)
    print()

    model, model_path = load_real_model()

    print(f"Model:   {model_path}")
    print(f"Model type: {type(model).__name__}")

    # Verify the checkpoint/model dimensionality before doing a long run.
    first_weight = None

    for name, parameter in model.named_parameters():
        if name == "encoder.0.weight":
            first_weight = parameter
            break

    if first_weight is not None:
        print(
            f"Input dimension reported by model: "
            f"{first_weight.shape[1]}"
        )

    v5_path, probes = load_v5_probes()

    print(f"V5 probes: {v5_path}")
    print(f"Probe count: {len(probes)}")
    print(f"Finite-difference delta: {DELTA}")
    print()

    # Do a compatibility check BEFORE the full run.
    #
    # The V5 CSV contains four scalar coordinates, while the real image
    # model has 4096 input pixels. Therefore V6 must have the same probe
    # construction used by V5 to convert those factors into images.
    #
    # We refuse to guess that construction.
    first = np.array(
        [
            probes[0]["A"],
            probes[0]["B"],
            probes[0]["C"],
            probes[0]["interaction"],
        ],
        dtype=np.float32,
    )

    input_dim = first_weight.shape[1] if first_weight is not None else None

    if input_dim != 4:
        raise RuntimeError(
            "\n"
            "V6 stopped before running because the real model expects "
            f"{input_dim} input values, while the V5 CSV stores only the "
            "four factor coordinates A/B/C/interaction.\n\n"
            "This is NOT another architecture bug. It means V5 used a "
            "factor-to-4096-pixel probe construction that V6 needs to reuse.\n"
            "I deliberately refuse to invent that mapping because doing so "
            "would invalidate the behavioral experiment.\n\n"
            "The next fix should import/call V5's actual probe construction "
            "function and then run these same finite differences through the "
            "real 4096-dimensional image model."
        )

    rows = []

    for i, row in enumerate(probes):
        rows.append(
            analyze_probe(
                model,
                row,
            )
        )

        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"probe {i + 1:5d}/{len(probes)}"
            )

    summary = summarize(rows)

    threshold = summary["top10_threshold"]

    for r in rows:
        r["top_10pct"] = (
            r["total_latent_sensitivity"] >= threshold
        )

    csv_path = write_csv(rows)

    json_path = OUT_DIR / "v6_behavioral_coupling_results.json"

    report = {
        "experiment": "V6_behavioral_coupling",
        "model": str(model_path),
        "v5_probe_source": str(v5_path),
        "delta": DELTA,
        "summary": summary,
        "guardrails": [
            "This measures correlation, not causation.",
            "Large latent movement with small reconstruction movement would indicate decoder compensation.",
            "Large reconstruction movement per unit latent movement indicates local behavioral amplification.",
            "The V6 probe locations are inherited from V5.",
        ],
    }

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    make_plots(rows)

    print()
    print("=" * 90)
    print("V6 COMPLETE")
    print("=" * 90)
    print()

    for factor in FACTORS:
        s = summary["factors"][factor]

        print(
            f"{factor:12s} "
            f"latent→output r={s['correlation_latent_output']:+.4f} | "
            f"gain median={s['gain_median']:.6g} | "
            f"top10 output={s['top10_output_median']:.6g} | "
            f"rest output={s['rest_output_median']:.6g}"
        )

    print()
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"DIR : {OUT_DIR}")


if __name__ == "__main__":
    main()
