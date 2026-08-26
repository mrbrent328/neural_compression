#!/usr/bin/env python3
"""
E33 — Latent Geometry Intervention

Goal
----
E32 showed that the decoder induces a non-Euclidean local metric:

    G = J^T J

where J is the decoder Jacobian.

E33 asks the stronger question:

    Can we ACT on that geometry?

For selected latent points z:

    1. Compute decoder Jacobian J.
    2. Compute G = J^T J.
    3. Find the most output-sensitive latent direction.
    4. Find the least output-sensitive latent direction.
    5. Walk along each direction at several step sizes.
    6. Decode every perturbed latent.
    7. Compare:
         - first-order metric prediction
         - actual decoded movement
         - alignment with predicted Jacobian movement

If the decoder metric is genuinely useful, then:

    actual output movement ≈ predicted metric movement

for sufficiently small steps.

We also test whether the high-sensitivity direction produces
substantially more output change than an equal-sized low-sensitivity
direction.

This uses the REAL V1 checkpoint and REAL V5 synthetic factor machinery.
It does not create a new synthetic mapping.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

V1_PATH = SCRIPT_DIR.parent / "known_factor_synthetic_latent_v1.py"
V5_PATH = SCRIPT_DIR / "known_factor_synthetic_latent_v5_vector_field.py"
MODEL_PATH = SCRIPT_DIR / "best_model.pt"

OUT = SCRIPT_DIR / "synthetic_latent_e33_geometry_intervention"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

DELTA = 0.01

# Latent-space walking distances.
# Small values test the local metric.
# Larger values test how far the local approximation remains useful.
STEPS = [
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
]

# Number of representative held-out latent points to test.
N_POINTS = 12

# Use fixed deterministic selection.
SEED = 1337


# ---------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------

def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required source file:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module from:\n{path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------
# Load model and V5 machinery
# ---------------------------------------------------------------------

def load_model():
    v1 = load_module(
        "real_latent_v1_e33",
        V1_PATH,
    )

    if not hasattr(v1, "TinyAutoencoder"):
        raise RuntimeError(
            "V1 does not expose TinyAutoencoder."
        )

    rows = int(getattr(v1, "ROWS", 64))
    cols = int(getattr(v1, "COLS", 64))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "best_model.pt is not a checkpoint dictionary."
        )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )

    if state is None:
        raise RuntimeError(
            "Could not find model state dictionary."
        )

    model = v1.TinyAutoencoder(
        rows * cols,
        2,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return model, rows, cols


def load_v5():
    return load_module(
        "real_latent_v5_e33",
        V5_PATH,
    )


# ---------------------------------------------------------------------
# Decoder geometry
# ---------------------------------------------------------------------

@torch.no_grad()
def decoder_jacobian(model, z):
    """
    Finite-difference decoder Jacobian.

    J:
        [output_dimension, latent_dimension]

    For this experiment:
        output_dimension = 4096
        latent_dimension = 2
    """

    columns = []

    for axis in range(2):
        zp = z.clone()
        zm = z.clone()

        zp[:, axis] += DELTA
        zm[:, axis] -= DELTA

        yp = model.decode(zp)
        ym = model.decode(zm)

        derivative = (
            yp - ym
        ) / (2.0 * DELTA)

        columns.append(
            derivative.reshape(-1).cpu().numpy()
        )

    return np.stack(
        columns,
        axis=1,
    )


def metric_from_jacobian(J):
    """
    G = J^T J

    Returns:
        G
        eigenvalues
        low-sensitivity direction
        high-sensitivity direction
    """

    G = J.T @ J

    eigenvalues, eigenvectors = np.linalg.eigh(G)

    low = eigenvectors[:, 0]
    high = eigenvectors[:, -1]

    return (
        G,
        eigenvalues,
        low,
        high,
    )


# ---------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------

def norm(x):
    return float(
        np.linalg.norm(x)
    )


def cosine(a, b):
    na = norm(a)
    nb = norm(b)

    if na < 1e-12 or nb < 1e-12:
        return float("nan")

    return float(
        np.dot(a, b) / (na * nb)
    )


def metric_magnitude(v, G):
    value = float(
        v @ G @ v
    )

    return math.sqrt(
        max(value, 0.0)
    )


def output_change(
    baseline,
    perturbed,
):
    delta = (
        perturbed - baseline
    ).reshape(-1)

    return (
        norm(delta.cpu().numpy()),
        delta.cpu().numpy(),
    )


# ---------------------------------------------------------------------
# Select representative latent points
# ---------------------------------------------------------------------

def select_points(z):
    """
    Select deterministic points spread through the latent population.

    We use a simple grid-like quantile strategy rather than choosing
    arbitrary random points.
    """

    z_np = np.asarray(z)

    rng = np.random.default_rng(SEED)

    n = len(z_np)

    if n <= N_POINTS:
        return np.arange(n)

    # First include points closest to several latent quantile locations.
    q = np.linspace(
        0.05,
        0.95,
        N_POINTS,
    )

    order = np.argsort(
        z_np[:, 0] + 0.37 * z_np[:, 1]
    )

    candidates = []

    for qi in q:
        idx = int(
            qi * (n - 1)
        )

        candidates.append(
            int(order[idx])
        )

    # Remove duplicates.
    candidates = list(
        dict.fromkeys(candidates)
    )

    # Fill if necessary.
    if len(candidates) < N_POINTS:
        remaining = [
            i for i in range(n)
            if i not in candidates
        ]

        rng.shuffle(remaining)

        candidates.extend(
            remaining[
                :N_POINTS - len(candidates)
            ]
        )

    return np.array(
        candidates[:N_POINTS],
        dtype=int,
    )


# ---------------------------------------------------------------------
# Main intervention experiment
# ---------------------------------------------------------------------

def run():
    print("=" * 90)
    print("E33 — LATENT GEOMETRY INTERVENTION")
    print("=" * 90)
    print()

    print(f"V1:       {V1_PATH}")
    print(f"V5:       {V5_PATH}")
    print(f"Model:    {MODEL_PATH}")
    print(f"Delta:    {DELTA}")
    print(f"Steps:    {STEPS}")
    print()

    model, rows, cols = load_model()
    v5 = load_v5()

    required = [
        "generate_exact_v1_dataset",
        "generate_from_factors",
        "encode_factors",
    ]

    for name in required:
        if not hasattr(v5, name):
            raise RuntimeError(
                f"V5 does not expose required function: {name}"
            )

    # ---------------------------------------------------------------
    # Recover exact V5 dataset and held-out probes.
    # ---------------------------------------------------------------

    (
        standardized,
        factors,
        patterns,
        norm_mean,
        norm_std,
    ) = v5.generate_exact_v1_dataset()

    split = int(
        len(factors) * 0.8
    )

    heldout = factors[split:]

    print(
        f"Full dataset:       {len(factors)}"
    )

    print(
        f"Held-out probes:    {len(heldout)}"
    )

    # ---------------------------------------------------------------
    # Encode exact held-out factor points.
    # ---------------------------------------------------------------

    device = getattr(
        v5,
        "device",
        torch.device("cpu"),
    )

    try:
        z_np = v5.encode_factors(
            model,
            device,
            heldout,
            patterns,
            norm_mean,
            norm_std,
        )
    except Exception:
        z_np = v5.encode_factors(
            model,
            torch.device("cpu"),
            heldout,
            patterns,
            norm_mean,
            norm_std,
        )

    z_np = np.asarray(
        z_np,
        dtype=np.float32,
    )

    if z_np.ndim != 2 or z_np.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected latent shape: {z_np.shape}"
        )

    print(
        f"Latent shape:       {z_np.shape}"
    )
    print()

    selected = select_points(
        z_np
    )

    print(
        f"Selected points:    {selected.tolist()}"
    )
    print()

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    rows_out = []

    # Keep detailed decoded arrays for visualization.
    visual_cases = []

    for point_number, probe_idx in enumerate(selected):

        z = torch.tensor(
            z_np[probe_idx:probe_idx + 1],
            dtype=torch.float32,
        )

        # -----------------------------------------------------------
        # Local decoder metric.
        # -----------------------------------------------------------

        J = decoder_jacobian(
            model,
            z,
        )

        G, eigenvalues, low, high = (
            metric_from_jacobian(J)
        )

        lambda_low = float(
            eigenvalues[0]
        )

        lambda_high = float(
            eigenvalues[1]
        )

        sensitivity_low = math.sqrt(
            max(lambda_low, 0.0)
        )

        sensitivity_high = math.sqrt(
            max(lambda_high, 0.0)
        )

        anisotropy = (
            lambda_high /
            max(lambda_low, 1e-12)
        )

        print(
            f"[{point_number + 1:02d}/{len(selected):02d}] "
            f"probe={probe_idx:4d} "
            f"z=({z_np[probe_idx,0]:+.3f}, "
            f"{z_np[probe_idx,1]:+.3f}) "
            f""
            f"lambda_low={lambda_low:.5g} "
            f"lambda_high={lambda_high:.5g}"
        )

        # -----------------------------------------------------------
        # Baseline decode.
        # -----------------------------------------------------------

        with torch.no_grad():
            baseline = model.decode(
                z
            )

        # -----------------------------------------------------------
        # Test high and low sensitivity directions.
        # -----------------------------------------------------------

        directions = {
            "high": high,
            "low": low,
        }

        for direction_name, direction in directions.items():

            direction_t = torch.tensor(
                direction,
                dtype=torch.float32,
            ).view(1, 2)

            predicted_unit_change = (
                J @ direction
            )

            predicted_unit_mag = norm(
                predicted_unit_change
            )

            metric_unit_mag = metric_magnitude(
                direction,
                G,
            )

            # These should agree up to numerical precision.
            metric_vs_jacobian_ratio = (
                metric_unit_mag /
                max(predicted_unit_mag, 1e-12)
            )

            for step in STEPS:

                # --------------------------------------------------
                # Positive direction
                # --------------------------------------------------

                z_plus = (
                    z +
                    step * direction_t
                )

                # --------------------------------------------------
                # Negative direction
                # --------------------------------------------------

                z_minus = (
                    z -
                    step * direction_t
                )

                with torch.no_grad():
                    y_plus = model.decode(
                        z_plus
                    )

                    y_minus = model.decode(
                        z_minus
                    )

                plus_mag, plus_delta = (
                    output_change(
                        baseline,
                        y_plus,
                    )
                )

                minus_mag, minus_delta = (
                    output_change(
                        baseline,
                        y_minus,
                    )
                )

                # --------------------------------------------------
                # First-order prediction.
                # --------------------------------------------------

                predicted_mag = (
                    step *
                    predicted_unit_mag
                )

                # --------------------------------------------------
                # Symmetric actual movement.
                #
                # This suppresses some first-order asymmetry and
                # gives us another estimate of local behavior.
                # --------------------------------------------------

                symmetric_delta = (
                    y_plus -
                    y_minus
                ).reshape(-1).cpu().numpy()

                symmetric_mag = (
                    0.5 *
                    norm(symmetric_delta)
                )

                # --------------------------------------------------
                # Directional alignment.
                #
                # Jv predicts the output-space direction for an
                # infinitesimal positive movement.
                # --------------------------------------------------

                plus_alignment = cosine(
                    plus_delta,
                    predicted_unit_change,
                )

                minus_alignment = cosine(
                    -minus_delta,
                    predicted_unit_change,
                )

                # --------------------------------------------------
                # Local metric prediction quality.
                # --------------------------------------------------

                plus_ratio = (
                    plus_mag /
                    max(predicted_mag, 1e-12)
                )

                minus_ratio = (
                    minus_mag /
                    max(predicted_mag, 1e-12)
                )

                symmetric_ratio = (
                    symmetric_mag /
                    max(predicted_mag, 1e-12)
                )

                row = {
                    "point": point_number,
                    "probe": int(probe_idx),
                    "z0": float(z_np[probe_idx, 0]),
                    "z1": float(z_np[probe_idx, 1]),
                    "direction": direction_name,
                    "step": float(step),

                    "lambda_low": lambda_low,
                    "lambda_high": lambda_high,
                    "sensitivity_low": sensitivity_low,
                    "sensitivity_high": sensitivity_high,
                    "anisotropy": anisotropy,

                    "direction_z0": float(direction[0]),
                    "direction_z1": float(direction[1]),

                    "metric_unit_magnitude": metric_unit_mag,
                    "jacobian_unit_magnitude": predicted_unit_mag,
                    "metric_vs_jacobian_ratio":
                        metric_vs_jacobian_ratio,

                    "predicted_output_magnitude":
                        predicted_mag,

                    "actual_plus_magnitude":
                        plus_mag,

                    "actual_minus_magnitude":
                        minus_mag,

                    "symmetric_actual_magnitude":
                        symmetric_mag,

                    "plus_ratio":
                        plus_ratio,

                    "minus_ratio":
                        minus_ratio,

                    "symmetric_ratio":
                        symmetric_ratio,

                    "plus_alignment":
                        plus_alignment,

                    "minus_alignment":
                        minus_alignment,
                }

                rows_out.append(
                    row
                )

                # Save a representative visual case.
                if (
                    point_number == 0
                    and abs(step - 1.0) < 1e-9
                ):
                    visual_cases.append(
                        {
                            "direction": direction_name,
                            "step": step,
                            "baseline": baseline.squeeze().cpu().numpy(),
                            "plus": y_plus.squeeze().cpu().numpy(),
                            "minus": y_minus.squeeze().cpu().numpy(),
                        }
                    )

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    csv_path = (
        OUT /
        "e33_intervention_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows_out[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows_out)

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    def arr(
        direction,
        field,
        step=None,
    ):
        values = []

        for r in rows_out:
            if r["direction"] != direction:
                continue

            if step is not None:
                if abs(r["step"] - step) > 1e-9:
                    continue

            values.append(
                r[field]
            )

        return np.asarray(
            values,
            dtype=float,
        )

    summary = {
        "experiment": (
            "E33 — latent geometry intervention"
        ),

        "purpose": (
            "Test whether decoder-induced local "
            "metric predicts actual decoded behavior "
            "under controlled latent interventions."
        ),

        "configuration": {
            "delta": DELTA,
            "steps": STEPS,
            "selected_points": int(
                len(selected)
            ),
        },

        "global": {},

        "directions": {},
    }

    for direction in [
        "high",
        "low",
    ]:

        summary["directions"][direction] = {}

        for step in STEPS:

            plus = arr(
                direction,
                "actual_plus_magnitude",
                step,
            )

            minus = arr(
                direction,
                "actual_minus_magnitude",
                step,
            )

            predicted = arr(
                direction,
                "predicted_output_magnitude",
                step,
            )

            plus_ratio = arr(
                direction,
                "plus_ratio",
                step,
            )

            minus_ratio = arr(
                direction,
                "minus_ratio",
                step,
            )

            plus_alignment = arr(
                direction,
                "plus_alignment",
                step,
            )

            minus_alignment = arr(
                direction,
                "minus_alignment",
                step,
            )

            summary["directions"][direction][
                str(step)
            ] = {
                "predicted_magnitude_median":
                    float(np.median(predicted)),

                "actual_plus_magnitude_median":
                    float(np.median(plus)),

                "actual_minus_magnitude_median":
                    float(np.median(minus)),

                "plus_prediction_ratio_median":
                    float(np.median(plus_ratio)),

                "minus_prediction_ratio_median":
                    float(np.median(minus_ratio)),

                "plus_alignment_median":
                    float(np.median(plus_alignment)),

                "minus_alignment_median":
                    float(np.median(minus_alignment)),
            }

    # ---------------------------------------------------------------
    # High-vs-low ratio.
    # ---------------------------------------------------------------

    high_by_step = {}
    low_by_step = {}

    for step in STEPS:

        high_actual = arr(
            "high",
            "actual_plus_magnitude",
            step,
        )

        low_actual = arr(
            "low",
            "actual_plus_magnitude",
            step,
        )

        high_by_step[str(step)] = float(
            np.median(high_actual)
        )

        low_by_step[str(step)] = float(
            np.median(low_actual)
        )

    summary["global"][
        "high_to_low_actual_gain"
    ] = {}

    for step in STEPS:

        high_value = high_by_step[
            str(step)
        ]

        low_value = low_by_step[
            str(step)
        ]

        summary["global"][
            "high_to_low_actual_gain"
        ][str(step)] = (
            high_value /
            max(low_value, 1e-12)
        )

    # ---------------------------------------------------------------
    # Overall local prediction quality.
    # ---------------------------------------------------------------

    all_ratios = np.asarray(
        [
            r["plus_ratio"]
            for r in rows_out
            if r["step"] <= 0.25
        ],
        dtype=float,
    )

    all_alignments = np.asarray(
        [
            r["plus_alignment"]
            for r in rows_out
            if r["step"] <= 0.25
        ],
        dtype=float,
    )

    summary["global"][
        "small_step_prediction_ratio"
    ] = {
        "median": float(
            np.median(all_ratios)
        ),
        "mean": float(
            np.mean(all_ratios)
        ),
        "p10": float(
            np.percentile(all_ratios, 10)
        ),
        "p90": float(
            np.percentile(all_ratios, 90)
        ),
    }

    summary["global"][
        "small_step_alignment"
    ] = {
        "median": float(
            np.median(all_alignments)
        ),
        "mean": float(
            np.mean(all_alignments)
        ),
        "p10": float(
            np.percentile(all_alignments, 10)
        ),
        "p90": float(
            np.percentile(all_alignments, 90)
        ),
    }

    json_path = (
        OUT /
        "e33_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # -----------------------------------------------------------------
    # Plot 1:
    # Actual vs predicted output movement
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    for direction in [
        "high",
        "low",
    ]:

        for step in STEPS:

            predicted = arr(
                direction,
                "predicted_output_magnitude",
                step,
            )

            actual = arr(
                direction,
                "actual_plus_magnitude",
                step,
            )

            plt.scatter(
                predicted,
                actual,
                s=32,
                alpha=0.65,
                label=f"{direction}, t={step}",
            )

    all_values = []

    for r in rows_out:
        all_values.extend(
            [
                r["predicted_output_magnitude"],
                r["actual_plus_magnitude"],
            ]
        )

    lo = min(all_values)
    hi = max(all_values)

    plt.plot(
        [lo, hi],
        [lo, hi],
        linewidth=1,
    )

    plt.xlabel(
        "predicted output movement"
    )

    plt.ylabel(
        "actual output movement"
    )

    plt.title(
        "E33 — predicted vs actual decoder movement"
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e33_predicted_vs_actual.png",
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 2:
    # Prediction ratio vs step size
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    for direction in [
        "high",
        "low",
    ]:

        medians = []

        for step in STEPS:

            values = arr(
                direction,
                "plus_ratio",
                step,
            )

            medians.append(
                np.median(values)
            )

        plt.plot(
            STEPS,
            medians,
            marker="o",
            label=direction,
        )

    plt.axhline(
        1.0,
        linewidth=1,
        linestyle="--",
    )

    plt.xlabel(
        "latent step size"
    )

    plt.ylabel(
        "actual / predicted movement"
    )

    plt.title(
        "E33 — local metric prediction accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e33_prediction_ratio_vs_step.png",
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 3:
    # High-vs-low actual response
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    high_values = [
        high_by_step[str(step)]
        for step in STEPS
    ]

    low_values = [
        low_by_step[str(step)]
        for step in STEPS
    ]

    plt.plot(
        STEPS,
        high_values,
        marker="o",
        label="high sensitivity",
    )

    plt.plot(
        STEPS,
        low_values,
        marker="o",
        label="low sensitivity",
    )

    plt.xlabel(
        "latent step size"
    )

    plt.ylabel(
        "actual decoded movement"
    )

    plt.title(
        "E33 — sensitive vs insensitive latent directions"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e33_high_vs_low.png",
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 4:
    # Alignment with predicted output direction.
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    for direction in [
        "high",
        "low",
    ]:

        medians = []

        for step in STEPS:

            values = arr(
                direction,
                "plus_alignment",
                step,
            )

            medians.append(
                np.median(values)
            )

        plt.plot(
            STEPS,
            medians,
            marker="o",
            label=direction,
        )

    plt.axhline(
        1.0,
        linewidth=1,
        linestyle="--",
    )

    plt.xlabel(
        "latent step size"
    )

    plt.ylabel(
        "cosine(actual, predicted)"
    )

    plt.title(
        "E33 — output-direction alignment"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e33_direction_alignment.png",
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 5:
    # Visual intervention for first selected point.
    # -----------------------------------------------------------------

    if visual_cases:

        for case in visual_cases:

            baseline = case["baseline"]
            plus = case["plus"]
            minus = case["minus"]

            diff_plus = (
                plus - baseline
            )

            diff_minus = (
                minus - baseline
            )

            fig = plt.figure(
                figsize=(12, 8)
            )

            panels = [
                (
                    baseline,
                    "baseline",
                ),
                (
                    plus,
                    f"+{case['direction']}",
                ),
                (
                    minus,
                    f"-{case['direction']}",
                ),
                (
                    diff_plus,
                    "positive difference",
                ),
                (
                    diff_minus,
                    "negative difference",
                ),
            ]

            for i, (
                image,
                title,
            ) in enumerate(panels):

                ax = fig.add_subplot(
                    2,
                    3,
                    i + 1,
                )

                ax.imshow(
                    image.reshape(
                        rows,
                        cols,
                    )
                )

                ax.set_title(
                    title
                )

                ax.axis(
                    "off"
                )

            fig.suptitle(
                "E33 — latent intervention visual inspection"
            )

            fig.tight_layout()

            fig.savefig(
                OUT /
                f"e33_visual_{case['direction']}.png",
                dpi=160,
            )

            plt.close(fig)

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    print()
    print("=" * 90)
    print("E33 COMPLETE")
    print("=" * 90)
    print()

    print(
        "Small-step prediction ratio:",
        summary["global"][
            "small_step_prediction_ratio"
        ],
    )

    print(
        "Small-step direction alignment:",
        summary["global"][
            "small_step_alignment"
        ],
    )

    print()

    print(
        "High / low actual movement ratios:"
    )

    for step in STEPS:
        ratio = summary[
            "global"
        ][
            "high_to_low_actual_gain"
        ][
            str(step)
        ]

        print(
            f"  step={step:>4}: "
            f"{ratio:.4f}x"
        )

    print()
    print(
        f"CSV:     {csv_path}"
    )

    print(
        f"SUMMARY: {json_path}"
    )

    print(
        f"OUTPUT:  {OUT}"
    )

    print()
    print(
        "Key question:"
    )

    print(
        "Does the high-sensitivity direction "
        "actually cause more decoded change than "
        "an equal-sized low-sensitivity direction?"
    )

    print(
        "And does the local metric predict that "
        "change accurately at small step sizes?"
    )


if __name__ == "__main__":
    run()
