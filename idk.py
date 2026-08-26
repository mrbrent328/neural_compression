#!/usr/bin/env python3
"""
E33 — Decoder Geometry Intervention

Direct follow-up to E32.

E32:
    - takes V5's measured latent factor-response vectors
    - measures them using G = J^T J

E33:
    - actually moves the latent along those vectors
    - decodes the perturbed latent
    - compares the actual output movement to J @ dz

Questions:

1. Does the local decoder metric predict actual decoder movement?
2. How quickly does the first-order approximation break down?
3. Do A/B/C/interaction factor directions behave differently?
4. Does decoder-effective sensitivity predict actual sensitivity?

No new model.
No new training.
No new probe distribution.

Everything comes directly from the existing V1/V5/E32 pipeline.
"""

from pathlib import Path
import csv
import json
import importlib.util
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

V1_PATH = ROOT.parent / "known_factor_synthetic_latent_v1.py"
V5_PATH = ROOT / "known_factor_synthetic_latent_v5_vector_field.py"
MODEL_PATH = ROOT / "best_model.pt"
V5_CSV = ROOT / "synthetic_latent_v5" / "probe_vector_geometry.csv"

OUT = ROOT / "synthetic_latent_e33_geometry_intervention"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DELTA = 0.01

FACTORS = [
    "A",
    "B",
    "C",
    "interaction",
]

VECTOR_COLUMNS = {
    "A": ("dA_z0", "dA_z1"),
    "B": ("dB_z0", "dB_z1"),
    "C": ("dC_z0", "dC_z1"),
    "interaction": (
        "dinteraction_z0",
        "dinteraction_z1",
    ),
}

# These are MULTIPLIERS of the V5 local derivative vector.
#
# The first values are genuinely local.
# Larger values test where the linear approximation stops working.
STEP_MULTIPLIERS = [
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
]


# ---------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------

def load_module(name, path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module:\n{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------
# Load V1 / V5
# ---------------------------------------------------------------------

v1 = load_module(
    "e33_v1",
    V1_PATH,
)

v5 = load_module(
    "e33_v5",
    V5_PATH,
)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get(
                "state_dict",
                checkpoint,
            ),
        )
    else:
        raise RuntimeError(
            "best_model.pt is not a dictionary checkpoint."
        )

    model = v1.TinyAutoencoder(
        v1.ROWS * v1.COLS,
        2,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return model


# ---------------------------------------------------------------------
# V5 CSV
# ---------------------------------------------------------------------

def load_v5_rows():

    if not V5_CSV.exists():
        raise FileNotFoundError(
            f"Missing V5 CSV:\n{V5_CSV}"
        )

    required = [
        "probe",
        "A",
        "B",
        "C",
        "interaction",
        "dA_z0",
        "dA_z1",
        "dB_z0",
        "dB_z1",
        "dC_z0",
        "dC_z1",
        "dinteraction_z0",
        "dinteraction_z1",
    ]

    rows = []

    with V5_CSV.open(
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        missing = [
            x for x in required
            if x not in reader.fieldnames
        ]

        if missing:
            raise RuntimeError(
                f"V5 CSV missing columns: {missing}"
            )

        for raw in reader:

            row = {
                k: raw[k]
                for k in required
            }

            row["probe"] = int(
                row["probe"]
            )

            for k in required:
                if k != "probe":
                    row[k] = float(
                        row[k]
                    )

            rows.append(row)

    return rows


# ---------------------------------------------------------------------
# Exact V5 latent recovery
# ---------------------------------------------------------------------

def recover_v5_latents():

    required = [
        "generate_exact_v1_dataset",
        "encode_factors",
    ]

    for name in required:

        if not hasattr(
            v5,
            name,
        ):
            raise RuntimeError(
                f"V5 does not expose {name}()"
            )

    (
        _standardized,
        factors,
        patterns,
        norm_mean,
        norm_std,
    ) = v5.generate_exact_v1_dataset()

    split = int(
        len(factors) * 0.8
    )

    heldout = factors[
        split:
    ]

    rows = load_v5_rows()

    if len(rows) != len(heldout):

        raise RuntimeError(
            "Probe count mismatch.\n"
            f"V5 CSV:      {len(rows)}\n"
            f"Held-out:    {len(heldout)}\n"
            "Refusing to silently misalign probes."
        )

    model = load_model()

    device = getattr(
        v5,
        "device",
        torch.device("cpu"),
    )

    try:

        z = v5.encode_factors(
            model,
            device,
            heldout,
            patterns,
            norm_mean,
            norm_std,
        )

    except Exception:

        z = v5.encode_factors(
            model,
            torch.device("cpu"),
            heldout,
            patterns,
            norm_mean,
            norm_std,
        )

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    if z.ndim != 2 or z.shape[1] != 2:

        raise RuntimeError(
            f"Unexpected latent shape: {z.shape}"
        )

    return (
        model,
        rows,
        z,
    )


# ---------------------------------------------------------------------
# Decoder Jacobian
# ---------------------------------------------------------------------

@torch.no_grad()
def decoder_jacobian(
    model,
    z,
):

    columns = []

    for axis in range(2):

        zp = z.clone()
        zm = z.clone()

        zp[:, axis] += DELTA
        zm[:, axis] -= DELTA

        yp = model.decode(
            zp
        )

        ym = model.decode(
            zm
        )

        derivative = (
            yp - ym
        ) / (
            2.0 * DELTA
        )

        columns.append(
            derivative
            .reshape(-1)
            .cpu()
            .numpy()
        )

    return np.stack(
        columns,
        axis=1,
    )


# ---------------------------------------------------------------------
# Basic math
# ---------------------------------------------------------------------

def norm(x):
    return float(
        np.linalg.norm(x)
    )


def cosine(a, b):

    na = norm(a)
    nb = norm(b)

    if (
        na < 1e-12
        or nb < 1e-12
    ):
        return np.nan

    return float(
        np.dot(a, b)
        / (na * nb)
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("=" * 90)
    print(
        "E33 — DECODER GEOMETRY INTERVENTION"
    )
    print("=" * 90)
    print()

    print(
        "Building directly from V1 + V5 + E32."
    )

    print(
        f"V1: {V1_PATH}"
    )

    print(
        f"V5: {V5_PATH}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"V5 CSV: {V5_CSV}"
    )

    print()

    model, rows, z_all = (
        recover_v5_latents()
    )

    print(
        f"Recovered {len(rows)} "
        f"exact V5 latent points."
    )

    print()

    results = []

    # -----------------------------------------------------------------
    # Every held-out V5 probe.
    # -----------------------------------------------------------------

    for probe_index, row in enumerate(
        rows
    ):

        z_np = z_all[
            probe_index
        ]

        z = torch.tensor(
            z_np[None, :],
            dtype=torch.float32,
        )

        # -------------------------------------------------------------
        # Decoder local Jacobian.
        # -------------------------------------------------------------

        J = decoder_jacobian(
            model,
            z,
        )

        G = (
            J.T @ J
        )

        # -------------------------------------------------------------
        # Baseline decoded output.
        # -------------------------------------------------------------

        with torch.no_grad():

            baseline = (
                model.decode(z)
                .reshape(-1)
                .cpu()
                .numpy()
            )

        # -------------------------------------------------------------
        # Each known factor direction.
        # -------------------------------------------------------------

        for factor in FACTORS:

            c0, c1 = (
                VECTOR_COLUMNS[
                    factor
                ]
            )

            v = np.array(
                [
                    row[c0],
                    row[c1],
                ],
                dtype=np.float64,
            )

            v_norm = norm(v)

            # E32's decoder-effective infinitesimal magnitude.
            metric_magnitude = float(
                np.sqrt(
                    max(
                        v @ G @ v,
                        0.0,
                    )
                )
            )

            # First-order output vector Jv.
            Jv = (
                J @ v
            )

            Jv_norm = norm(
                Jv
            )

            # These mathematically should match.
            metric_jacobian_ratio = (
                metric_magnitude
                / max(
                    Jv_norm,
                    1e-12,
                )
            )

            for multiplier in (
                STEP_MULTIPLIERS
            ):

                dz = (
                    multiplier
                    * v
                )

                z_plus = (
                    z_np
                    + dz
                )

                z_minus = (
                    z_np
                    - dz
                )

                with torch.no_grad():

                    y_plus = (
                        model.decode(
                            torch.tensor(
                                z_plus[None, :],
                                dtype=torch.float32,
                            )
                        )
                        .reshape(-1)
                        .cpu()
                        .numpy()
                    )

                    y_minus = (
                        model.decode(
                            torch.tensor(
                                z_minus[None, :],
                                dtype=torch.float32,
                            )
                        )
                        .reshape(-1)
                        .cpu()
                        .numpy()
                    )

                # -----------------------------------------------------
                # Actual output movements.
                # -----------------------------------------------------

                plus_delta = (
                    y_plus
                    - baseline
                )

                minus_delta = (
                    y_minus
                    - baseline
                )

                actual_plus = norm(
                    plus_delta
                )

                actual_minus = norm(
                    minus_delta
                )

                # -----------------------------------------------------
                # First-order prediction.
                # -----------------------------------------------------

                predicted = (
                    multiplier
                    * Jv_norm
                )

                # -----------------------------------------------------
                # Symmetric central displacement.
                #
                # For a locally linear decoder:
                #
                # 0.5 ||D(z+dv)-D(z-dv)||
                #
                # should approximately equal
                #
                # ||J dv||
                #
                # -----------------------------------------------------

                symmetric_actual = (
                    0.5
                    * norm(
                        y_plus
                        - y_minus
                    )
                )

                # -----------------------------------------------------
                # Prediction ratios.
                # -----------------------------------------------------

                plus_ratio = (
                    actual_plus
                    / max(
                        predicted,
                        1e-12,
                    )
                )

                minus_ratio = (
                    actual_minus
                    / max(
                        predicted,
                        1e-12,
                    )
                )

                symmetric_ratio = (
                    symmetric_actual
                    / max(
                        predicted,
                        1e-12,
                    )
                )

                # -----------------------------------------------------
                # Directional agreement.
                # -----------------------------------------------------

                plus_alignment = cosine(
                    plus_delta,
                    Jv,
                )

                minus_alignment = cosine(
                    -minus_delta,
                    Jv,
                )

                # -----------------------------------------------------
                # Store.
                # -----------------------------------------------------

                results.append({

                    "probe":
                        probe_index,

                    "factor":
                        factor,

                    "A":
                        row["A"],

                    "B":
                        row["B"],

                    "C":
                        row["C"],

                    "interaction":
                        row["interaction"],

                    "z0":
                        float(z_np[0]),

                    "z1":
                        float(z_np[1]),

                    "multiplier":
                        float(multiplier),

                    "latent_step_magnitude":
                        float(
                            norm(dz)
                        ),

                    "euclidean_factor_derivative":
                        float(v_norm),

                    "decoder_metric_derivative":
                        float(
                            metric_magnitude
                        ),

                    "Jv_magnitude":
                        float(Jv_norm),

                    "metric_Jv_ratio":
                        float(
                            metric_jacobian_ratio
                        ),

                    "predicted_output_movement":
                        float(predicted),

                    "actual_plus_movement":
                        float(actual_plus),

                    "actual_minus_movement":
                        float(actual_minus),

                    "symmetric_actual_movement":
                        float(
                            symmetric_actual
                        ),

                    "plus_prediction_ratio":
                        float(
                            plus_ratio
                        ),

                    "minus_prediction_ratio":
                        float(
                            minus_ratio
                        ),

                    "symmetric_prediction_ratio":
                        float(
                            symmetric_ratio
                        ),

                    "plus_direction_alignment":
                        float(
                            plus_alignment
                        ),

                    "minus_direction_alignment":
                        float(
                            minus_alignment
                        ),
                })

        if (
            probe_index == 0
            or (probe_index + 1) % 100 == 0
        ):

            print(
                f"Processed "
                f"{probe_index + 1}/"
                f"{len(rows)}"
            )

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------

    csv_path = (
        OUT /
        "e33_intervention_results.csv"
    )

    fields = list(
        results[0].keys()
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------------------
    # Summary.
    # -----------------------------------------------------------------

    def subset(
        factor,
        multiplier,
        field,
    ):

        return np.asarray(
            [
                r[field]
                for r in results
                if (
                    r["factor"]
                    == factor
                    and abs(
                        r["multiplier"]
                        - multiplier
                    )
                    < 1e-12
                )
            ],
            dtype=float,
        )

    summary = {
        "experiment":
            "E33 — decoder geometry intervention",

        "source":
            "V1 checkpoint + V5 probe geometry + E32 decoder metric",

        "delta":
            DELTA,

        "probe_count":
            len(rows),

        "step_multipliers":
            STEP_MULTIPLIERS,

        "factor_summary":
            {},

        "metric_sanity":
            {},
    }

    # ---------------------------------------------------------------
    # Metric sanity.
    # ---------------------------------------------------------------

    metric_ratios = np.asarray(
        [
            r["metric_Jv_ratio"]
            for r in results
        ],
        dtype=float,
    )

    summary["metric_sanity"] = {
        "median":
            float(
                np.median(
                    metric_ratios
                )
            ),

        "max_abs_error":
            float(
                np.max(
                    np.abs(
                        metric_ratios
                        - 1.0
                    )
                )
            ),
    }

    # ---------------------------------------------------------------
    # Per-factor summary.
    # ---------------------------------------------------------------

    for factor in FACTORS:

        summary[
            "factor_summary"
        ][factor] = {}

        for multiplier in (
            STEP_MULTIPLIERS
        ):

            ratios = subset(
                factor,
                multiplier,
                "symmetric_prediction_ratio",
            )

            align = subset(
                factor,
                multiplier,
                "plus_direction_alignment",
            )

            actual = subset(
                factor,
                multiplier,
                "symmetric_actual_movement",
            )

            predicted = subset(
                factor,
                multiplier,
                "predicted_output_movement",
            )

            summary[
                "factor_summary"
            ][factor][
                str(multiplier)
            ] = {

                "predicted_median":
                    float(
                        np.median(
                            predicted
                        )
                    ),

                "actual_median":
                    float(
                        np.median(
                            actual
                        )
                    ),

                "prediction_ratio_median":
                    float(
                        np.median(
                            ratios
                        )
                    ),

                "prediction_ratio_p10":
                    float(
                        np.percentile(
                            ratios,
                            10,
                        )
                    ),

                "prediction_ratio_p90":
                    float(
                        np.percentile(
                            ratios,
                            90,
                        )
                    ),

                "direction_alignment_median":
                    float(
                        np.median(
                            align
                        )
                    ),
            }

    summary_path = (
        OUT /
        "e33_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # -----------------------------------------------------------------
    # Plot 1 — actual / predicted vs step.
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    for factor in FACTORS:

        medians = []

        for multiplier in (
            STEP_MULTIPLIERS
        ):

            vals = subset(
                factor,
                multiplier,
                "symmetric_prediction_ratio",
            )

            medians.append(
                np.median(vals)
            )

        plt.plot(
            STEP_MULTIPLIERS,
            medians,
            marker="o",
            label=factor,
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "factor-vector multiplier"
    )

    plt.ylabel(
        "actual / predicted decoder movement"
    )

    plt.title(
        "E33 — First-order decoder metric validity"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e33_prediction_validity.png",
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 2 — direction alignment.
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    for factor in FACTORS:

        medians = []

        for multiplier in (
            STEP_MULTIPLIERS
        ):

            vals = subset(
                factor,
                multiplier,
                "plus_direction_alignment",
            )

            medians.append(
                np.median(vals)
            )

        plt.plot(
            STEP_MULTIPLIERS,
            medians,
            marker="o",
            label=factor,
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "factor-vector multiplier"
    )

    plt.ylabel(
        "cosine(actual movement, Jv)"
    )

    plt.title(
        "E33 — Actual decoder direction vs predicted direction"
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
    # Plot 3 — actual vs predicted.
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    for factor in FACTORS:

        actual = []
        predicted = []

        for multiplier in (
            STEP_MULTIPLIERS
        ):

            actual.extend(
                subset(
                    factor,
                    multiplier,
                    "symmetric_actual_movement",
                )
            )

            predicted.extend(
                subset(
                    factor,
                    multiplier,
                    "predicted_output_movement",
                )
            )

        plt.scatter(
            predicted,
            actual,
            s=10,
            alpha=0.35,
            label=factor,
        )

    all_vals = []

    for r in results:

        all_vals.extend([
            r["predicted_output_movement"],
            r["symmetric_actual_movement"],
        ])

    lo = min(all_vals)
    hi = max(all_vals)

    plt.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "predicted output movement"
    )

    plt.ylabel(
        "actual output movement"
    )

    plt.title(
        "E33 — Metric prediction vs actual decoder movement"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e33_actual_vs_predicted.png",
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Console summary.
    # -----------------------------------------------------------------

    print()
    print("=" * 90)
    print("E33 COMPLETE")
    print("=" * 90)
    print()

    print(
        "Metric sanity:"
    )

    print(
        f"  median metric / ||Jv|| = "
        f"{summary['metric_sanity']['median']:.8f}"
    )

    print(
        f"  max error from 1 = "
        f"{summary['metric_sanity']['max_abs_error']:.3e}"
    )

    print()

    print(
        "Smallest-step prediction:"
    )

    for factor in FACTORS:

        r = summary[
            "factor_summary"
        ][factor][
            str(
                STEP_MULTIPLIERS[0]
            )
        ]

        print(
            f"  {factor:12s} "
            f"ratio={r['prediction_ratio_median']:.5f} "
            f"alignment={r['direction_alignment_median']:.5f}"
        )

    print()

    print(
        f"Results: {OUT}"
    )

    print(
        f"CSV:     {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
