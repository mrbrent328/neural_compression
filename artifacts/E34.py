#!/usr/bin/env python3
"""
E34 — Known-Factor Semantic Alignment

E31:
    measured decoder geometry

E32:
    projected known-factor latent directions into decoder geometry

E33:
    intervened directly in latent space and verified that the decoder
    metric predicts actual decoder movement

E34:
    closes the loop.

For each held-out synthetic probe:

    known factor
        |
        v
    exact V5 generator
        |
        v
      image
        |
        v
     encoder
        |
        v
      latent
        |
        v
     decoder
        |
        v
 reconstructed image

We compare the reconstructed factor derivative against the TRUE
generator factor derivative.

Question:

    Does the learned autoencoder preserve the semantic direction
    of the known generating factor?

For each factor we measure:

    1. cosine(decoded derivative, true derivative)
    2. magnitude ratio
    3. relative derivative error
    4. latent derivative magnitude
    5. output gain

The interaction factor is especially interesting because V5's
generator defines it as:

    d * a * b * D

so its true local image derivative is:

    a * b * D
"""

from __future__ import annotations

import csv
import json
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# ================================================================
# Paths
# ================================================================

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

V1_PATH = PROJECT_ROOT / "known_factor_synthetic_latent_v1.py"
V5_PATH = HERE / "known_factor_synthetic_latent_v5_vector_field.py"
MODEL_PATH = HERE / "best_model.pt"

OUT = HERE / "synthetic_latent_e34_semantic_alignment"
OUT.mkdir(exist_ok=True)

DELTA = 0.01

FACTORS = [
    "A",
    "B",
    "C",
    "interaction",
]


# ================================================================
# Module loading
# ================================================================

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
            f"Could not load:\n{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


v1 = load_module(
    "e34_v1",
    V1_PATH,
)

v5 = load_module(
    "e34_v5",
    V5_PATH,
)


# ================================================================
# Model
# ================================================================

def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get(
            "state_dict",
            checkpoint,
        ),
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


# ================================================================
# Exact V5 dataset
# ================================================================

def load_exact_dataset():

    (
        standardized,
        factors,
        patterns,
        mean,
        std,
    ) = v5.generate_exact_v1_dataset()

    split = int(
        len(factors) * 0.8
    )

    heldout = factors[
        split:
    ]

    return (
        heldout,
        patterns,
        mean,
        std,
    )


# ================================================================
# Exact generator
# ================================================================

def generate_raw(
    factors,
    patterns,
):

    A, B, C, D = patterns

    return v5.generate_from_factors(
        factors[:, 0],
        factors[:, 1],
        factors[:, 2],
        factors[:, 3],
        A,
        B,
        C,
        D,
    )


def standardize(
    raw,
    mean,
    std,
):

    return (
        (raw - mean) / std
    ).flatten(1)


# ================================================================
# Model evaluation
# ================================================================

@torch.no_grad()
def encode_decode(
    model,
    x,
):

    z = model.encode(x)
    y = model.decode(z)

    return (
        z.cpu().numpy(),
        y.cpu().numpy(),
    )


# ================================================================
# Math
# ================================================================

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


# ================================================================
# True generator derivative
# ================================================================

def true_factor_derivative(
    factors,
    patterns,
    factor_index,
):

    """
    Exact derivative of the underlying synthetic generator.

    Generator:

        a*A
      + b*B
      + c*C
      + d*a*b*D

    Therefore:

        d/dA = A
        d/dB = B
        d/dC = C
        d/dd = a*b*D
    """

    A, B, C, D = patterns

    n = factors.shape[0]

    if factor_index == 0:

        derivative = (
            A
            .unsqueeze(0)
            .expand(n, -1, -1)
        )

    elif factor_index == 1:

        derivative = (
            B
            .unsqueeze(0)
            .expand(n, -1, -1)
        )

    elif factor_index == 2:

        derivative = (
            C
            .unsqueeze(0)
            .expand(n, -1, -1)
        )

    elif factor_index == 3:

        a = factors[:, 0].view(
            -1,
            1,
            1,
        )

        b = factors[:, 1].view(
            -1,
            1,
            1,
        )

        derivative = (
            a
            * b
            * D.unsqueeze(0)
        )

    else:
        raise ValueError(
            factor_index
        )

    return derivative.reshape(
        n,
        -1,
    ).cpu().numpy()


# ================================================================
# Main
# ================================================================

def main():

    print("=" * 90)
    print(
        "E34 — KNOWN-FACTOR SEMANTIC ALIGNMENT"
    )
    print("=" * 90)
    print()

    print(
        f"V1:    {V1_PATH}"
    )

    print(
        f"V5:    {V5_PATH}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print()

    model = load_model()

    (
        factors,
        patterns,
        mean,
        std,
    ) = load_exact_dataset()

    print(
        f"Held-out probes: {len(factors)}"
    )

    print(
        f"Finite difference delta: {DELTA}"
    )

    print()

    results = []

    # ============================================================
    # Process each probe
    # ============================================================

    for i in range(
        len(factors)
    ):

        base = factors[
            i:i + 1
        ]

        # --------------------------------------------------------
        # Base reconstruction
        # --------------------------------------------------------

        base_raw = generate_raw(
            base,
            patterns,
        )

        base_x = standardize(
            base_raw,
            mean,
            std,
        )

        (
            base_z,
            base_y,
        ) = encode_decode(
            model,
            base_x,
        )

        base_z = base_z[0]
        base_y = base_y[0]

        # --------------------------------------------------------
        # Factors
        # --------------------------------------------------------

        row = {
            "probe": i,

            "A": float(
                base[0, 0]
            ),

            "B": float(
                base[0, 1]
            ),

            "C": float(
                base[0, 2]
            ),

            "interaction": float(
                base[0, 3]
            ),
        }

        for factor_index, factor in enumerate(
            FACTORS
        ):

            # ----------------------------------------------------
            # True generator derivative
            # ----------------------------------------------------

            true_derivative = (
                true_factor_derivative(
                    base,
                    patterns,
                    factor_index,
                )[0]
                / float(std)
            )

            # ----------------------------------------------------
            # Actual factor intervention
            # ----------------------------------------------------

            plus = base.clone()
            minus = base.clone()

            plus[
                0,
                factor_index
            ] += DELTA

            minus[
                0,
                factor_index
            ] -= DELTA

            plus_raw = generate_raw(
                plus,
                patterns,
            )

            minus_raw = generate_raw(
                minus,
                patterns,
            )

            plus_x = standardize(
                plus_raw,
                mean,
                std,
            )

            minus_x = standardize(
                minus_raw,
                mean,
                std,
            )

            (
                z_plus,
                y_plus,
            ) = encode_decode(
                model,
                plus_x,
            )

            (
                z_minus,
                y_minus,
            ) = encode_decode(
                model,
                minus_x,
            )

            # ----------------------------------------------------
            # Learned latent derivative
            # ----------------------------------------------------

            dz = (
                z_plus[0]
                - z_minus[0]
            ) / (
                2.0 * DELTA
            )

            # ----------------------------------------------------
            # Decoded output derivative
            # ----------------------------------------------------

            decoded_derivative = (
                y_plus[0]
                - y_minus[0]
            ) / (
                2.0 * DELTA
            )

            # ----------------------------------------------------
            # Compare learned reconstruction derivative
            # against the true generating derivative.
            # ----------------------------------------------------

            true_mag = norm(
                true_derivative
            )

            decoded_mag = norm(
                decoded_derivative
            )

            c = cosine(
                decoded_derivative,
                true_derivative,
            )

            magnitude_ratio = (
                decoded_mag
                / max(
                    true_mag,
                    1e-12,
                )
            )

            relative_error = (
                norm(
                    decoded_derivative
                    - true_derivative
                )
                / max(
                    true_mag,
                    1e-12,
                )
            )

            latent_mag = norm(
                dz
            )

            output_per_latent = (
                decoded_mag
                / max(
                    latent_mag,
                    1e-12,
                )
            )

            # ----------------------------------------------------
            # Store
            # ----------------------------------------------------

            row[
                f"{factor}_semantic_cosine"
            ] = c

            row[
                f"{factor}_magnitude_ratio"
            ] = magnitude_ratio

            row[
                f"{factor}_relative_error"
            ] = relative_error

            row[
                f"{factor}_latent_magnitude"
            ] = latent_mag

            row[
                f"{factor}_decoded_magnitude"
            ] = decoded_mag

            row[
                f"{factor}_output_per_latent"
            ] = output_per_latent

        results.append(
            row
        )

        if (
            i == 0
            or (i + 1) % 100 == 0
        ):

            print(
                f"Processed "
                f"{i + 1}/"
                f"{len(factors)}"
            )

    # ============================================================
    # CSV
    # ============================================================

    csv_path = (
        OUT /
        "e34_semantic_alignment.csv"
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
        writer.writerows(
            results
        )

    # ============================================================
    # Summary
    # ============================================================

    summary = {
        "experiment":
            "E34 — known-factor semantic alignment",

        "probe_count":
            len(results),

        "delta":
            DELTA,

        "generator":
            "V5 exact generator",

        "factors":
            {},
    }

    for factor in FACTORS:

        cosine_values = np.asarray(
            [
                r[
                    f"{factor}_semantic_cosine"
                ]
                for r in results
            ],
            dtype=float,
        )

        ratio_values = np.asarray(
            [
                r[
                    f"{factor}_magnitude_ratio"
                ]
                for r in results
            ],
            dtype=float,
        )

        error_values = np.asarray(
            [
                r[
                    f"{factor}_relative_error"
                ]
                for r in results
            ],
            dtype=float,
        )

        latent_values = np.asarray(
            [
                r[
                    f"{factor}_latent_magnitude"
                ]
                for r in results
            ],
            dtype=float,
        )

        summary[
            "factors"
        ][factor] = {

            "semantic_cosine_median":
                float(
                    np.median(
                        cosine_values
                    )
                ),

            "semantic_cosine_p10":
                float(
                    np.percentile(
                        cosine_values,
                        10,
                    )
                ),

            "semantic_cosine_p90":
                float(
                    np.percentile(
                        cosine_values,
                        90,
                    )
                ),

            "magnitude_ratio_median":
                float(
                    np.median(
                        ratio_values
                    )
                ),

            "relative_error_median":
                float(
                    np.median(
                        error_values
                    )
                ),

            "latent_magnitude_median":
                float(
                    np.median(
                        latent_values
                    )
                ),

            "fraction_cosine_above_0.99":
                float(
                    np.mean(
                        cosine_values
                        > 0.99
                    )
                ),

            "fraction_cosine_above_0.95":
                float(
                    np.mean(
                        cosine_values
                        > 0.95
                    )
                ),
        }

    summary_path = (
        OUT /
        "e34_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # ============================================================
    # Plot 1 — semantic alignment
    # ============================================================

    plt.figure(
        figsize=(9, 7)
    )

    for factor in FACTORS:

        values = np.asarray(
            [
                r[
                    f"{factor}_semantic_cosine"
                ]
                for r in results
            ]
        )

        plt.scatter(
            np.arange(
                len(values)
            ),
            values,
            s=12,
            alpha=0.4,
            label=factor,
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        0.95,
        linestyle=":",
        linewidth=1,
    )

    plt.xlabel(
        "held-out probe"
    )

    plt.ylabel(
        "cosine(decoded derivative, true derivative)"
    )

    plt.title(
        "E34 — Semantic preservation of known factor directions"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e34_semantic_alignment.png",
        dpi=160,
    )

    plt.close()

    # ============================================================
    # Plot 2 — magnitude preservation
    # ============================================================

    plt.figure(
        figsize=(9, 7)
    )

    for factor in FACTORS:

        values = np.asarray(
            [
                r[
                    f"{factor}_magnitude_ratio"
                ]
                for r in results
            ]
        )

        plt.scatter(
            np.arange(
                len(values)
            ),
            values,
            s=12,
            alpha=0.4,
            label=factor,
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "held-out probe"
    )

    plt.ylabel(
        "decoded derivative magnitude / true magnitude"
    )

    plt.title(
        "E34 — Magnitude preservation"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e34_magnitude_ratio.png",
        dpi=160,
    )

    plt.close()

    # ============================================================
    # Plot 3 — summary distributions
    # ============================================================

    plt.figure(
        figsize=(10, 7)
    )

    data = []

    for factor in FACTORS:

        data.append(
            [
                r[
                    f"{factor}_semantic_cosine"
                ]
                for r in results
            ]
        )

    plt.boxplot(
        data,
        label=FACTORS,
        showfliers=False,
    )

    plt.ylabel(
        "semantic cosine"
    )

    plt.title(
        "E34 — Distribution of semantic alignment"
    )

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e34_semantic_boxplot.png",
        dpi=160,
    )

    plt.close()

    # ============================================================
    # Console summary
    # ============================================================

    print()
    print("=" * 90)
    print("E34 COMPLETE")
    print("=" * 90)
    print()

    for factor in FACTORS:

        s = summary[
            "factors"
        ][factor]

        print(
            f"{factor:12s} "
            f"cos={s['semantic_cosine_median']:.6f} "
            f"ratio={s['magnitude_ratio_median']:.6f} "
            f"error={s['relative_error_median']:.6f}"
        )

    print()

    print(
        f"CSV:     {csv_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Plots:   {OUT}"
    )


if __name__ == "__main__":
    main()
