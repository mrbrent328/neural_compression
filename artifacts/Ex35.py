#!/usr/bin/env python3
"""
E35 — Physical Rotation Manifold Probe

E29 rotated the LATENT COORDINATE SYSTEM.

This experiment is different:

    rotate the actual generated image
        ->
    encode it
        ->
    observe the latent trajectory.

Question:

    Does a continuous physical transformation produce a
    continuous / coherent trajectory through learned latent space?

We use the exact V5 synthetic generator and the real V1 checkpoint.

For each selected held-out synthetic object:

    image(theta)
        ->
    encoder
        ->
    z(theta)

We measure:

    - latent position
    - consecutive latent displacement
    - local directional consistency
    - distance from starting point
    - whether the trajectory closes after 360 degrees

No training.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

V1_PATH = PROJECT_ROOT / "known_factor_synthetic_latent_v1.py"
V5_PATH = HERE / "known_factor_synthetic_latent_v5_vector_field.py"
MODEL_PATH = HERE / "best_model.pt"

OUT = HERE / "synthetic_latent_e35_rotation_manifold"
OUT.mkdir(exist_ok=True)


# ============================================================
# Experiment settings
# ============================================================

# Full rotation.
ANGLES = list(range(0, 360, 15))

# A manageable number of representative held-out objects.
# We use the same held-out distribution as V5.
N_OBJECTS = 12

# Image rotation uses bilinear interpolation.
# align_corners=False is intentional and fixed.
INTERPOLATION = "bilinear"


# ============================================================
# Module loading
# ============================================================

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

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


v1 = load_module(
    "e35_v1",
    V1_PATH,
)

v5 = load_module(
    "e35_v5",
    V5_PATH,
)


# ============================================================
# Model
# ============================================================

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
            "best_model.pt is not a checkpoint dictionary."
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


# ============================================================
# Exact V5 data
# ============================================================

def load_exact_data():

    (
        tensors,
        factors,
        patterns,
        mean,
        std,
    ) = v5.generate_exact_v1_dataset()

    split = int(
        len(factors) * 0.8
    )

    heldout_factors = factors[
        split:
    ]

    return (
        heldout_factors,
        patterns,
        mean,
        std,
    )


def generate_images(
    factors,
    patterns,
):

    A, B, C, D = patterns

    a = factors[:, 0].view(-1, 1, 1)
    b = factors[:, 1].view(-1, 1, 1)
    c = factors[:, 2].view(-1, 1, 1)
    d = factors[:, 3].view(-1, 1, 1)

    return (
        a * A
        + b * B
        + c * C
        + d * a * b * D
    )


# ============================================================
# Image rotation
# ============================================================

def rotate_images(
    images,
    degrees,
):

    """
    images:
        [N, H, W]

    Returns:
        [N, H, W]
    """

    n, h, w = images.shape

    x = images.unsqueeze(1)

    theta = math.radians(
        degrees
    )

    c = math.cos(theta)
    s = math.sin(theta)

    # Affine matrix mapping output coordinates back into input.
    matrix = torch.tensor(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
        ],
        dtype=torch.float32,
    )

    matrix = matrix.unsqueeze(0).repeat(
        n,
        1,
        1,
    )

    grid = F.affine_grid(
        matrix,
        x.size(),
        align_corners=False,
    )

    rotated = F.grid_sample(
        x,
        grid,
        mode=INTERPOLATION,
        padding_mode="zeros",
        align_corners=False,
    )

    return rotated[:, 0]


# ============================================================
# Encoding
# ============================================================

@torch.no_grad()
def encode_images(
    model,
    images,
    mean,
    std,
):

    x = (
        (images - mean)
        / std
    ).flatten(1)

    z = model.encode(x)

    return z.cpu().numpy()


# ============================================================
# Geometry
# ============================================================

def norm(v):

    return float(
        np.linalg.norm(v)
    )


def cosine(a, b):

    na = norm(a)
    nb = norm(b)

    if na < 1e-12 or nb < 1e-12:
        return np.nan

    return float(
        np.dot(a, b)
        / (na * nb)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 88)
    print(
        "E35 — PHYSICAL ROTATION MANIFOLD PROBE"
    )
    print("=" * 88)
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
    ) = load_exact_data()

    # --------------------------------------------------------
    # Choose representative objects.
    #
    # Deterministic spread through held-out set.
    # --------------------------------------------------------

    if len(factors) < N_OBJECTS:
        raise RuntimeError(
            "Not enough held-out probes."
        )

    indices = np.linspace(
        0,
        len(factors) - 1,
        N_OBJECTS,
        dtype=int,
    )

    selected = factors[
        indices
    ]

    images = generate_images(
        selected,
        patterns,
    )

    print(
        f"Objects: {len(selected)}"
    )

    print(
        f"Angles:  {len(ANGLES)}"
    )

    print()

    all_results = []

    trajectories = {}

    # ========================================================
    # Process objects
    # ========================================================

    for object_id in range(
        len(selected)
    ):

        base_image = images[
            object_id:
            object_id + 1
        ]

        zs = []

        # ----------------------------------------------------
        # Sweep angle
        # ----------------------------------------------------

        for angle in ANGLES:

            rotated = rotate_images(
                base_image,
                angle,
            )

            z = encode_images(
                model,
                rotated,
                mean,
                std,
            )[0]

            zs.append(z)

        zs = np.asarray(
            zs,
            dtype=float,
        )

        trajectories[
            str(object_id)
        ] = zs.tolist()

        # ----------------------------------------------------
        # Consecutive displacement
        # ----------------------------------------------------

        displacements = (
            np.diff(
                zs,
                axis=0,
            )
        )

        displacement_magnitudes = np.linalg.norm(
            displacements,
            axis=1,
        )

        # Direction consistency.
        direction_cosines = []

        for i in range(
            len(displacements) - 1
        ):

            direction_cosines.append(
                cosine(
                    displacements[i],
                    displacements[i + 1],
                )
            )

        # ----------------------------------------------------
        # Relative to starting point
        # ----------------------------------------------------

        relative = (
            zs
            - zs[0]
        )

        radius = np.linalg.norm(
            relative,
            axis=1,
        )

        # ----------------------------------------------------
        # Closure after 360°
        #
        # Since the final angle is 345°, compare 345° to 0°.
        # The missing 360° point is generated explicitly.
        # ----------------------------------------------------

        final_rotated = rotate_images(
            base_image,
            360,
        )

        z360 = encode_images(
            model,
            final_rotated,
            mean,
            std,
        )[0]

        closure_error = norm(
            z360
            - zs[0]
        )

        # ----------------------------------------------------
        # Full-cycle displacement
        # ----------------------------------------------------

        full_cycle = (
            z360
            - zs[0]
        )

        full_cycle_mag = norm(
            full_cycle
        )

        # ----------------------------------------------------
        # Record summary
        # ----------------------------------------------------

        result = {

            "object_id":
                object_id,

            "source_probe":
                int(indices[object_id]),

            "A":
                float(
                    selected[
                        object_id,
                        0,
                    ]
                ),

            "B":
                float(
                    selected[
                        object_id,
                        1,
                    ]
                ),

            "C":
                float(
                    selected[
                        object_id,
                        2,
                    ]
                ),

            "interaction":
                float(
                    selected[
                        object_id,
                        3,
                    ]
                ),

            "mean_step_magnitude":
                float(
                    np.mean(
                        displacement_magnitudes
                    )
                ),

            "median_step_magnitude":
                float(
                    np.median(
                        displacement_magnitudes
                    )
                ),

            "mean_direction_cosine":
                float(
                    np.nanmean(
                        direction_cosines
                    )
                ),

            "median_direction_cosine":
                float(
                    np.nanmedian(
                        direction_cosines
                    )
                ),

            "minimum_direction_cosine":
                float(
                    np.nanmin(
                        direction_cosines
                    )
                ),

            "maximum_radius":
                float(
                    np.max(radius)
                ),

            "mean_radius":
                float(
                    np.mean(radius)
                ),

            "closure_error":
                closure_error,

            "full_cycle_magnitude":
                full_cycle_mag,
        }

        all_results.append(
            result
        )

        print(
            f"Object {object_id:2d} | "
            f"mean step="
            f"{result['mean_step_magnitude']:.5f} | "
            f"direction cos="
            f"{result['median_direction_cosine']:.5f} | "
            f"closure="
            f"{closure_error:.5f}"
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "experiment":
            "E35 physical rotation manifold probe",

        "objects":
            len(selected),

        "angles":
            ANGLES,

        "interpolation":
            INTERPOLATION,

        "results":
            all_results,

        "aggregate": {

            "median_direction_cosine":
                float(
                    np.median(
                        [
                            r[
                                "median_direction_cosine"
                            ]
                            for r in all_results
                        ]
                    )
                ),

            "median_closure_error":
                float(
                    np.median(
                        [
                            r[
                                "closure_error"
                            ]
                            for r in all_results
                        ]
                    )
                ),

            "median_mean_step":
                float(
                    np.median(
                        [
                            r[
                                "mean_step_magnitude"
                            ]
                            for r in all_results
                        ]
                    )
                ),

            "median_max_radius":
                float(
                    np.median(
                        [
                            r[
                                "maximum_radius"
                            ]
                            for r in all_results
                        ]
                    )
                ),
        },
    }

    (OUT / "e35_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    (OUT / "e35_trajectories.json").write_text(
        json.dumps(
            trajectories,
            indent=2,
        )
    )

    # ========================================================
    # Plot latent trajectories
    # ========================================================

    plt.figure(
        figsize=(9, 8)
    )

    for object_id in range(
        len(selected)
    ):

        z = np.asarray(
            trajectories[
                str(object_id)
            ]
        )

        plt.plot(
            z[:, 0],
            z[:, 1],
            marker="o",
            markersize=3,
            alpha=0.7,
            label=f"object {object_id}",
        )

    plt.xlabel(
        "latent z0"
    )

    plt.ylabel(
        "latent z1"
    )

    plt.title(
        "E35 — Latent trajectories under physical image rotation"
    )

    plt.legend(
        fontsize=7,
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e35_latent_rotation_trajectories.png",
        dpi=160,
    )

    plt.close()

    # ========================================================
    # Plot direction consistency
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    values = [
        r[
            "median_direction_cosine"
        ]
        for r in all_results
    ]

    plt.bar(
        np.arange(
            len(values)
        ),
        values,
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    plt.axhline(
        0.0,
        linestyle=":",
        linewidth=1,
    )

    plt.xlabel(
        "object"
    )

    plt.ylabel(
        "median consecutive direction cosine"
    )

    plt.title(
        "E35 — Local rotational trajectory consistency"
    )

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e35_direction_consistency.png",
        dpi=160,
    )

    plt.close()

    # ========================================================
    # Plot closure
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    closure = [
        r[
            "closure_error"
        ]
        for r in all_results
    ]

    plt.bar(
        np.arange(
            len(closure)
        ),
        closure,
    )

    plt.xlabel(
        "object"
    )

    plt.ylabel(
        "latent distance: 0° → 360°"
    )

    plt.title(
        "E35 — Rotational manifold closure"
    )

    plt.tight_layout()

    plt.savefig(
        OUT /
        "e35_rotation_closure.png",
        dpi=160,
    )

    plt.close()

    # ========================================================
    # Done
    # ========================================================

    print()
    print("=" * 88)
    print("E35 COMPLETE")
    print("=" * 88)

    print(
        f"Median direction cosine: "
        f"{summary['aggregate']['median_direction_cosine']:.6f}"
    )

    print(
        f"Median closure error: "
        f"{summary['aggregate']['median_closure_error']:.6f}"
    )

    print(
        f"Median mean step: "
        f"{summary['aggregate']['median_mean_step']:.6f}"
    )

    print()

    print(
        f"Results: {OUT}"
    )


if __name__ == "__main__":
    main()
