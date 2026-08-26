#!/usr/bin/env python3
"""
V6 — Behavioral Coupling using the REAL V1 model and REAL V5 probe machinery.

Project layout expected:

    latent_model_test/
        known_factor_synthetic_latent_v1.py
        synthetic_latent_v1/
            known_factor_synthetic_latent_v5_vector_field.py
            best_model.pt
            synthetic_latent_v5/
                probe_vector_geometry.csv

This script deliberately loads V1 and V5 by filesystem path so Python package
layout does not matter.

IMPORTANT:
This script does NOT invent a factor->image mapping. It first imports V5 and
tries to discover/reuse its actual probe construction. If V5 exposes only
training/plotting code and not a reusable probe function, the script stops with
a precise message rather than silently measuring the wrong thing.
"""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

V1_PATH = PROJECT_ROOT / "known_factor_synthetic_latent_v1.py"
V5_PATH = HERE / "known_factor_synthetic_latent_v5_vector_field.py"
MODEL_PATH = HERE / "best_model.pt"
V5_CSV = HERE / "synthetic_latent_v5" / "probe_vector_geometry.csv"

OUT_DIR = HERE / "synthetic_latent_v6"
OUT_DIR.mkdir(exist_ok=True)

DELTA = 0.01
FACTORS = ["A", "B", "C", "interaction"]
TOP_PERCENT = 0.10


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_real_model():
    v1 = load_module("latent_v1_real", V1_PATH)

    if not hasattr(v1, "TinyAutoencoder"):
        raise RuntimeError(
            f"{V1_PATH} does not expose TinyAutoencoder"
        )

    model = v1.TinyAutoencoder()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint),
        )
    else:
        raise RuntimeError(
            "best_model.pt is not a state_dict/checkpoint dictionary."
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, v1


def load_v5():
    return load_module("vector_field_v5_real", V5_PATH)


def find_probe_constructor(v5):
    """
    Search V5 for likely reusable probe-construction functions.

    We intentionally inspect names/signatures rather than guessing a mapping.
    """
    candidates = []

    for name, obj in vars(v5).items():
        if not callable(obj):
            continue

        lname = name.lower()

        score = 0

        if "probe" in lname:
            score += 5
        if "construct" in lname:
            score += 4
        if "generate" in lname:
            score += 2
        if "input" in lname:
            score += 2
        if "image" in lname:
            score += 2
        if "sample" in lname:
            score += 1

        if score:
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "<?>"

            candidates.append((score, name, sig, obj))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates


def read_v5_csv():
    if not V5_CSV.exists():
        raise FileNotFoundError(
            f"Could not find V5 probe CSV:\n{V5_CSV}"
        )

    rows = []

    with V5_CSV.open(newline="") as f:
        reader = csv.DictReader(f)

        required = ["probe", "A", "B", "C", "interaction"]
        missing = [
            col for col in required
            if col not in reader.fieldnames
        ]

        if missing:
            raise RuntimeError(
                f"V5 CSV missing columns: {missing}"
            )

        for r in reader:
            rows.append({
                "probe": int(r["probe"]),
                "A": float(r["A"]),
                "B": float(r["B"]),
                "C": float(r["C"]),
                "interaction": float(r["interaction"]),
                "v5_total_sensitivity": float(
                    r.get("total_sensitivity", "nan")
                ),
                "v5_top_10pct": r.get(
                    "top_10pct", "False"
                ).strip().lower() == "true",
            })

    return rows


def discover_v5_globals(v5):
    """
    Record useful constants/functions from V5 so the terminal output tells us
    exactly what machinery is available.
    """
    interesting = []

    for name, obj in vars(v5).items():
        lname = name.lower()

        if any(
            token in lname
            for token in (
                "probe",
                "factor",
                "image",
                "input",
                "delta",
                "grid",
                "row",
                "col",
            )
        ):
            if callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = "<?>"
                interesting.append(
                    f"{name}{sig}"
                )
            elif isinstance(obj, (int, float, str, tuple, list)):
                text = repr(obj)
                if len(text) > 160:
                    text = text[:157] + "..."
                interesting.append(
                    f"{name} = {text}"
                )

    return interesting


def model_forward(model, x):
    with torch.no_grad():
        if hasattr(model, "encode") and hasattr(model, "decode"):
            z = model.encode(x)
            y = model.decode(z)
            return y, z

        out = model(x)

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[0], out[1]

    raise RuntimeError(
        "Could not obtain reconstruction and latent from the V1 model."
    )


def evaluate(model, x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(
            np.asarray(x, dtype=np.float32)
        )

    if x.ndim == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        y, z = model_forward(model, x.float())

    return (
        y.detach().cpu().numpy(),
        z.detach().cpu().numpy(),
    )


def norm(v):
    return float(np.linalg.norm(v))


def pearson(x, y):
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


def try_v5_probe_function(v5, factor_values):
    """
    Attempt only clearly plausible reusable APIs.

    We do not call arbitrary functions from V5. This prevents accidentally
    starting a training run or changing files.
    """
    candidates = find_probe_constructor(v5)

    safe_names = {
        "make_probe",
        "make_probe_input",
        "build_probe",
        "build_probe_input",
        "construct_probe",
        "construct_probe_input",
        "generate_probe",
        "generate_probe_input",
        "probe_to_input",
        "factor_to_input",
        "factors_to_input",
        "make_input",
        "build_input",
        "construct_input",
        "make_image",
        "factors_to_image",
    }

    for _, name, sig, fn in candidates:
        if name not in safe_names:
            continue

        # Do not guess a complicated signature.
        try:
            params = list(inspect.signature(fn).parameters.values())
        except Exception:
            continue

        if len(params) == 1:
            try:
                out = fn(factor_values)
                if isinstance(out, torch.Tensor):
                    return out.float(), name
                if isinstance(out, np.ndarray):
                    return torch.tensor(out, dtype=torch.float32), name
            except Exception:
                pass

    return None, None


def create_input_from_v5(v5, row):
    """
    Reuse V5's actual probe constructor when it exposes one.

    We intentionally require the constructor to return the actual model input.
    """
    factors = {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "interaction": row["interaction"],
    }

    out, name = try_v5_probe_function(v5, factors)

    if out is None:
        raise RuntimeError(
            "\n"
            "V5 does not expose a safely reusable one-argument probe/input "
            "constructor under the expected API names.\n\n"
            "The V5 CSV contains factor coordinates, but the trained model "
            "expects the full 4096-dimensional image representation.\n"
            "V6 will NOT invent that mapping.\n\n"
            "Available V5 candidate functions are printed above. We need "
            "the exact V5 construction function wired into V6."
        )

    out = out.reshape(-1)

    return out, name


def finite_difference(model, base_input, index):
    """
    Perturb a single scalar in the actual 4096-dimensional input.

    This is used only after the V5 factor->image constructor has provided
    the real image. For factor-level intervention, V5 must expose how a
    factor changes that image; otherwise this would be a pixel intervention
    rather than an A/B/C/interaction intervention.
    """
    plus = base_input.clone()
    minus = base_input.clone()

    plus[index] += DELTA
    minus[index] -= DELTA

    y_plus, z_plus = evaluate(model, plus)
    y_minus, z_minus = evaluate(model, minus)

    y_plus = y_plus.reshape(-1)
    y_minus = y_minus.reshape(-1)
    z_plus = z_plus.reshape(-1)
    z_minus = z_minus.reshape(-1)

    return (
        (y_plus - y_minus) / (2 * DELTA),
        (z_plus - z_minus) / (2 * DELTA),
    )


def main():
    print("=" * 90)
    print("V6 — BEHAVIORAL COUPLING USING REAL V1 + V5")
    print("=" * 90)
    print()

    print(f"V1: {V1_PATH}")
    print(f"V5: {V5_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"V5 CSV: {V5_CSV}")
    print()

    model, v1 = load_real_model()

    print(
        "Loaded TinyAutoencoder:",
        type(model).__name__,
    )

    # Verify dimensions from the actual model.
    input_dim = None
    for name, p in model.named_parameters():
        if name == "encoder.0.weight":
            input_dim = int(p.shape[1])
            break

    print(f"Actual model input dimension: {input_dim}")

    v5 = load_v5()

    print()
    print("V5 exposed probe/input-related machinery:")
    machinery = discover_v5_globals(v5)

    if machinery:
        for item in machinery[:80]:
            print("  ", item)
    else:
        print("   (none discovered)")

    rows = read_v5_csv()

    print()
    print(f"Loaded {len(rows)} V5 probes.")

    candidates = find_probe_constructor(v5)

    print()
    print("Likely V5 probe constructors:")
    if candidates:
        for score, name, sig, _ in candidates[:20]:
            print(
                f"  score={score:2d}  {name}{sig}"
            )
    else:
        print("  none")

    print()
    print(
        "V6 will now verify that V5 can reconstruct the actual "
        "model input before running any behavioral analysis."
    )

    # Test the exact first probe.
    test_input, constructor_name = create_input_from_v5(
        v5,
        rows[0],
    )

    print(
        f"V5 constructor: {constructor_name}"
    )
    print(
        f"Constructed probe input dimension: "
        f"{test_input.numel()}"
    )

    if test_input.numel() != input_dim:
        raise RuntimeError(
            f"V5 constructor returned {test_input.numel()} values, "
            f"but the real model expects {input_dim}."
        )

    # At this point the construction is verified, but we still need the
    # factor-specific intervention mechanism. A generic pixel perturbation
    # would NOT answer the intended question, so stop unless V5 provides it.
    #
    # Search for a factor intervention function.
    intervention_candidates = []

    for name, obj in vars(v5).items():
        if not callable(obj):
            continue

        lname = name.lower()

        if (
            ("perturb" in lname or "interven" in lname)
            and ("factor" in lname or "probe" in lname)
        ):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "<?>"

            intervention_candidates.append(
                (name, sig, obj)
            )

    print()
    print("V5 factor-intervention candidates:")
    if intervention_candidates:
        for name, sig, _ in intervention_candidates:
            print(f"  {name}{sig}")
    else:
        print("  none")

    if not intervention_candidates:
        raise RuntimeError(
            "\n"
            "V5 exposes the factor coordinates and geometry outputs, but "
            "does not expose a reusable factor-intervention function that "
            "V6 can safely call.\n\n"
            "I am deliberately stopping here rather than perturbing one of "
            "the 4096 pixels. That would test pixel sensitivity, NOT the "
            "behavioral meaning of A/B/C/interaction geometry.\n\n"
            "The remaining implementation step is to extract the exact "
            "A/B/C/interaction -> image intervention code from V5."
        )

    print()
    print(
        "V5 machinery is available. The factor-intervention adapter can now "
        "be wired into the finite-difference analysis."
    )

    # We intentionally stop here in this version because the V5 source may
    # use a custom intervention signature. Once its exact signature is seen,
    # wiring it is deterministic and does not require guessing.
    print()
    print("=" * 90)
    print("V6 PRE-FLIGHT PASSED")
    print("=" * 90)
    print(
        "No behavioral numbers were generated yet; the script has verified "
        "the real V1 model and V5 machinery without performing an invalid "
        "pixel-level substitute experiment."
    )


if __name__ == "__main__":
    main()
