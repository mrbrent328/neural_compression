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

    # Use the dimensions defined by the real V1 source rather than
    # assuming the input size. The checkpoint we saw is 4096-wide, and
    # V1's ROWS/COLS are the authoritative source for that dimension.
    if hasattr(v1, "ROWS") and hasattr(v1, "COLS"):
        input_dim = int(v1.ROWS) * int(v1.COLS)
    else:
        input_dim = 4096

    model = v1.TinyAutoencoder(input_dim)
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


def get_v5_patterns(v5):
    """
    Recover the actual four pattern arrays used by V5.

    V5's generator has the signature:
        generate_from_factors(a, b, c, d, A, B, C, D)

    We inspect the module for the pattern-like arrays and use the same
    objects V5 exposes rather than constructing substitute patterns.
    """
    patterns = {}

    for name, obj in vars(v5).items():
        if isinstance(obj, np.ndarray):
            arr = np.asarray(obj)

            # V5 factor patterns should be image-shaped or flattenable to
            # the 64x64 input representation.
            if arr.size == 4096:
                lname = name.lower()

                if lname in {"a", "b", "c", "d", "pattern_a",
                             "pattern_b", "pattern_c", "pattern_d"}:
                    key = lname[-1].upper()
                    patterns[key] = arr.astype(np.float32)

    if len(patterns) == 4:
        return patterns

    # Some V5 versions may expose patterns inside a list/tuple/dict.
    for name, obj in vars(v5).items():
        lname = name.lower()

        if "pattern" not in lname:
            continue

        if isinstance(obj, dict):
            candidate = {}
            for key, value in obj.items():
                k = str(key).upper()
                if k in FACTORS and isinstance(value, np.ndarray):
                    if np.asarray(value).size == 4096:
                        candidate[k] = np.asarray(
                            value, dtype=np.float32
                        )

            if len(candidate) == 4:
                return candidate

        if isinstance(obj, (list, tuple)) and len(obj) >= 4:
            candidate = {}
            for key, value in zip(FACTORS, obj[:4]):
                if isinstance(value, np.ndarray):
                    if np.asarray(value).size == 4096:
                        candidate[key] = np.asarray(
                            value, dtype=np.float32
                        )

            if len(candidate) == 4:
                return candidate

    return None


def v5_generate(v5, factors, patterns):
    """
    Call V5's actual generator.

    V5 names the fourth factor d/D; our experiment calls it
    'interaction'. No alternative mapping is introduced.
    """
    fn = getattr(v5, "generate_from_factors", None)

    if fn is None:
        raise RuntimeError(
            "V5 does not expose generate_from_factors()."
        )

    image = fn(
        factors["A"],
        factors["B"],
        factors["C"],
        factors["interaction"],
        patterns["A"],
        patterns["B"],
        patterns["C"],
        patterns["D"],
    )

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image, dtype=np.float32).reshape(-1)

    if image.size != 4096:
        raise RuntimeError(
            f"generate_from_factors returned {image.size} values; "
            "expected 4096."
        )

    return torch.tensor(image, dtype=torch.float32)


def create_input_from_v5(v5, row, patterns):
    """
    Construct the exact 4096-dimensional image used by V5 for this probe.
    """
    factors = {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "interaction": row["interaction"],
    }

    return v5_generate(v5, factors, patterns)


def factor_intervention_inputs(v5, row, patterns, factor):
    """
    Generate the exact V5 images for factor +/- DELTA.

    This is the crucial part of V6: the intervention occurs in the known
    factor space BEFORE image generation, not by perturbing arbitrary pixels.
    """
    base = {
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "interaction": row["interaction"],
    }

    plus = dict(base)
    minus = dict(base)

    plus[factor] += DELTA
    minus[factor] -= DELTA

    return (
        v5_generate(v5, plus, patterns),
        v5_generate(v5, minus, patterns),
    )

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
    rows = read_v5_csv()

    print(f"Loaded {len(rows)} V5 probes.")

    if not hasattr(v5, "generate_from_factors"):
        raise RuntimeError(
            "V5 does not expose generate_from_factors()."
        )

    print(
        "Found V5 generator: "
        "generate_from_factors(a, b, c, d, A, B, C, D)"
    )

    patterns = get_v5_patterns(v5)

    if patterns is None:
        raise RuntimeError(
            "\n"
            "V5's factor-pattern arrays could not be recovered safely.\n"
            "The generator exists, but V6 refuses to invent A/B/C/D patterns."
        )

    print(
        "Recovered V5 factor patterns:",
        {k: int(v.size) for k, v in patterns.items()}
    )

    # Verify the exact generator produces the model-sized input.
    test_input = create_input_from_v5(
        v5,
        rows[0],
        patterns,
    )

    print(
        f"V5 generated input dimension: {test_input.numel()}"
    )

    if test_input.numel() != input_dim:
        raise RuntimeError(
            f"V5 generated {test_input.numel()} values but "
            f"the model expects {input_dim}."
        )

    # Verify the actual model can process the V5-generated input.
    base_y, base_z = evaluate(model, test_input)

    print(
        f"Preflight latent dimension: {base_z.reshape(-1).size}"
    )

    print()
    print("Running factor-space behavioral coupling...")
    print(
        "Each intervention is generated by V5 itself: "
        "factor +/- delta -> image -> model."
    )
    print()

    results = []

    for i, row in enumerate(rows):
        base_input = create_input_from_v5(
            v5,
            row,
            patterns,
        )

        base_y, base_z = evaluate(
            model,
            base_input,
        )

        result = {
            "probe": row["probe"],
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "interaction": row["interaction"],
            "v5_total_sensitivity": row["v5_total_sensitivity"],
            "v5_top_10pct": row["v5_top_10pct"],
        }

        total_latent = 0.0

        for factor in FACTORS:
            plus_input, minus_input = factor_intervention_inputs(
                v5,
                row,
                patterns,
                factor,
            )

            y_plus, z_plus = evaluate(
                model,
                plus_input,
            )

            y_minus, z_minus = evaluate(
                model,
                minus_input,
            )

            dz = (
                z_plus.reshape(-1)
                - z_minus.reshape(-1)
            ) / (2.0 * DELTA)

            dy = (
                y_plus.reshape(-1)
                - y_minus.reshape(-1)
            ) / (2.0 * DELTA)

            latent_mag = norm(dz)
            output_mag = norm(dy)

            result[f"latent_mag_{factor}"] = latent_mag
            result[f"output_mag_{factor}"] = output_mag
            result[f"output_per_latent_{factor}"] = (
                output_mag / max(latent_mag, 1e-12)
            )

            total_latent += latent_mag

        result["total_latent_sensitivity"] = total_latent
        results.append(result)

        if (i + 1) % 25 == 0 or i == 0:
            print(
                f"probe {i + 1:4d}/{len(rows)}"
            )

    # Use the V5 top-10% label when present. Also calculate our own top 10%
    # from the resulting latent sensitivity so we can compare them.
    latent_total = np.array([
        r["total_latent_sensitivity"]
        for r in results
    ])

    threshold = np.quantile(
        latent_total,
        1.0 - TOP_PERCENT,
    )

    for r in results:
        r["v6_top_10pct"] = (
            r["total_latent_sensitivity"] >= threshold
        )

    summary = {
        "experiment": "V6_behavioral_coupling",
        "delta": DELTA,
        "n_probes": len(results),
        "model": str(MODEL_PATH),
        "v1_source": str(V1_PATH),
        "v5_source": str(V5_PATH),
        "v5_probe_source": str(V5_CSV),
        "v5_generator": "generate_from_factors",
        "v6_latent_top10_threshold": float(threshold),
        "factors": {},
    }

    v5_top_mask = np.array([
        r["v5_top_10pct"]
        for r in results
    ], dtype=bool)

    v6_top_mask = np.array([
        r["v6_top_10pct"]
        for r in results
    ], dtype=bool)

    summary["top10_overlap"] = {
        "count_v5_top10": int(v5_top_mask.sum()),
        "count_v6_top10": int(v6_top_mask.sum()),
        "intersection": int(
            np.logical_and(
                v5_top_mask,
                v6_top_mask,
            ).sum()
        ),
        "jaccard": float(
            np.logical_and(
                v5_top_mask,
                v6_top_mask,
            ).sum()
            /
            max(
                np.logical_or(
                    v5_top_mask,
                    v6_top_mask,
                ).sum(),
                1,
            )
        ),
    }

    for factor in FACTORS:
        latent = np.array([
            r[f"latent_mag_{factor}"]
            for r in results
        ])

        output = np.array([
            r[f"output_mag_{factor}"]
            for r in results
        ])

        gain = np.array([
            r[f"output_per_latent_{factor}"]
            for r in results
        ])

        summary["factors"][factor] = {
            "latent_mean": float(np.mean(latent)),
            "latent_median": float(np.median(latent)),
            "latent_max": float(np.max(latent)),
            "output_mean": float(np.mean(output)),
            "output_median": float(np.median(output)),
            "output_max": float(np.max(output)),
            "gain_mean": float(np.mean(gain)),
            "gain_median": float(np.median(gain)),
            "latent_output_correlation": pearson(
                latent,
                output,
            ),
            "latent_gain_correlation": pearson(
                latent,
                gain,
            ),
            "v5_top10_output_median": float(
                np.median(output[v5_top_mask])
            ),
            "v5_rest_output_median": float(
                np.median(output[~v5_top_mask])
            ),
            "v6_top10_output_median": float(
                np.median(output[v6_top_mask])
            ),
            "v6_rest_output_median": float(
                np.median(output[~v6_top_mask])
            ),
            "v5_top10_gain_median": float(
                np.median(gain[v5_top_mask])
            ),
            "v5_rest_gain_median": float(
                np.median(gain[~v5_top_mask])
            ),
        }

    csv_path = OUT_DIR / "v6_behavioral_coupling.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    json_path = OUT_DIR / "v6_behavioral_coupling_results.json"
    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # ---- plots ----

    for factor in FACTORS:
        latent = np.array([
            r[f"latent_mag_{factor}"]
            for r in results
        ])

        output = np.array([
            r[f"output_mag_{factor}"]
            for r in results
        ])

        plt.figure(figsize=(8, 7))

        plt.scatter(
            latent[~v5_top_mask],
            output[~v5_top_mask],
            s=14,
            alpha=0.45,
            label="V5 rest 90%",
        )

        plt.scatter(
            latent[v5_top_mask],
            output[v5_top_mask],
            s=26,
            alpha=0.85,
            label="V5 top 10%",
        )

        plt.xlabel(f"|dZ/d{factor}|")
        plt.ylabel(f"|dY/d{factor}|")
        plt.title(
            f"V6 — {factor}: latent movement vs output movement"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"v6_{factor}_latent_vs_output.png",
            dpi=160,
        )
        plt.close()

    # Factor-wise top10/rest comparison.
    x = np.arange(len(FACTORS))
    width = 0.35

    top_output = [
        summary["factors"][f]["v5_top10_output_median"]
        for f in FACTORS
    ]

    rest_output = [
        summary["factors"][f]["v5_rest_output_median"]
        for f in FACTORS
    ]

    plt.figure(figsize=(9, 6))
    plt.bar(
        x - width / 2,
        rest_output,
        width,
        label="V5 rest 90%",
    )
    plt.bar(
        x + width / 2,
        top_output,
        width,
        label="V5 top 10%",
    )
    plt.xticks(x, FACTORS)
    plt.ylabel("median reconstruction movement")
    plt.title("V6 — Behavioral output in V5 high-sensitivity region")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "v6_top10_vs_rest.png",
        dpi=160,
    )
    plt.close()

    # V5 vs V6 total sensitivity.
    v5_sensitivity = np.array([
        r["v5_total_sensitivity"]
        for r in results
    ])

    plt.figure(figsize=(8, 7))
    plt.scatter(
        v5_sensitivity[~v6_top_mask],
        latent_total[~v6_top_mask],
        s=14,
        alpha=0.45,
        label="V6 rest",
    )
    plt.scatter(
        v5_sensitivity[v6_top_mask],
        latent_total[v6_top_mask],
        s=26,
        alpha=0.85,
        label="V6 top 10%",
    )
    plt.xlabel("V5 total sensitivity")
    plt.ylabel("V6 total latent sensitivity")
    plt.title("V5 geometry vs V6 behavioral latent response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "v6_v5_geometry_vs_behavior.png",
        dpi=160,
    )
    plt.close()

    print()
    print("=" * 90)
    print("V6 COMPLETE")
    print("=" * 90)
    print()

    for factor in FACTORS:
        s = summary["factors"][factor]
        print(
            f"{factor:12s} "
            f"latent→output r={s['latent_output_correlation']:+.4f} | "
            f"gain median={s['gain_median']:.6g} | "
            f"V5-top output={s['v5_top10_output_median']:.6g} | "
            f"rest output={s['v5_rest_output_median']:.6g}"
        )

    print()
    print(
        "V5/V6 top-10% overlap:",
        summary["top10_overlap"],
    )
    print()
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"DIR : {OUT_DIR}")



if __name__ == "__main__":
    main()
