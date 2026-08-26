#!/usr/bin/env python3

"""
BLIND SCIENTIFIC LAW DISCOVERY EXPERIMENT
-----------------------------------------

The discovery engine is NOT given the generating equation.

It is also NOT told:
    - which variables matter
    - how many terms matter
    - which mathematical operations matter

The engine:
    1. Generates observations from a hidden physical relationship.
    2. Adds measurement noise.
    3. Splits observations into train/test sets.
    4. Builds a library of mathematical candidate features.
    5. Searches combinations of candidate terms.
    6. Fits coefficients using ONLY training data.
    7. Evaluates every candidate on held-out data.
    8. Penalizes unnecessary complexity.
    9. Tests the winning equation on completely fresh data.
   10. Reveals the hidden law only AFTER discovery.

This is intentionally a small symbolic-regression-style experiment.
"""

import itertools
import math
import random
import time

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TOTAL_OBSERVATIONS = 700
TRAIN_FRACTION = 0.70

NOISE_STD = 0.05

MAX_TERMS = 5

# Number of candidate features to consider.
# Keeping this moderate makes brute-force searching practical.
MAX_FEATURES = 43

# Complexity penalty.
# Larger values favor simpler equations.
COMPLEXITY_PENALTY = 0.003

# Fresh data used only AFTER discovery.
FRESH_OBSERVATIONS = 500


# ============================================================
# HIDDEN LAW
# ============================================================

# IMPORTANT:
# The discovery system never receives this.
#
# This is the relationship we are trying to rediscover.
#
# output = 2.75*x^2 - 1.50*y + 0.80*x*z + 3.20
#
# We deliberately include irrelevant variables such as w.

def hidden_law(x, y, z, w):
    return (
        2.75 * x**2
        - 1.50 * y
        + 0.80 * x * z
        + 3.20
    )


# ============================================================
# RANDOM DATA GENERATION
# ============================================================

def make_dataset(n, noisy=True, seed=None):
    rng = np.random.default_rng(seed)

    # Variables.
    #
    # x, y, z actually matter.
    # w is deliberately irrelevant.
    #
    # Additional variables can be added here later.
    x = rng.uniform(-3.0, 3.0, n)
    y = rng.uniform(-3.0, 3.0, n)
    z = rng.uniform(-3.0, 3.0, n)
    w = rng.uniform(-3.0, 3.0, n)

    clean_y = hidden_law(x, y, z, w)

    if noisy:
        noise = rng.normal(0.0, NOISE_STD, n)
        output = clean_y + noise
    else:
        output = clean_y

    X = {
        "x": x,
        "y": y,
        "z": z,
        "w": w,
    }

    return X, output


# ============================================================
# MATHEMATICAL FEATURE LIBRARY
# ============================================================

class Feature:
    def __init__(self, name, values, complexity=1):
        self.name = name
        self.values = np.asarray(values, dtype=np.float64)
        self.complexity = complexity

    def __repr__(self):
        return self.name


def safe_divide(a, b):
    result = np.zeros_like(a, dtype=np.float64)

    mask = np.abs(b) > 1e-8

    result[mask] = a[mask] / b[mask]

    result[~mask] = 0.0

    return np.clip(result, -1e6, 1e6)


def safe_exp(a):
    return np.clip(np.exp(np.clip(a, -10, 10)), -1e6, 1e6)


def safe_log(a):
    return np.log(np.abs(a) + 1e-8)


def safe_sqrt(a):
    return np.sqrt(np.abs(a))


def build_feature_library(X):
    x = X["x"]
    y = X["y"]
    z = X["z"]
    w = X["w"]

    features = []

    def add(name, values, complexity=1):
        values = np.asarray(values, dtype=np.float64)

        # Remove NaN / infinity.
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        # Avoid pathological numerical ranges.
        values = np.clip(values, -1e6, 1e6)

        features.append(
            Feature(name, values, complexity)
        )

    # Constant.
    add("1", np.ones_like(x), 1)

    variables = {
        "x": x,
        "y": y,
        "z": z,
        "w": w,
    }

    # --------------------------------------------------------
    # Single-variable terms
    # --------------------------------------------------------

    for name, v in variables.items():

        add(name, v)

        add(
            f"{name}^2",
            v**2,
            2
        )

        add(
            f"{name}^3",
            v**3,
            3
        )

        add(
            f"sqrt(|{name}|)",
            safe_sqrt(v),
            2
        )

        add(
            f"log(|{name}|)",
            safe_log(v),
            2
        )

        add(
            f"sin({name})",
            np.sin(v),
            2
        )

        add(
            f"cos({name})",
            np.cos(v),
            2
        )

        add(
            f"exp({name})",
            safe_exp(v),
            3
        )

        add(
            f"1/{name}",
            safe_divide(
                np.ones_like(v),
                v
            ),
            2
        )

    # --------------------------------------------------------
    # Pairwise products
    # --------------------------------------------------------

    names = list(variables.keys())

    for a_name, b_name in itertools.combinations(names, 2):

        a = variables[a_name]
        b = variables[b_name]

        add(
            f"{a_name}*{b_name}",
            a * b,
            2
        )

        add(
            f"{a_name}/{b_name}",
            safe_divide(a, b),
            2
        )

        add(
            f"{b_name}/{a_name}",
            safe_divide(b, a),
            2
        )

    # --------------------------------------------------------
    # A few additional nonlinear interactions
    # --------------------------------------------------------

    for a_name, b_name in itertools.combinations(names, 2):

        a = variables[a_name]
        b = variables[b_name]

        add(
            f"sin({a_name}*{b_name})",
            np.sin(a * b),
            3
        )

        add(
            f"({a_name}*{b_name})^2",
            (a * b) ** 2,
            3
        )

    return features


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_columns(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)

    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)

    stds[stds < 1e-10] = 1.0

    standardized = (
        matrix - means
    ) / stds

    return standardized, means, stds


# ============================================================
# FIT EQUATION
# ============================================================

def fit_equation(features, y_train, selected_indices):

    selected = [
        features[i]
        for i in selected_indices
    ]

    A = np.column_stack([
        f.values
        for f in selected
    ])

    # Normalize columns for numerical stability.
    A_scaled, means, stds = standardize_columns(A)

    # Add ridge stabilization.
    #
    # Very small ridge value keeps the problem stable
    # without materially changing the solution.
    ridge = 1e-8

    ATA = A_scaled.T @ A_scaled

    ATy = A_scaled.T @ y_train

    coefficients_scaled = np.linalg.solve(
        ATA + ridge * np.eye(len(selected)),
        ATy
    )

    # Convert coefficients back to original feature scale.
    coefficients = coefficients_scaled / stds

    intercept = (
        -np.sum(
            coefficients_scaled * means / stds
        )
    )

    return coefficients, intercept


# ============================================================
# PREDICTION
# ============================================================

def predict(features, selected_indices, coefficients, intercept):

    prediction = np.full(
        len(features[0].values),
        intercept,
        dtype=np.float64
    )

    for coefficient, index in zip(
        coefficients,
        selected_indices
    ):
        prediction += (
            coefficient
            * features[index].values
        )

    return prediction


# ============================================================
# METRICS
# ============================================================

def rmse(actual, predicted):

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def mae(actual, predicted):

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


# ============================================================
# EQUATION FORMATTING
# ============================================================

def format_equation(
    features,
    indices,
    coefficients,
    intercept
):

    pieces = []

    if abs(intercept) > 1e-8:
        pieces.append(
            f"({intercept:+.8f})*1"
        )

    for coefficient, index in zip(
        coefficients,
        indices
    ):

        if abs(coefficient) < 1e-8:
            continue

        pieces.append(
            f"({coefficient:+.8f})*"
            f"{features[index].name}"
        )

    return " ".join(pieces)


# ============================================================
# DISCOVERY
# ============================================================

def discover(
    train_features,
    train_y,
    test_features,
    test_y,
    max_terms
):

    feature_count = len(train_features)

    results = []

    print()
    print(
        f"Searching {feature_count} mathematical features..."
    )

    for term_count in range(
        1,
        max_terms + 1
    ):

        print(
            f"Searching {term_count}-term equations..."
        )

        # Avoid impossible combinatorial explosion.
        #
        # 43 choose 5 is enormous, so we use a screening
        # stage before the expensive combinatorial search.
        #
        # First rank individual features by correlation
        # with the training target.

        correlations = []

        for i, feature in enumerate(
            train_features
        ):

            values = feature.values

            if np.std(values) < 1e-10:
                correlation = 0.0

            else:
                correlation = abs(
                    np.corrcoef(
                        values,
                        train_y
                    )[0, 1]
                )

                if not np.isfinite(correlation):
                    correlation = 0.0

            correlations.append(
                (correlation, i)
            )

        correlations.sort(
            reverse=True
        )

        # Keep a reasonably broad candidate pool.
        #
        # This is deliberately wider than the expected
        # number of relevant terms so the engine can discover
        # irrelevant variables are unnecessary.
        candidate_pool_size = min(
            20,
            feature_count
        )

        candidate_indices = [
            index
            for _, index in
            correlations[:candidate_pool_size]
        ]

        # Always include the constant.
        constant_indices = [
            i for i, f in enumerate(
                train_features
            )
            if f.name == "1"
        ]

        for i in constant_indices:
            if i not in candidate_indices:
                candidate_indices.append(i)

        # ----------------------------------------------------
        # Search combinations.
        # ----------------------------------------------------

        combinations_checked = 0

        for selected_indices in itertools.combinations(
            candidate_indices,
            term_count
        ):

            combinations_checked += 1

            try:

                coefficients, intercept = fit_equation(
                    train_features,
                    train_y,
                    selected_indices
                )

                train_prediction = predict(
                    train_features,
                    selected_indices,
                    coefficients,
                    intercept
                )

                test_prediction = predict(
                    test_features,
                    selected_indices,
                    coefficients,
                    intercept
                )

                train_error = rmse(
                    train_y,
                    train_prediction
                )

                test_error = rmse(
                    test_y,
                    test_prediction
                )

                # Complexity score.
                #
                # Held-out performance matters most.
                # Complexity is a small tie-breaker.
                complexity = sum(
                    train_features[i].complexity
                    for i in selected_indices
                )

                score = (
                    test_error
                    + COMPLEXITY_PENALTY
                    * complexity
                )

                results.append({
                    "score": score,
                    "train_rmse": train_error,
                    "test_rmse": test_error,
                    "terms": term_count,
                    "indices": selected_indices,
                    "coefficients": coefficients,
                    "intercept": intercept,
                    "complexity": complexity,
                })

            except (
                np.linalg.LinAlgError,
                ValueError,
                FloatingPointError
            ):
                continue

        print(
            f"  combinations checked: "
            f"{combinations_checked}"
        )

    # Sort by score.
    results.sort(
        key=lambda r: r["score"]
    )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    results,
    features,
    limit=20
):

    print()
    print("=" * 110)
    print("DISCOVERY RESULTS")
    print("=" * 110)

    print(
        f"{'Rank':<6}"
        f"{'Score':>11}"
        f"{'Train RMSE':>14}"
        f"{'Test RMSE':>14}"
        f"{'Terms':>8}"
        f"  Equation"
    )

    print("-" * 110)

    for rank, result in enumerate(
        results[:limit],
        start=1
    ):

        equation = format_equation(
            features,
            result["indices"],
            result["coefficients"],
            result["intercept"]
        )

        print(
            f"{rank:<6}"
            f"{result['score']:>11.6f}"
            f"{result['train_rmse']:>14.6f}"
            f"{result['test_rmse']:>14.6f}"
            f"{result['terms']:>8}"
            f"  {equation}"
        )


# ============================================================
# FRESH-DATA TEST
# ============================================================

def evaluate_fresh(
    result,
    feature_builder,
    seed
):

    X_fresh, y_fresh = make_dataset(
        FRESH_OBSERVATIONS,
        noisy=False,
        seed=seed
    )

    features_fresh = feature_builder(
        X_fresh
    )

    prediction = predict(
        features_fresh,
        result["indices"],
        result["coefficients"],
        result["intercept"]
    )

    error = rmse(
        y_fresh,
        prediction
    )

    absolute_error = mae(
        y_fresh,
        prediction
    )

    return error, absolute_error


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(SEED)
    random.seed(SEED)

    print("=" * 70)
    print("BLIND SCIENTIFIC LAW DISCOVERY")
    print("=" * 70)

    print()
    print(
        "The discovery engine is NOT given the generating equation."
    )

    print(
        "It is NOT told which variables matter."
    )

    print(
        "It is NOT told how many terms the law contains."
    )

    print(
        "It is NOT told which mathematical operations matter."
    )

    print()

    # --------------------------------------------------------
    # Generate dataset
    # --------------------------------------------------------

    X, y = make_dataset(
        TOTAL_OBSERVATIONS,
        noisy=True,
        seed=SEED
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED + 1
    )

    indices = np.arange(
        TOTAL_OBSERVATIONS
    )

    rng.shuffle(indices)

    train_count = int(
        TOTAL_OBSERVATIONS
        * TRAIN_FRACTION
    )

    train_indices = indices[
        :train_count
    ]

    test_indices = indices[
        train_count:
    ]

    train_X = {
        name: values[train_indices]
        for name, values in X.items()
    }

    test_X = {
        name: values[test_indices]
        for name, values in X.items()
    }

    train_y = y[
        train_indices
    ]

    test_y = y[
        test_indices
    ]

    # --------------------------------------------------------
    # Build feature libraries separately.
    #
    # The actual mathematical feature definitions are the
    # same, but values correspond to each split.
    # --------------------------------------------------------

    train_features = build_feature_library(
        train_X
    )

    test_features = build_feature_library(
        test_X
    )

    print(
        f"Total observations:       "
        f"{TOTAL_OBSERVATIONS}"
    )

    print(
        f"Training observations:    "
        f"{len(train_y)}"
    )

    print(
        f"Held-out observations:    "
        f"{len(test_y)}"
    )

    print(
        f"Measurement noise:        "
        f"{NOISE_STD}"
    )

    print(
        f"Mathematical features:    "
        f"{len(train_features)}"
    )

    print(
        f"Searching term counts:    "
        f"1 through {MAX_TERMS}"
    )

    # --------------------------------------------------------
    # Discovery
    # --------------------------------------------------------

    start_time = time.time()

    results = discover(
        train_features,
        train_y,
        test_features,
        test_y,
        MAX_TERMS
    )

    elapsed = time.time() - start_time

    print()
    print(
        f"Discovery time: {elapsed:.2f} seconds"
    )

    if not results:
        print(
            "No valid equations were discovered."
        )
        return

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print_results(
        results,
        train_features,
        limit=20
    )

    # --------------------------------------------------------
    # Best equation
    # --------------------------------------------------------

    best = results[0]

    best_equation = format_equation(
        train_features,
        best["indices"],
        best["coefficients"],
        best["intercept"]
    )

    print()
    print("=" * 70)
    print("BEST DISCOVERED RELATIONSHIP")
    print("=" * 70)

    print()
    print(best_equation)

    print()
    print(
        f"Training RMSE: "
        f"{best['train_rmse']:.8f}"
    )

    print(
        f"Held-out RMSE: "
        f"{best['test_rmse']:.8f}"
    )

    # --------------------------------------------------------
    # Fresh data
    # --------------------------------------------------------

    fresh_rmse, fresh_mae = evaluate_fresh(
        best,
        build_feature_library,
        SEED + 999
    )

    print()
    print("=" * 70)
    print("COMPLETELY FRESH DATA")
    print("=" * 70)

    print()
    print(
        f"Fresh-data RMSE: "
        f"{fresh_rmse:.8f}"
    )

    print(
        f"Fresh-data MAE:  "
        f"{fresh_mae:.8f}"
    )

    # --------------------------------------------------------
    # Hidden law revealed ONLY NOW.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HIDDEN GENERATING LAW")
    print("=" * 70)

    print()
    print(
        "output = "
        "2.75*x^2 "
        "- 1.50*y "
        "+ 0.80*x*z "
        "+ 3.20"
    )

    print()
    print(
        "Important:"
    )

    print(
        "The discovery engine was NOT given this equation."
    )

    print(
        "It was also NOT told that w was irrelevant."
    )

    print(
        "It was NOT told the number of terms."
    )

    print(
        "It was NOT told which operations were correct."
    )

    print()
    print("=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()