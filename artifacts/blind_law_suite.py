import itertools
import math
import time
import numpy as np


# ================================================================
# BLIND SCIENTIFIC LAW DISCOVERY SUITE
#
# The discovery engine is NOT given:
#   - the generating equation
#   - which variables matter
#   - how many terms matter
#   - which mathematical operations matter
#
# It receives ONLY observations.
#
# The hidden generating laws are used ONLY AFTER discovery
# to evaluate whether the discovered structure is correct.
# ================================================================


RNG = np.random.default_rng(42)

N_TOTAL = 700
TRAIN_FRAC = 0.70
TEST_FRAC = 0.30

NOISE = 0.05

TERM_COUNTS = range(1, 6)

# Complexity penalty.
# A small positive value favors simpler equations.
LAMBDA = 0.01

TOP_RESULTS = 10


# ================================================================
# SAFE MATHEMATICAL FUNCTIONS
# ================================================================

def safe_log(x):
    return np.log(np.abs(x) + 1e-8)


def safe_sqrt(x):
    return np.sqrt(np.abs(x) + 1e-12)


def safe_inv(x):
    return 1.0 / (np.abs(x) + 1e-8)


def safe_exp(x):
    # Prevent overflow.
    return np.exp(np.clip(x, -8, 8))


def safe_div(a, b):
    return a / (np.abs(b) + 1e-8)


# ================================================================
# FEATURE REPRESENTATION
# ================================================================

class Feature:
    def __init__(self, name, values, complexity=1.0):
        self.name = name
        self.values = np.asarray(values, dtype=np.float64)
        self.complexity = complexity


def build_features(X):
    """
    Build a deliberately broad mathematical vocabulary.

    Variables:
        x, y, z, w

    The engine does NOT know which ones matter.
    """

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    w = X[:, 3]

    features = []

    def add(name, values, complexity=1.0):
        values = np.asarray(values, dtype=np.float64)

        # Replace numerical disasters.
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6
        )

        # Clip pathological feature values.
        values = np.clip(values, -1e6, 1e6)

        features.append(
            Feature(name, values, complexity)
        )

    # Constants
    add("1", np.ones_like(x), 1.0)

    # Variables
    add("x", x)
    add("y", y)
    add("z", z)
    add("w", w)

    # Powers
    for name, a in [
        ("x", x),
        ("y", y),
        ("z", z),
        ("w", w),
    ]:
        add(f"{name}^2", a ** 2, 1.5)
        add(f"{name}^3", a ** 3, 2.0)

    # Square roots
    add("sqrt(|x|)", safe_sqrt(x), 2.0)
    add("sqrt(|y|)", safe_sqrt(y), 2.0)
    add("sqrt(|z|)", safe_sqrt(z), 2.0)
    add("sqrt(|w|)", safe_sqrt(w), 2.0)

    # Logs
    add("log(|x|)", safe_log(x), 2.0)
    add("log(|y|)", safe_log(y), 2.0)
    add("log(|z|)", safe_log(z), 2.0)
    add("log(|w|)", safe_log(w), 2.0)

    # Exponentials
    add("exp(x)", safe_exp(x), 3.0)
    add("exp(y)", safe_exp(y), 3.0)
    add("exp(z)", safe_exp(z), 3.0)
    add("exp(w)", safe_exp(w), 3.0)

    # Trigonometric
    add("sin(x)", np.sin(x), 2.0)
    add("sin(y)", np.sin(y), 2.0)
    add("sin(z)", np.sin(z), 2.0)
    add("sin(w)", np.sin(w), 2.0)

    add("cos(x)", np.cos(x), 2.0)
    add("cos(y)", np.cos(y), 2.0)
    add("cos(z)", np.cos(z), 2.0)
    add("cos(w)", np.cos(w), 2.0)

    # Pairwise products
    add("x*y", x * y, 2.0)
    add("x*z", x * z, 2.0)
    add("x*w", x * w, 2.0)
    add("y*z", y * z, 2.0)
    add("y*w", y * w, 2.0)
    add("z*w", z * w, 2.0)

    # Pairwise squares
    add("(x*y)^2", (x * y) ** 2, 3.0)
    add("(x*z)^2", (x * z) ** 2, 3.0)
    add("(x*w)^2", (x * w) ** 2, 3.0)
    add("(y*z)^2", (y * z) ** 2, 3.0)
    add("(y*w)^2", (y * w) ** 2, 3.0)
    add("(z*w)^2", (z * w) ** 2, 3.0)

    # Some ratios
    add("x/y", safe_div(x, y), 3.0)
    add("x/z", safe_div(x, z), 3.0)
    add("x/w", safe_div(x, w), 3.0)
    add("y/x", safe_div(y, x), 3.0)
    add("y/z", safe_div(y, z), 3.0)
    add("z/x", safe_div(z, x), 3.0)
    add("z/y", safe_div(z, y), 3.0)

    return features


# ================================================================
# REGRESSION
# ================================================================

def normalize_features(A):
    """
    Normalize each candidate feature before regression.

    This keeps huge exponentials or polynomial terms from
    dominating the least-squares calculation.
    """

    A = np.asarray(A, dtype=np.float64)

    means = np.mean(A, axis=0)
    stds = np.std(A, axis=0)

    stds[stds < 1e-10] = 1.0

    normalized = (A - means) / stds

    return normalized, means, stds


def fit_candidate(features, y_train, y_test, selected):
    """
    Fit a linear combination of selected mathematical features.
    """

    train_matrix = np.column_stack([
        features[i].values[:len(y_train)]
        for i in selected
    ])

    test_matrix = np.column_stack([
        features[i].values[:len(y_train) + len(y_test)][len(y_train):]
        for i in selected
    ])

    train_matrix, means, stds = normalize_features(train_matrix)

    test_matrix = (
        test_matrix - means
    ) / stds

    # Add no separate intercept because feature "1" exists.
    try:
        coefficients, *_ = np.linalg.lstsq(
            train_matrix,
            y_train,
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    train_prediction = train_matrix @ coefficients
    test_prediction = test_matrix @ coefficients

    train_rmse = float(
        np.sqrt(np.mean((y_train - train_prediction) ** 2))
    )

    test_rmse = float(
        np.sqrt(np.mean((y_test - test_prediction) ** 2))
    )

    complexity = sum(
        features[i].complexity
        for i in selected
    )

    score = train_rmse + LAMBDA * complexity

    return {
        "selected": selected,
        "coefficients": coefficients,
        "means": means,
        "stds": stds,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "complexity": complexity,
        "score": score,
    }


# ================================================================
# DISCOVERY
# ================================================================

def discover(train_X, train_y, test_X, test_y):
    """
    Exhaustively search combinations of mathematical features.
    """

    # Build features separately so test data never leaks into training.
    train_features = build_features(train_X)
    test_features = build_features(test_X)

    # Recombine into the structure expected by fit_candidate.
    combined_features = []

    for train_feature, test_feature in zip(
        train_features,
        test_features
    ):
        combined = np.concatenate([
            train_feature.values,
            test_feature.values
        ])

        combined_features.append(
            Feature(
                train_feature.name,
                combined,
                train_feature.complexity
            )
        )

    features = combined_features

    print()
    print("Mathematical features:", len(features))
    print("Searching term counts:",
          min(TERM_COUNTS), "through", max(TERM_COUNTS))
    print()

    all_results = []

    for term_count in TERM_COUNTS:

        total = math.comb(len(features), term_count)

        print(
            f"Searching {term_count}-term equations..."
        )
        print(
            f"combinations checked: {total}"
        )

        checked = 0

        for selected in itertools.combinations(
            range(len(features)),
            term_count
        ):

            result = fit_candidate(
                features,
                train_y,
                test_y,
                selected
            )

            if result is not None:
                all_results.append(result)

            checked += 1

    # Rank by complexity-aware score.
    all_results.sort(
        key=lambda r: r["score"]
    )

    return all_results, features


# ================================================================
# FORMAT EQUATION
# ================================================================

def format_equation(result, features):

    parts = []

    for coefficient, index in zip(
        result["coefficients"],
        result["selected"]
    ):
        name = features[index].name

        # Account for normalized feature scale.
        # The underlying physical coefficient is:
        #
        # coefficient / std
        #
        # and the intercept contribution from normalization
        # is handled separately.
        physical_coefficient = coefficient / result["stds"][
            list(result["selected"]).index(index)
        ]

        parts.append(
            f"({physical_coefficient:+.8f})*{name}"
        )

    return " ".join(parts)


# ================================================================
# GENERATING HIDDEN DATASETS
# ================================================================

def generate_dataset(name, n, noise, seed):
    """
    Generate a hidden scientific relationship.

    IMPORTANT:
    The returned generating equation is kept completely separate
    from the discovery engine.
    """

    rng = np.random.default_rng(seed)

    # Keep variables in ranges that make the candidate functions
    # numerically manageable.
    x = rng.uniform(0.5, 3.0, n)
    y = rng.uniform(-2.0, 2.0, n)
    z = rng.uniform(-2.0, 2.0, n)
    w = rng.uniform(-2.0, 2.0, n)

    X = np.column_stack([x, y, z, w])

    if name == "QUADRATIC":
        clean = (
            4.905 * x**2
        )

        equation = "output = 4.905*x^2"

    elif name == "MIXED":
        clean = (
            3.20
            - 1.50 * y
            + 2.75 * x**2
            + 0.80 * x * z
        )

        equation = (
            "output = 2.75*x^2 - 1.50*y "
            "+ 0.80*x*z + 3.20"
        )

    elif name == "INVERSE":
        clean = (
            2.50
            + 3.20 / x
            - 1.10 * y
        )

        equation = (
            "output = 2.50 + 3.20/x - 1.10*y"
        )

    elif name == "EXPONENTIAL":
        clean = (
            1.20
            + 0.80 * np.exp(x / 2.0)
            - 1.40 * z
        )

        equation = (
            "output = 1.20 + 0.80*exp(x/2) - 1.40*z"
        )

    elif name == "TRIG":
        clean = (
            2.00
            + 1.50 * np.sin(x)
            - 0.75 * y
        )

        equation = (
            "output = 2.00 + 1.50*sin(x) - 0.75*y"
        )

    else:
        raise ValueError(
            f"Unknown dataset: {name}"
        )

    noisy = clean + rng.normal(
        0.0,
        noise,
        n
    )

    return X, noisy, clean, equation


# ================================================================
# EVALUATION
# ================================================================

def evaluate_on_fresh_data(result, features, X):
    """
    Evaluate the discovered normalized model on completely fresh
    observations.

    Because the candidate features are normalized, we reproduce
    exactly the same normalization used during discovery.
    """

    fresh_features = build_features(X)

    selected_matrix = np.column_stack([
        fresh_features[i].values
        for i in result["selected"]
    ])

    normalized = (
        selected_matrix - result["means"]
    ) / result["stds"]

    return normalized @ result["coefficients"]


def rmse(a, b):
    return float(
        np.sqrt(np.mean((a - b) ** 2))
    )


def mae(a, b):
    return float(
        np.mean(np.abs(a - b))
    )


# ================================================================
# RUN ONE EXPERIMENT
# ================================================================

def run_experiment(name, seed):

    print()
    print("=" * 70)
    print("DATASET:", name)
    print("=" * 70)

    # ------------------------------------------------------------
    # Training / held-out data
    # ------------------------------------------------------------

    X, y, clean, hidden_equation = generate_dataset(
        name,
        N_TOTAL,
        NOISE,
        seed
    )

    split = int(
        N_TOTAL * TRAIN_FRAC
    )

    train_X = X[:split]
    test_X = X[split:]

    train_y = y[:split]
    test_y = y[split:]

    print()
    print("Total observations:", N_TOTAL)
    print("Training observations:", len(train_y))
    print("Held-out observations:", len(test_y))
    print("Measurement noise:", NOISE)

    # ------------------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------------------

    start = time.time()

    results, features = discover(
        train_X,
        train_y,
        test_X,
        test_y
    )

    elapsed = time.time() - start

    print()
    print(
        f"Discovery time: {elapsed:.2f} seconds"
    )

    print()
    print(
        "## TOP DISCOVERED STRUCTURES"
    )

    print()

    for rank, result in enumerate(
        results[:TOP_RESULTS],
        1
    ):

        equation = format_equation(
            result,
            features
        )

        print(
            f"{rank:2d}  "
            f"Score={result['score']:.6f}  "
            f"Train={result['train_rmse']:.6f}  "
            f"Test={result['test_rmse']:.6f}  "
            f"Complexity={result['complexity']:.3f}"
        )

        print(
            "    ",
            equation
        )

    # ------------------------------------------------------------
    # BEST COMPLEXITY-AWARE MODEL
    # ------------------------------------------------------------

    best = results[0]

    equation = format_equation(
        best,
        features
    )

    print()
    print(
        "BEST DISCOVERED MODEL"
    )
    print(
        "Score:",
        f"{best['score']:.8f}"
    )
    print(
        "Training RMSE:",
        f"{best['train_rmse']:.8f}"
    )
    print(
        "Held-out RMSE:",
        f"{best['test_rmse']:.8f}"
    )
    print(
        "Complexity:",
        f"{best['complexity']:.4f}"
    )
    print(
        "Terms:",
        len(best["selected"])
    )

    print()
    print(
        "Discovered structure:"
    )
    print(
        equation
    )

    # ------------------------------------------------------------
    # VARIABLES USED
    # ------------------------------------------------------------

    variables = set()

    for index in best["selected"]:
        name_text = features[index].name

        for variable in [
            "x",
            "y",
            "z",
            "w"
        ]:
            if variable in name_text:
                variables.add(variable)

    print()
    print(
        "Variables appearing in discovered model:"
    )
    print(
        ", ".join(sorted(variables))
    )

    # ------------------------------------------------------------
    # FRESH DATA
    # ------------------------------------------------------------

    fresh_X, fresh_noisy, fresh_clean, _ = generate_dataset(
        name,
        500,
        NOISE,
        seed + 10000
    )

    fresh_prediction = evaluate_on_fresh_data(
        best,
        features,
        fresh_X
    )

    fresh_rmse_noisy = rmse(
        fresh_noisy,
        fresh_prediction
    )

    fresh_rmse_clean = rmse(
        fresh_clean,
        fresh_prediction
    )

    fresh_mae_clean = mae(
        fresh_clean,
        fresh_prediction
    )

    print()
    print(
        "FRESH DATA"
    )

    print(
        "Fresh-data RMSE vs noisy observations:",
        f"{fresh_rmse_noisy:.8f}"
    )

    print(
        "Fresh-data RMSE vs clean law:",
        f"{fresh_rmse_clean:.8f}"
    )

    print(
        "Fresh-data MAE vs clean law:",
        f"{fresh_mae_clean:.8f}"
    )

    # ------------------------------------------------------------
    # HIDDEN ANSWER — REVEALED ONLY AFTER DISCOVERY
    # ------------------------------------------------------------

    print()
    print(
        "HIDDEN GENERATING RELATIONSHIP"
    )

    print(
        hidden_equation
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The discovery engine was NOT given the generating equation."
    )

    print(
        "It was NOT told which variables matter."
    )

    print(
        "It was NOT told the number of terms."
    )

    print(
        "It was NOT told which operations matter."
    )

    print()

    return {
        "name": name,
        "best": best,
        "features": features,
        "hidden_equation": hidden_equation,
        "fresh_rmse_clean": fresh_rmse_clean,
        "elapsed": elapsed,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("BLIND SCIENTIFIC LAW DISCOVERY SUITE")
    print("COMPLEXITY-AWARE SYMBOLIC REGRESSION")
    print("=" * 70)

    print()
    print(
        "The engine receives observations only."
    )

    print(
        "The generating equations remain hidden."
    )

    print()
    print(
        "Datasets:"
    )

    print(
        "  1. QUADRATIC"
    )
    print(
        "  2. MIXED"
    )
    print(
        "  3. INVERSE"
    )
    print(
        "  4. EXPONENTIAL"
    )
    print(
        "  5. TRIG"
    )

    print()

    datasets = [
        ("QUADRATIC", 100),
        ("MIXED", 200),
        ("INVERSE", 300),
        ("EXPONENTIAL", 400),
        ("TRIG", 500),
    ]

    results = []

    for name, seed in datasets:

        result = run_experiment(
            name,
            seed
        )

        results.append(
            result
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print()
    print("=" * 70)
    print("FINAL DISCOVERY SUMMARY")
    print("=" * 70)

    print()

    for result in results:

        best = result["best"]

        print(
            f"{result['name']:12s} | "
            f"Train {best['train_rmse']:.5f} | "
            f"Test {best['test_rmse']:.5f} | "
            f"Fresh {result['fresh_rmse_clean']:.5f} | "
            f"Terms {len(best['selected'])}"
        )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
