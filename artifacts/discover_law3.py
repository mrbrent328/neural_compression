import numpy as np
import itertools
import math
import time

# ================================================================
# BLIND MATHEMATICAL LAW DISCOVERY
# Complexity-aware symbolic regression experiment
#
# The discovery engine is NOT given:
#   - the generating equation
#   - which variables matter
#   - how many terms the equation contains
#   - which mathematical operations matter
#
# It IS given:
#   - observations of x, y, z, w
#   - corresponding observed output
#
# ================================================================


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

SEED = 42

TOTAL_OBSERVATIONS = 700
TRAIN_FRACTION = 0.70

NOISE_STD = 0.05

MAX_TERMS = 5

TOP_RESULTS = 25

# Complexity penalty.
#
# Larger values favor simpler equations.
# We will report several values so we can see
# how the preferred law changes.
#
COMPLEXITY_LAMBDAS = [
    0.00,
    0.01,
    0.03,
    0.05,
    0.10,
    0.20,
]

# Small ridge value for numerical stability.
RIDGE = 1e-8


# ------------------------------------------------
# HIDDEN DATA GENERATOR
# ------------------------------------------------

def generate_data(n, rng, noise=True):
    """
    IMPORTANT:
    The discovery engine never receives this function.
    """

    x = rng.uniform(-3.0, 3.0, n)
    y = rng.uniform(-3.0, 3.0, n)
    z = rng.uniform(-3.0, 3.0, n)
    w = rng.uniform(-3.0, 3.0, n)

    # Hidden generating relationship.
    #
    # DO NOT expose this to the discovery engine.
    clean_output = (
        2.75 * x**2
        - 1.50 * y
        + 0.80 * x * z
        + 3.20
    )

    if noise:
        output = clean_output + rng.normal(0.0, NOISE_STD, n)
    else:
        output = clean_output

    X = {
        "x": x,
        "y": y,
        "z": z,
        "w": w,
    }

    return X, output


# ------------------------------------------------
# SAFE MATHEMATICAL OPERATIONS
# ------------------------------------------------

def safe_divide(a, b):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.divide(a, b)

    result = np.where(np.isfinite(result), result, 0.0)
    result = np.clip(result, -1e6, 1e6)

    return result


def safe_log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(np.abs(a) + 1e-6)

    result = np.where(np.isfinite(result), result, 0.0)
    result = np.clip(result, -1e6, 1e6)

    return result


def safe_sqrt(a):
    with np.errstate(invalid="ignore"):
        result = np.sqrt(np.abs(a))

    result = np.where(np.isfinite(result), result, 0.0)

    return result


def safe_exp(a):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp(np.clip(a, -10, 10))

    result = np.where(np.isfinite(result), result, 0.0)

    return result


def safe_sin(a):
    result = np.sin(a)
    return np.where(np.isfinite(result), result, 0.0)


def safe_cos(a):
    result = np.cos(a)
    return np.where(np.isfinite(result), result, 0.0)


# ------------------------------------------------
# FEATURE OBJECT
# ------------------------------------------------

class Feature:
    def __init__(self, name, values, complexity):
        self.name = name
        self.values = np.asarray(values, dtype=np.float64)
        self.complexity = complexity

    def __repr__(self):
        return self.name


# ------------------------------------------------
# BUILD FEATURE LIBRARY
# ------------------------------------------------

def build_features(X):
    x = X["x"]
    y = X["y"]
    z = X["z"]
    w = X["w"]

    features = []

    def add(name, values, complexity):
        values = np.asarray(values, dtype=np.float64)

        if not np.all(np.isfinite(values)):
            return

        # Normalize extremely large features.
        scale = np.max(np.abs(values))

        if scale > 1e8:
            return

        features.append(
            Feature(name, values, complexity)
        )

    # ------------------------------------------------
    # CONSTANT
    # ------------------------------------------------

    add("1", np.ones_like(x), 1.0)

    variables = {
        "x": x,
        "y": y,
        "z": z,
        "w": w,
    }

    # ------------------------------------------------
    # BASIC VARIABLES
    # ------------------------------------------------

    for name, values in variables.items():
        add(name, values, 1.0)

    # ------------------------------------------------
    # POWERS
    # ------------------------------------------------

    for name, values in variables.items():
        add(f"{name}^2", values**2, 1.5)
        add(f"{name}^3", values**3, 2.0)

    # ------------------------------------------------
    # UNARY FUNCTIONS
    # ------------------------------------------------

    for name, values in variables.items():

        add(
            f"sqrt(|{name}|)",
            safe_sqrt(values),
            2.0
        )

        add(
            f"log(|{name}|)",
            safe_log(values),
            2.0
        )

        add(
            f"sin({name})",
            safe_sin(values),
            2.0
        )

        add(
            f"cos({name})",
            safe_cos(values),
            2.0
        )

        add(
            f"exp({name})",
            safe_exp(values),
            3.0
        )

    # ------------------------------------------------
    # PAIRWISE PRODUCTS
    # ------------------------------------------------

    names = list(variables.keys())

    for a, b in itertools.combinations(names, 2):

        av = variables[a]
        bv = variables[b]

        add(
            f"{a}*{b}",
            av * bv,
            2.0
        )

        add(
            f"({a}*{b})^2",
            (av * bv)**2,
            3.0
        )

    # ------------------------------------------------
    # SELECTED RATIOS
    # ------------------------------------------------

    for a, b in itertools.permutations(names, 2):

        values = safe_divide(
            variables[a],
            variables[b]
        )

        add(
            f"{a}/{b}",
            values,
            3.0
        )

    # ------------------------------------------------
    # COMBINED PRODUCTS WITH Z
    # ------------------------------------------------

    add(
        "x*z",
        x * z,
        2.0
    )

    add(
        "x*y",
        x * y,
        2.0
    )

    add(
        "x*w",
        x * w,
        2.0
    )

    add(
        "y*z",
        y * z,
        2.0
    )

    add(
        "y*w",
        y * w,
        2.0
    )

    add(
        "z*w",
        z * w,
        2.0
    )

    # ------------------------------------------------
    # REMOVE DUPLICATE FEATURES
    # ------------------------------------------------

    unique = []
    signatures = set()

    for feature in features:

        # Quantized signature prevents mathematically
        # duplicate expressions from appearing repeatedly.
        sample = feature.values[:100]

        signature = tuple(
            np.round(sample, 8)
        )

        if signature in signatures:
            continue

        signatures.add(signature)
        unique.append(feature)

    return unique


# ------------------------------------------------
# NORMALIZE FEATURES
# ------------------------------------------------

def normalize_feature(feature_values):

    values = np.asarray(
        feature_values,
        dtype=np.float64
    )

    scale = np.std(values)

    if scale < 1e-12:
        scale = 1.0

    normalized = values / scale

    return normalized, scale


# ------------------------------------------------
# FIT A SET OF TERMS
# ------------------------------------------------

def fit_terms(
    selected_features,
    train_y,
    test_y,
    train_indices,
    test_indices
):

    train_columns = []
    test_columns = []
    scales = []

    for feature in selected_features:

        train_values = feature.values[train_indices]
        test_values = feature.values[test_indices]

        normalized, scale = normalize_feature(
            train_values
        )

        test_normalized = test_values / scale

        train_columns.append(normalized)
        test_columns.append(test_normalized)

        scales.append(scale)

    A_train = np.column_stack(train_columns)
    A_test = np.column_stack(test_columns)

    # ------------------------------------------------
    # RIDGE / LEAST-SQUARES FIT
    # ------------------------------------------------

    ATA = A_train.T @ A_train
    ATy = A_train.T @ train_y

    regularization = RIDGE * np.eye(
        ATA.shape[0]
    )

    try:
        coefficients = np.linalg.solve(
            ATA + regularization,
            ATy
        )

    except np.linalg.LinAlgError:
        return None

    train_prediction = A_train @ coefficients
    test_prediction = A_test @ coefficients

    train_rmse = math.sqrt(
        np.mean(
            (train_y - train_prediction)**2
        )
    )

    test_rmse = math.sqrt(
        np.mean(
            (test_y - test_prediction)**2
        )
    )

    return {
        "features": selected_features,
        "coefficients": coefficients,
        "scales": scales,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }


# ------------------------------------------------
# FORMAT EQUATION
# ------------------------------------------------

def format_equation(result):

    pieces = []

    for coefficient, feature in zip(
        result["coefficients"],
        result["features"]
    ):

        # Convert normalized coefficient back
        # into coefficient for original feature.
        scale = result["scales"][
            len(pieces)
        ]

        real_coefficient = coefficient / scale

        if abs(real_coefficient) < 1e-7:
            continue

        sign = "+" if real_coefficient >= 0 else "-"

        magnitude = abs(real_coefficient)

        pieces.append(
            f"({sign}{magnitude:.8f})*{feature.name}"
        )

    if not pieces:
        return "0"

    return " ".join(pieces)


# ------------------------------------------------
# COMPLEXITY CALCULATION
# ------------------------------------------------

def equation_complexity(result):

    complexity = 0.0

    for coefficient, feature in zip(
        result["coefficients"],
        result["features"]
    ):

        scale = result["scales"][
            list(result["features"]).index(feature)
        ]

        real_coefficient = coefficient / scale

        # Base feature complexity.
        complexity += feature.complexity

        # Penalize tiny "cheating" terms less.
        magnitude = abs(real_coefficient)

        if magnitude < 1e-3:
            complexity += 0.1

        elif magnitude < 1e-2:
            complexity += 0.25

        elif magnitude < 0.1:
            complexity += 0.5

        else:
            complexity += 1.0

    # Explicit term-count penalty.
    complexity += len(
        result["features"]
    ) * 1.0

    return complexity


# ------------------------------------------------
# DISCOVERY
# ------------------------------------------------

def discover(
    features,
    train_y,
    test_y,
    train_indices,
    test_indices
):

    results = []

    total = len(features)

    print()
    print(
        f"Searching {total} mathematical features..."
    )

    for term_count in range(
        1,
        MAX_TERMS + 1
    ):

        print(
            f"Searching {term_count}-term equations..."
        )

        checked = 0

        for combination in itertools.combinations(
            features,
            term_count
        ):

            checked += 1

            result = fit_terms(
                combination,
                train_y,
                test_y,
                train_indices,
                test_indices
            )

            if result is None:
                continue

            complexity = equation_complexity(
                result
            )

            result["complexity"] = complexity

            results.append(result)

        print(
            f"combinations checked: {checked}"
        )

    return results


# ------------------------------------------------
# SCORE RESULTS
# ------------------------------------------------

def score_result(
    result,
    complexity_lambda
):

    return (
        result["train_rmse"]
        + complexity_lambda
        * result["complexity"]
    )


# ------------------------------------------------
# FRESH DATA TEST
# ------------------------------------------------

def evaluate_fresh(
    result,
    rng
):

    fresh_X, fresh_y = generate_data(
        500,
        rng,
        noise=False
    )

    prediction = np.zeros_like(
        fresh_y,
        dtype=np.float64
    )

    for coefficient, feature, scale in zip(
        result["coefficients"],
        result["features"],
        result["scales"]
    ):

        name = feature.name

        # Rebuild feature using fresh data.
        fresh_features = build_features(
            fresh_X
        )

        matching = None

        for candidate in fresh_features:
            if candidate.name == name:
                matching = candidate
                break

        if matching is None:
            continue

        prediction += (
            coefficient
            * matching.values
            / scale
        )

    rmse = math.sqrt(
        np.mean(
            (fresh_y - prediction)**2
        )
    )

    mae = np.mean(
        np.abs(
            fresh_y - prediction
        )
    )

    return rmse, mae


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    rng = np.random.default_rng(
        SEED
    )

    print("=" * 70)
    print("BLIND SCIENTIFIC LAW DISCOVERY")
    print("COMPLEXITY-AWARE SYMBOLIC REGRESSION")
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

    # ------------------------------------------------
    # GENERATE DATA
    # ------------------------------------------------

    X, output = generate_data(
        TOTAL_OBSERVATIONS,
        rng,
        noise=True
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

    train_y = output[
        train_indices
    ]

    test_y = output[
        test_indices
    ]

    print()
    print(
        f"Total observations:       {TOTAL_OBSERVATIONS}"
    )

    print(
        f"Training observations:    {len(train_indices)}"
    )

    print(
        f"Held-out observations:    {len(test_indices)}"
    )

    print(
        f"Measurement noise:        {NOISE_STD}"
    )

    # ------------------------------------------------
    # BUILD FEATURES
    # ------------------------------------------------

    features = build_features(X)

    print(
        f"Mathematical features:    {len(features)}"
    )

    print(
        f"Searching term counts:    1 through {MAX_TERMS}"
    )

    # ------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------

    start = time.time()

    results = discover(
        features,
        train_y,
        test_y,
        train_indices,
        test_indices
    )

    discovery_time = (
        time.time() - start
    )

    print()
    print(
        f"Discovery time: {discovery_time:.2f} seconds"
    )

    # ------------------------------------------------
    # SORT BY PURE TRAINING ERROR
    # ------------------------------------------------

    raw_sorted = sorted(
        results,
        key=lambda r: r["train_rmse"]
    )

    # ------------------------------------------------
    # SHOW RAW BEST
    # ------------------------------------------------

    print()
    print("=" * 70)
    print("BEST BY PREDICTION ERROR ONLY")
    print("=" * 70)

    print()

    print(
        "Rank  Train RMSE  Test RMSE  Complexity  Terms  Equation"
    )

    for rank, result in enumerate(
        raw_sorted[:10],
        1
    ):

        print(
            f"{rank:4d}  "
            f"{result['train_rmse']:10.6f}  "
            f"{result['test_rmse']:9.6f}  "
            f"{result['complexity']:10.3f}  "
            f"{len(result['features']):5d}  "
            f"{format_equation(result)}"
        )

    # ------------------------------------------------
    # COMPLEXITY-AWARE RESULTS
    # ------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLEXITY-AWARE DISCOVERY")
    print("=" * 70)

    for lam in COMPLEXITY_LAMBDAS:

        ranked = sorted(
            results,
            key=lambda r: score_result(
                r,
                lam
            )
        )

        best = ranked[0]

        score = score_result(
            best,
            lam
        )

        print()
        print(
            f"Lambda = {lam:.3f}"
        )

        print(
            f"Score:       {score:.8f}"
        )

        print(
            f"Train RMSE:  {best['train_rmse']:.8f}"
        )

        print(
            f"Test RMSE:   {best['test_rmse']:.8f}"
        )

        print(
            f"Complexity:  {best['complexity']:.4f}"
        )

        print(
            f"Terms:       {len(best['features'])}"
        )

        print(
            "Equation:"
        )

        print(
            format_equation(best)
        )

    # ------------------------------------------------
    # PICK A REASONABLE DEFAULT
    # ------------------------------------------------

    # 0.05 is intentionally moderate.
    DEFAULT_LAMBDA = 0.05

    ranked = sorted(
        results,
        key=lambda r: score_result(
            r,
            DEFAULT_LAMBDA
        )
    )

    best = ranked[0]

    # ------------------------------------------------
    # FRESH DATA
    # ------------------------------------------------

    fresh_rmse, fresh_mae = evaluate_fresh(
        best,
        rng
    )

    print()
    print("=" * 70)
    print("SELECTED COMPLEXITY-AWARE MODEL")
    print("=" * 70)

    print()
    print(
        f"Complexity penalty lambda: {DEFAULT_LAMBDA}"
    )

    print(
        f"Training RMSE:   {best['train_rmse']:.8f}"
    )

    print(
        f"Held-out RMSE:   {best['test_rmse']:.8f}"
    )

    print(
        f"Complexity:      {best['complexity']:.4f}"
    )

    print(
        f"Terms:           {len(best['features'])}"
    )

    print()
    print(
        "Discovered relationship:"
    )

    print(
        format_equation(best)
    )

    print()
    print(
        f"Fresh-data RMSE: {fresh_rmse:.8f}"
    )

    print(
        f"Fresh-data MAE:  {fresh_mae:.8f}"
    )

    # ------------------------------------------------
    # VARIABLE USAGE
    # ------------------------------------------------

    used_variables = set()

    for feature in best["features"]:

        name = feature.name

        for variable in [
            "x",
            "y",
            "z",
            "w"
        ]:

            if variable in name:
                used_variables.add(
                    variable
                )

    print()
    print(
        "Variables appearing in discovered law:"
    )

    print(
        ", ".join(
            sorted(used_variables)
        )
    )

    print()
    print(
        "NOTE: The discovery engine was never told"
    )

    print(
        "that w was irrelevant."
    )

    print()
    print("=" * 70)
    print("HIDDEN LAW — FOR HUMAN VERIFICATION ONLY")
    print("=" * 70)

    print()
    print(
        "output = 2.75*x^2 - 1.50*y + 0.80*x*z + 3.20"
    )

    print()
    print(
        "The equation above was NOT provided to the"
    )

    print(
        "discovery engine."
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()