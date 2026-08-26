import numpy as np
import itertools
import math

# ============================================================
# SCIENTIFIC LAW DISCOVERY — EXPERIMENT 1
#
# The discovery system is NOT given the generating equation.
#
# It receives:
#       time -> observed position
#
# It searches for a compact mathematical relationship that
# predicts unseen observations.
#
# The hidden equation is used ONLY at the very end to evaluate
# whether the experiment succeeded.
# ============================================================


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

N_TOTAL = 250
TRAIN_FRACTION = 0.70

NOISE_STD = 0.02

MAX_TERMS = 2
MAX_POWER = 3

RANDOM_TEST_POINTS = 500


# ============================================================
# HIDDEN PHYSICS
#
# IMPORTANT:
#
# The discovery algorithm never receives this function.
#
# This represents the "law of nature" producing the observations.
# ============================================================

def hidden_physics(t):
    """
    Hidden generating relationship.

    The discovery system does NOT see this function.
    """
    g = 9.81
    return 0.5 * g * t**2


# ============================================================
# DATA GENERATION
# ============================================================

def generate_data():
    rng = np.random.default_rng(SEED)

    # Experimental observations
    t = rng.uniform(0.0, 3.0, N_TOTAL)

    true_position = hidden_physics(t)

    # Simulated measurement noise
    measured_position = (
        true_position
        + rng.normal(0.0, NOISE_STD, N_TOTAL)
    )

    # Randomly divide observations into training/testing
    indices = np.arange(N_TOTAL)
    rng.shuffle(indices)

    split = int(N_TOTAL * TRAIN_FRACTION)

    train_idx = indices[:split]
    test_idx = indices[split:]

    return (
        t[train_idx],
        measured_position[train_idx],
        t[test_idx],
        measured_position[test_idx]
    )


# ============================================================
# MATHEMATICAL BUILDING BLOCKS
#
# The discovery system is allowed to combine these.
#
# We intentionally do NOT include "gravity" or any physics
# variable here.
# ============================================================

def build_library(x):
    return {
        "1": np.ones_like(x),

        "x": x,
        "x^2": x**2,
        "x^3": x**3,

        "1/x": 1.0 / np.maximum(x, 1e-8),

        "sqrt(x)": np.sqrt(np.maximum(x, 0.0)),

        "sin(x)": np.sin(x),
        "cos(x)": np.cos(x),

        "log(x)": np.log(np.maximum(x, 1e-8)),

        "exp(x)": np.exp(np.clip(x, -10, 10)),
    }


# ============================================================
# CANDIDATE EQUATION
# ============================================================

class Candidate:
    def __init__(self, names, X, coefficients):
        self.names = names
        self.X = X
        self.coefficients = coefficients

    def predict(self):
        return self.X @ self.coefficients

    def complexity(self):
        """
        Very simple complexity penalty.

        Fewer terms = simpler equation.
        """
        return len(self.names)

    def description(self):
        pieces = []

        for name, coefficient in zip(
            self.names,
            self.coefficients
        ):
            if abs(coefficient) < 1e-10:
                continue

            pieces.append(
                f"({coefficient:+.6f})*{name}"
            )

        if not pieces:
            return "0"

        return " ".join(pieces)


# ============================================================
# FIT LINEAR COMBINATIONS OF BASIS FUNCTIONS
#
# This lets us search things such as:
#
# position = a*x² + b*x
#
# without knowing beforehand which terms matter.
# ============================================================

def fit_candidate(names, library, y):

    X = np.column_stack([
        library[name]
        for name in names
    ])

    try:
        coefficients, *_ = np.linalg.lstsq(
            X,
            y,
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    prediction = X @ coefficients

    mse = np.mean(
        (prediction - y) ** 2
    )

    return Candidate(
        names,
        X,
        coefficients
    ), mse


# ============================================================
# DISCOVERY SEARCH
# ============================================================

def discover_law(x_train, y_train):

    library = build_library(x_train)

    names = list(library.keys())

    results = []

    print()
    print("=" * 70)
    print("SEARCHING FOR MATHEMATICAL RELATIONSHIPS")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # One-term models
    # --------------------------------------------------------

    for name in names:

        result = fit_candidate(
            [name],
            library,
            y_train
        )

        if result is not None:

            candidate, mse = result

            results.append(
                (mse, candidate.complexity(), candidate)
            )

    # --------------------------------------------------------
    # Two-term models
    # --------------------------------------------------------

    if MAX_TERMS >= 2:

        for a, b in itertools.combinations(
            names,
            2
        ):

            result = fit_candidate(
                [a, b],
                library,
                y_train
            )

            if result is not None:

                candidate, mse = result

                results.append(
                    (mse, candidate.complexity(), candidate)
                )

    # --------------------------------------------------------
    # Three-term models
    # --------------------------------------------------------

    if MAX_TERMS >= 3:

        for combo in itertools.combinations(
            names,
            3
        ):

            result = fit_candidate(
                list(combo),
                library,
                y_train
            )

            if result is not None:

                candidate, mse = result

                results.append(
                    (mse, candidate.complexity(), candidate)
                )

    # --------------------------------------------------------
    # Sort primarily by training error, secondarily by
    # complexity.
    # --------------------------------------------------------

    results.sort(
        key=lambda r: (
            r[0],
            r[1]
        )
    )

    return results


# ============================================================
# EVALUATE ON DATA THE DISCOVERY SYSTEM NEVER SAW
# ============================================================

def evaluate(candidate, x_test, y_test):

    library = build_library(x_test)

    X = np.column_stack([
        library[name]
        for name in candidate.names
    ])

    prediction = X @ candidate.coefficients

    mse = np.mean(
        (prediction - y_test) ** 2
    )

    rmse = math.sqrt(mse)

    mae = np.mean(
        np.abs(prediction - y_test)
    )

    return mse, rmse, mae


# ============================================================
# PRINT TOP DISCOVERIES
# ============================================================

def show_candidates(results, x_test, y_test, count=15):

    print()
    print("=" * 70)
    print("TOP CANDIDATE RELATIONSHIPS")
    print("=" * 70)
    print()

    print(
        f"{'Rank':<6}"
        f"{'Train RMSE':<15}"
        f"{'Test RMSE':<15}"
        f"{'Terms':<8}"
        f"Equation"
    )

    print("-" * 70)

    for rank, (train_mse, complexity, candidate) in enumerate(
        results[:count],
        start=1
    ):

        train_rmse = math.sqrt(train_mse)

        test_mse, test_rmse, test_mae = evaluate(
            candidate,
            x_test,
            y_test
        )

        print(
            f"{rank:<6}"
            f"{train_rmse:<15.6f}"
            f"{test_rmse:<15.6f}"
            f"{complexity:<8}"
            f"{candidate.description()}"
        )


# ============================================================
# RANDOM GENERALIZATION TEST
#
# This is separate from the held-out observations.
#
# We generate completely new points and see whether the
# discovered equation continues to work.
# ============================================================

def generalization_test(candidate):

    rng = np.random.default_rng(
        SEED + 1000
    )

    t = rng.uniform(
        0.0,
        3.0,
        RANDOM_TEST_POINTS
    )

    actual = hidden_physics(t)

    library = build_library(t)

    X = np.column_stack([
        library[name]
        for name in candidate.names
    ])

    predicted = X @ candidate.coefficients

    mse = np.mean(
        (predicted - actual) ** 2
    )

    rmse = math.sqrt(mse)

    return rmse


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SCIENTIFIC LAW DISCOVERY EXPERIMENT")
    print("=" * 70)

    print()
    print("The discovery system is NOT given the generating equation.")
    print()

    # --------------------------------------------------------
    # Generate observations
    # --------------------------------------------------------

    (
        x_train,
        y_train,
        x_test,
        y_test
    ) = generate_data()

    print(
        f"Total observations: {N_TOTAL}"
    )

    print(
        f"Training observations: {len(x_train)}"
    )

    print(
        f"Held-out observations: {len(x_test)}"
    )

    print(
        f"Measurement noise: {NOISE_STD}"
    )

    # --------------------------------------------------------
    # DISCOVER
    # --------------------------------------------------------

    results = discover_law(
        x_train,
        y_train
    )

    # --------------------------------------------------------
    # SHOW RESULTS
    # --------------------------------------------------------

    show_candidates(
        results,
        x_test,
        y_test
    )

    # --------------------------------------------------------
    # BEST CANDIDATE
    # --------------------------------------------------------

    best = results[0][2]

    print()
    print("=" * 70)
    print("BEST DISCOVERED RELATIONSHIP")
    print("=" * 70)
    print()

    print(
        best.description()
    )

    train_mse = results[0][0]

    test_mse, test_rmse, test_mae = evaluate(
        best,
        x_test,
        y_test
    )

    print()
    print(
        f"Training RMSE: {math.sqrt(train_mse):.8f}"
    )

    print(
        f"Held-out RMSE: {test_rmse:.8f}"
    )

    print(
        f"Held-out MAE:  {test_mae:.8f}"
    )

    # --------------------------------------------------------
    # GENERALIZATION
    # --------------------------------------------------------

    generalization_rmse = generalization_test(
        best
    )

    print()
    print(
        f"Fresh-data RMSE: {generalization_rmse:.8f}"
    )

    # --------------------------------------------------------
    # HUMAN INTERPRETATION
    #
    # This section is deliberately AFTER discovery.
    #
    # The discovery system itself never receives this equation.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("POST-EXPERIMENT VERIFICATION")
    print("=" * 70)
    print()

    print(
        "Known generating relationship (hidden from discovery):"
    )

    print(
        "position = 4.905 * time^2"
    )

    print()

    print(
        "If the discovered equation is approximately:"
    )

    print(
        "position ≈ 4.905 * time^2"
    )

    print(
        "then the system successfully rediscovered"
    )

    print(
        "the underlying relationship."
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()