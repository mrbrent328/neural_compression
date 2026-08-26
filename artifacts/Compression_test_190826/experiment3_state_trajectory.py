#!/usr/bin/env python3
"""
Experiment 3: State-Trajectory Equivalence

Question:
    Is the useful information in a neural compressor's evolving STATE
    rather than in any single latent representation?

We deliberately avoid rotating a latent vector.

Instead, we train two small online predictors from the same initial state
while processing the same structured byte stream:

    stream -> state_trajectory_A -> predictions
    stream -> state_trajectory_B -> predictions

Then we test whether different representations of the same trajectory can
produce the same future behavior.

The key experiment:
    1. Run a reference model over a prefix and save its state trajectory.
    2. Create an equivalent representation by applying an invertible
       coordinate transform to the hidden state AND transform the model's
       weights so that its predictions remain mathematically identical.
    3. Continue both models on the same future bytes.
    4. Compare predictions and final states.

This tests a stronger version of the "different angle" idea:
the coordinates may change, while the state-transition behavior remains
invariant.

A second control applies the same hidden-state rotation WITHOUT transforming
the downstream weights. That should fail.

Dependencies: numpy
"""

import hashlib
import math
import numpy as np


SEED = 42
VOCAB = 16
EMBED = 8
HIDDEN = 16
CONTEXT = 4
STREAM_LEN = 2400
PREFIX = 1200
LR = 0.025


def make_stream():
    """Structured byte-like symbols with deterministic relationships."""
    x = np.zeros(STREAM_LEN, dtype=np.int64)

    # Seed.
    x[:CONTEXT] = [1, 3, 5, 7]

    for i in range(CONTEXT, STREAM_LEN):
        a = x[i - 1]
        b = x[i - 2]
        c = x[i - 4]

        # A deterministic nonlinear-ish rule with a little periodic structure.
        x[i] = (a + 2 * b + 3 * c + (i % 5)) % VOCAB

    return x


class OnlinePredictor:
    """
    Embedding -> hidden -> logits.

    The hidden state is deliberately exposed so we can transform its
    coordinates and correspondingly transform the output weights.
    """

    def __init__(self, rng):
        self.E = rng.normal(0, 0.20, (VOCAB, EMBED))
        self.W = rng.normal(0, 0.15, (EMBED * CONTEXT, HIDDEN))
        self.b = np.zeros(HIDDEN)

        self.V = rng.normal(0, 0.15, (HIDDEN, VOCAB))
        self.c = np.zeros(VOCAB)

    def copy(self):
        other = object.__new__(OnlinePredictor)
        other.E = self.E.copy()
        other.W = self.W.copy()
        other.b = self.b.copy()
        other.V = self.V.copy()
        other.c = self.c.copy()
        return other

    def hidden(self, ctx):
        emb = self.E[ctx].reshape(-1)
        return np.tanh(emb @ self.W + self.b)

    def logits(self, ctx):
        h = self.hidden(ctx)
        return h @ self.V + self.c

    def step(self, ctx, target):
        """
        Online learning step, similar in spirit to the user's compressor:
        predict -> observe target -> update model state.
        """
        emb = self.E[ctx].reshape(-1)
        pre = emb @ self.W + self.b
        h = np.tanh(pre)
        logits = h @ self.V + self.c

        z = logits - np.max(logits)
        p = np.exp(z)
        p /= p.sum()

        loss = -math.log(max(p[target], 1e-15))

        dlogits = p
        dlogits[target] -= 1.0

        dV = np.outer(h, dlogits)
        dc = dlogits

        dh = dlogits @ self.V.T
        dpre = dh * (1.0 - h * h)

        dW = np.outer(emb, dpre)
        db = dpre

        demb = dpre @ self.W.T
        dE = demb.reshape(CONTEXT, EMBED)

        self.V -= LR * dV
        self.c -= LR * dc
        self.W -= LR * dW
        self.b -= LR * db

        for pos, idx in enumerate(ctx):
            self.E[idx] -= LR * dE[pos]

        return loss, logits, h


def softmax(logits):
    z = logits - np.max(logits)
    p = np.exp(z)
    p /= p.sum()
    return p


def run_prefix(model, stream, end):
    losses = []
    predictions = []
    states = []

    for i in range(CONTEXT, end):
        ctx = stream[i - CONTEXT:i]
        loss, logits, h = model.step(ctx, int(stream[i]))
        losses.append(loss)
        predictions.append(int(np.argmax(logits)))
        states.append(h.copy())

    return np.array(losses), predictions, np.array(states)


def transform_hidden_basis(model, R):
    """
    Change hidden coordinates from h to h' = h R.

    For logits to remain identical:

        h V + c
        =
        h R R.T V + c

    Therefore the transformed output weights are:

        V' = R.T V

    The hidden-producing W/b are transformed so that the tanh activation
    remains coordinate-wise meaningful only for signed permutations. For a
    general rotation, this is NOT exactly representable through tanh.

    Therefore this function is used with signed permutation matrices for the
    exact-equivalence test.
    """
    out = model.copy()

    # h' = h R, where R is a signed permutation.
    # Because tanh commutes with signed permutations:
    # tanh(x R) = tanh(x) R.
    out.W = out.W @ R
    out.b = out.b @ R
    out.V = R.T @ out.V

    return out


def signed_permutation(dim, rng):
    p = rng.permutation(dim)
    signs = rng.choice([-1.0, 1.0], size=dim)
    R = np.zeros((dim, dim))
    for i, j in enumerate(p):
        R[j, i] = signs[i]
    return R


def hash_model(model):
    h = hashlib.sha256()
    for arr in [model.E, model.W, model.b, model.V, model.c]:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def max_abs_model_difference(a, b):
    vals = [
        np.max(np.abs(a.E - b.E)),
        np.max(np.abs(a.W - b.W)),
        np.max(np.abs(a.b - b.b)),
        np.max(np.abs(a.V - b.V)),
        np.max(np.abs(a.c - b.c)),
    ]
    return float(max(vals))


def main():
    rng = np.random.default_rng(SEED)
    stream = make_stream()

    print("=" * 72)
    print("EXPERIMENT 3: STATE-TRAJECTORY EQUIVALENCE")
    print("=" * 72)
    print(f"Vocabulary       : {VOCAB}")
    print(f"Hidden dimension : {HIDDEN}")
    print(f"Stream length    : {STREAM_LEN}")
    print(f"Prefix           : {PREFIX}")
    print()

    # ---------------------------------------------------------------
    # A. Reference model learns the prefix.
    # ---------------------------------------------------------------
    reference = OnlinePredictor(rng)

    print("[1] Learning the reference trajectory...")
    prefix_losses, _, prefix_states = run_prefix(reference, stream, PREFIX)

    print(f"Prefix mean loss : {prefix_losses.mean():.6f}")
    print(f"Final prefix loss: {prefix_losses[-1]:.6f}")
    print(f"Reference hash   : {hash_model(reference)}")

    # ---------------------------------------------------------------
    # B. Create an alternate coordinate system for the SAME state.
    #
    # We use a signed permutation because tanh makes arbitrary rotations
    # nonlinear. This gives us an exact coordinate transformation while
    # still changing the numerical representation substantially.
    # ---------------------------------------------------------------
    R = signed_permutation(HIDDEN, rng)

    transformed = transform_hidden_basis(reference, R)

    print()
    print("[2] Creating alternate state coordinates...")
    print(f"R.T @ R error   : {np.max(np.abs(R.T @ R - np.eye(HIDDEN))):.3e}")
    print(
        "Reference and transformed parameter bytes identical? "
        f"{hash_model(reference) == hash_model(transformed)}"
    )

    # Check that predictions are identical before continuing.
    max_logit_diff = 0.0
    for i in range(PREFIX, min(PREFIX + 100, STREAM_LEN)):
        ctx = stream[i - CONTEXT:i]
        a = reference.logits(ctx)
        b = transformed.logits(ctx)
        max_logit_diff = max(max_logit_diff, float(np.max(np.abs(a - b))))

    print(f"Max pre-continuation logit difference: {max_logit_diff:.3e}")

    # ---------------------------------------------------------------
    # C. Continue both models on identical future data.
    # ---------------------------------------------------------------
    print()
    print("[3] Continuing BOTH state trajectories on identical future data...")

    future_a = reference.copy()
    future_b = transformed.copy()

    losses_a = []
    losses_b = []
    pred_matches = 0
    total = 0
    max_logit_diff_future = 0.0

    for i in range(PREFIX, STREAM_LEN):
        ctx = stream[i - CONTEXT:i]
        target = int(stream[i])

        loss_a, logits_a, h_a = future_a.step(ctx, target)
        loss_b, logits_b, h_b = future_b.step(ctx, target)

        losses_a.append(loss_a)
        losses_b.append(loss_b)

        pred_matches += int(np.argmax(logits_a) == np.argmax(logits_b))
        total += 1

        max_logit_diff_future = max(
            max_logit_diff_future,
            float(np.max(np.abs(logits_a - logits_b))),
        )

    prediction_agreement = pred_matches / total
    mean_loss_diff = float(abs(np.mean(losses_a) - np.mean(losses_b)))
    transformed_state = future_b.hidden(stream[STREAM_LEN-CONTEXT:STREAM_LEN])
    reference_state = future_a.hidden(stream[STREAM_LEN-CONTEXT:STREAM_LEN])

    # The state vectors should differ in coordinates, but be equivalent
    # under the known transformation.
    expected_transformed = reference_state @ R
    state_equivalence_error = float(
        np.max(np.abs(expected_transformed - transformed_state))
    )

    print(f"Future mean loss A        : {np.mean(losses_a):.8f}")
    print(f"Future mean loss B        : {np.mean(losses_b):.8f}")
    print(f"Mean loss difference      : {mean_loss_diff:.3e}")
    print(f"Prediction agreement      : {prediction_agreement * 100:.2f}%")
    print(f"Max future logit difference: {max_logit_diff_future:.3e}")
    print(f"State coordinate error    : {state_equivalence_error:.3e}")

    # ---------------------------------------------------------------
    # D. CONTROL: rotate the hidden representation but DON'T rotate
    # downstream weights. This should break behavior.
    # ---------------------------------------------------------------
    print()
    print("[4] Destructive control: same coordinate change WITHOUT compensating weights")

    broken = reference.copy()
    broken.W = broken.W @ R
    broken.b = broken.b @ R
    # V intentionally left untouched.

    broken_matches = 0
    broken_total = 0

    for i in range(PREFIX, STREAM_LEN):
        ctx = stream[i - CONTEXT:i]
        target = int(stream[i])

        _, logits_good, _ = reference.step(ctx, target)
        _, logits_broken, _ = broken.step(ctx, target)

        broken_matches += int(
            np.argmax(logits_good) == np.argmax(logits_broken)
        )
        broken_total += 1

    broken_agreement = broken_matches / broken_total

    print(f"Broken-control prediction agreement: {broken_agreement * 100:.2f}%")

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    if (
        max_logit_diff < 1e-10
        and max_logit_diff_future < 1e-8
        and state_equivalence_error < 1e-8
        and prediction_agreement == 1.0
        and broken_agreement < 1.0
    ):
        print("PASS: two different state representations produced identical behavior.")
        print()
        print(
            "This does NOT prove that arbitrary rotations are equivalent. "
            "It demonstrates something narrower and important:"
        )
        print(
            "the numerical coordinates of a model state can change while "
            "the state-transition behavior remains invariant, provided the "
            "whole representation is transformed consistently."
        )
    else:
        print("INCONCLUSIVE / FAIL: the expected exact equivalence was not observed.")
        print()
        print(
            "Check the numerical differences above. The signed-permutation "
            "construction should be exactly equivalent up to floating-point "
            "roundoff."
        )


if __name__ == "__main__":
    main()
