#!/usr/bin/env python3
"""
Experiment 5: Generalized + Functional State Compression

This experiment fixes the two major confounds from Experiment 4:

1. The trajectory compressor is trained on MANY independent trajectories,
   then evaluated on trajectories it has never seen.
2. We distinguish exact numerical state reconstruction from FUNCTIONAL
   reconstruction: whether the reconstructed state behaves like the original
   on unseen future inputs.

We test several latent sizes:
    4, 8, 16, 32

For each latent size we report:
    - state RMSE
    - future loss
    - future prediction agreement
    - per-instance latent storage
    - shared decoder size
    - total one-off package size

The shared trajectory decoder is counted separately from the per-instance
latent. This lets us distinguish:
    "cheap additional instances after the machinery exists"
from:
    "cheap compression including all machinery."

Important:
This is a research probe, not a production compressor.

Dependencies:
    numpy
"""

import math
import hashlib
import numpy as np


SEED = 42

VOCAB = 16
EMBED = 8
HIDDEN = 16
CONTEXT = 4

STATE_DIM = (
    VOCAB * EMBED
    + EMBED * CONTEXT * HIDDEN
    + HIDDEN
    + HIDDEN * VOCAB
    + VOCAB
)

STREAM_PREFIX = 700
STREAM_FUTURE = 500

TRAIN_TRAJECTORIES = 40
TEST_TRAJECTORIES = 10

LATENT_SIZES = [4, 8, 16, 32]

TRAIN_STEPS = 1200
LR = 0.01
MODEL_LR = 0.025


def make_stream(seed, length):
    """Generate a family of related but different structured streams."""
    rng = np.random.default_rng(seed)

    x = np.zeros(length, dtype=np.int64)

    # Every trajectory gets a different starting condition.
    x[:CONTEXT] = rng.integers(0, VOCAB, size=CONTEXT)

    # A trajectory-specific rule parameter.
    phase = int(rng.integers(0, VOCAB))

    for i in range(CONTEXT, length):
        a = x[i - 1]
        b = x[i - 2]
        c = x[i - 4]

        x[i] = (
            a
            + 2 * b
            + 3 * c
            + ((i + phase) % 7)
        ) % VOCAB

    return x


class OnlinePredictor:
    """Small online-learning predictor."""

    def __init__(self, rng):
        self.E = rng.normal(0, 0.20, (VOCAB, EMBED))
        self.W = rng.normal(0, 0.15, (EMBED * CONTEXT, HIDDEN))
        self.b = np.zeros(HIDDEN)

        self.V = rng.normal(0, 0.15, (HIDDEN, VOCAB))
        self.c = np.zeros(VOCAB)

    def state_vector(self):
        return np.concatenate(
            [
                self.E.ravel(),
                self.W.ravel(),
                self.b.ravel(),
                self.V.ravel(),
                self.c.ravel(),
            ]
        )

    def load_state_vector(self, state):
        p = 0

        n = self.E.size
        self.E = state[p:p+n].reshape(self.E.shape).copy()
        p += n

        n = self.W.size
        self.W = state[p:p+n].reshape(self.W.shape).copy()
        p += n

        n = self.b.size
        self.b = state[p:p+n].reshape(self.b.shape).copy()
        p += n

        n = self.V.size
        self.V = state[p:p+n].reshape(self.V.shape).copy()
        p += n

        n = self.c.size
        self.c = state[p:p+n].reshape(self.c.shape).copy()

    def copy(self):
        out = object.__new__(OnlinePredictor)
        out.E = self.E.copy()
        out.W = self.W.copy()
        out.b = self.b.copy()
        out.V = self.V.copy()
        out.c = self.c.copy()
        return out

    def step(self, ctx, target):
        emb = self.E[ctx].reshape(-1)

        pre = emb @ self.W + self.b
        h = np.tanh(pre)
        logits = h @ self.V + self.c

        z = logits - np.max(logits)
        p = np.exp(z)
        p /= p.sum()

        loss = -math.log(max(p[target], 1e-15))

        dlogits = p.copy()
        dlogits[target] -= 1.0

        dV = np.outer(h, dlogits)
        dc = dlogits

        dh = dlogits @ self.V.T
        dpre = dh * (1.0 - h * h)

        dW = np.outer(emb, dpre)
        db = dpre

        demb = dpre @ self.W.T
        dE = demb.reshape(CONTEXT, EMBED)

        self.V -= MODEL_LR * dV
        self.c -= MODEL_LR * dc
        self.W -= MODEL_LR * dW
        self.b -= MODEL_LR * db

        for pos, idx in enumerate(ctx):
            self.E[idx] -= MODEL_LR * dE[pos]

        return loss, logits

    def predict_without_update(self, ctx):
        emb = self.E[ctx].reshape(-1)
        h = np.tanh(emb @ self.W + self.b)
        return h @ self.V + self.c


def generate_trajectory(seed):
    """
    Create one independent model/stream trajectory.

    All trajectories use the same architecture but different initial model
    weights and different data streams.
    """
    model_rng = np.random.default_rng(seed + 100000)
    model = OnlinePredictor(model_rng)
    stream = make_stream(seed, STREAM_PREFIX + STREAM_FUTURE)

    losses = []
    states = []

    for i in range(CONTEXT, STREAM_PREFIX):
        ctx = stream[i-CONTEXT:i]
        loss, _ = model.step(ctx, int(stream[i]))

        losses.append(loss)
        states.append(model.state_vector())

    return {
        "stream": stream,
        "states": np.array(states),
        "final_state": states[-1],
        "losses": np.array(losses),
    }


def future_behavior(state, stream):
    model = OnlinePredictor(np.random.default_rng(123456))
    model.load_state_vector(state)

    losses = []
    predictions = []

    for i in range(STREAM_PREFIX, len(stream)):
        ctx = stream[i-CONTEXT:i]
        target = int(stream[i])

        loss, logits = model.step(ctx, target)

        losses.append(loss)
        predictions.append(int(np.argmax(logits)))

    return np.array(losses), np.array(predictions)


class StateAutoencoder:
    """
    Shared encoder/decoder.

    Input is a final state vector.
    Output is either:
      - reconstructed state (state objective), or
      - a state used later for behavioral evaluation.

    We train the state reconstruction objective because it provides a stable,
    observable target. Functional behavior is evaluated separately on
    held-out trajectories.
    """

    def __init__(self, latent_dim, rng):
        hidden = 128

        self.W1 = rng.normal(0, 0.01, (STATE_DIM, hidden))
        self.b1 = np.zeros(hidden)

        self.W2 = rng.normal(0, 0.01, (hidden, latent_dim))
        self.b2 = np.zeros(latent_dim)

        self.W3 = rng.normal(0, 0.01, (latent_dim, hidden))
        self.b3 = np.zeros(hidden)

        self.W4 = rng.normal(0, 0.01, (hidden, STATE_DIM))
        self.b4 = np.zeros(STATE_DIM)

        self.hidden = hidden

    def encode(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decode(self, z):
        h = np.tanh(z @ self.W3 + self.b3)
        return h @ self.W4 + self.b4

    def parameter_bytes(self):
        return sum(
            a.nbytes
            for a in [
                self.W1, self.b1,
                self.W2, self.b2,
                self.W3, self.b3,
                self.W4, self.b4,
            ]
        )


def train_autoencoder(states, latent_dim, rng):
    """
    Train on MANY trajectories.

    State normalization is global across the training set and therefore is
    part of the shared decoder package.
    """
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-6

    x = (states - mean) / std

    model = StateAutoencoder(latent_dim, rng)

    n = len(x)

    for step in range(TRAIN_STEPS):
        h1 = np.tanh(x @ model.W1 + model.b1)
        z = h1 @ model.W2 + model.b2

        h2 = np.tanh(z @ model.W3 + model.b3)
        y = h2 @ model.W4 + model.b4

        dy = (2.0 / n) * (y - x)

        dW4 = h2.T @ dy
        db4 = dy.sum(axis=0)

        dh2 = dy @ model.W4.T
        dh2_pre = dh2 * (1.0 - h2 * h2)

        dW3 = z.T @ dh2_pre
        db3 = dh2_pre.sum(axis=0)

        dz = dh2_pre @ model.W3.T

        dW2 = h1.T @ dz
        db2 = dz.sum(axis=0)

        dh1 = dz @ model.W2.T
        dh1_pre = dh1 * (1.0 - h1 * h1)

        dW1 = x.T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)

        model.W4 -= LR * dW4
        model.b4 -= LR * db4
        model.W3 -= LR * dW3
        model.b3 -= LR * db3
        model.W2 -= LR * dW2
        model.b2 -= LR * db2
        model.W1 -= LR * dW1
        model.b1 -= LR * db1

        if step % 300 == 0 or step == TRAIN_STEPS - 1:
            loss = float(np.mean((y - x) ** 2))
            print(
                f"    step {step:4d} | training state MSE {loss:.8f}"
            )

    return model, mean, std


def reconstruct_state(model, mean, std, state):
    normalized = (state - mean) / std
    z = model.encode(normalized.reshape(1, -1))
    decoded = model.decode(z)[0]
    reconstructed = decoded * std + mean
    return z[0], reconstructed


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sha256_array(x):
    return hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("EXPERIMENT 5: GENERALIZED + FUNCTIONAL STATE COMPRESSION")
    print("=" * 78)
    print(f"State dimension       : {STATE_DIM}")
    print(f"Training trajectories : {TRAIN_TRAJECTORIES}")
    print(f"Held-out trajectories : {TEST_TRAJECTORIES}")
    print(f"Latent sizes          : {LATENT_SIZES}")
    print()

    # ---------------------------------------------------------------
    # Generate independent trajectories.
    # ---------------------------------------------------------------
    print("[1] Generating independent trajectories...")

    all_trajectories = [
        generate_trajectory(SEED + i)
        for i in range(TRAIN_TRAJECTORIES + TEST_TRAJECTORIES)
    ]

    train_traj = all_trajectories[:TRAIN_TRAJECTORIES]
    test_traj = all_trajectories[TRAIN_TRAJECTORIES:]

    train_states = np.stack(
        [t["final_state"] for t in train_traj]
    )

    test_states = np.stack(
        [t["final_state"] for t in test_traj]
    )

    print(f"Training states shape : {train_states.shape}")
    print(f"Test states shape     : {test_states.shape}")
    print()

    # ---------------------------------------------------------------
    # Full-state behavioral baseline.
    # ---------------------------------------------------------------
    print("[2] Establishing full-state behavioral baseline...")

    full_losses = []
    full_predictions = []

    for t in test_traj:
        losses, predictions = future_behavior(
            t["final_state"], t["stream"]
        )
        full_losses.append(losses)
        full_predictions.append(predictions)

    print(
        f"Mean held-out future loss: "
        f"{np.mean([x.mean() for x in full_losses]):.8f}"
    )

    # ---------------------------------------------------------------
    # Train and test multiple latent sizes.
    # ---------------------------------------------------------------
    print()
    print("[3] Training shared state compressors...")

    results = []

    for latent_dim in LATENT_SIZES:
        print()
        print("-" * 78)
        print(f"LATENT DIMENSION = {latent_dim}")
        print("-" * 78)

        local_rng = np.random.default_rng(SEED + latent_dim)

        model, mean, std = train_autoencoder(
            train_states,
            latent_dim,
            local_rng,
        )

        state_errors = []
        future_losses = []
        agreements = []
        latent_bytes_total = 0

        for idx, t in enumerate(test_traj):
            z, reconstructed = reconstruct_state(
                model,
                mean,
                std,
                t["final_state"],
            )

            state_errors.append(
                rmse(reconstructed, t["final_state"])
            )

            latent_bytes_total += z.nbytes

            learned_losses, learned_predictions = future_behavior(
                reconstructed,
                t["stream"],
            )

            future_losses.append(learned_losses.mean())

            agreements.append(
                np.mean(
                    learned_predictions
                    == full_predictions[idx]
                )
            )

        shared_decoder_bytes = model.parameter_bytes() + mean.nbytes + std.nbytes

        # The model is shared across all test instances.
        per_instance_bytes = latent_dim * 8  # NumPy float64 latent.
        amortized_package = shared_decoder_bytes / TEST_TRAJECTORIES + per_instance_bytes
        one_off_package = shared_decoder_bytes + per_instance_bytes

        result = {
            "latent_dim": latent_dim,
            "state_rmse": float(np.mean(state_errors)),
            "future_loss": float(np.mean(future_losses)),
            "prediction_agreement": float(np.mean(agreements)),
            "per_instance_latent_bytes": int(per_instance_bytes),
            "shared_decoder_bytes": int(shared_decoder_bytes),
            "amortized_bytes_per_instance": float(amortized_package),
            "one_off_package_bytes": int(one_off_package),
        }

        results.append(result)

        print(f"Test state RMSE           : {result['state_rmse']:.8f}")
        print(f"Test future loss           : {result['future_loss']:.8f}")
        print(
            f"Prediction agreement      : "
            f"{result['prediction_agreement'] * 100:.2f}%"
        )
        print(
            f"Per-instance latent bytes : "
            f"{result['per_instance_latent_bytes']}"
        )
        print(
            f"Shared decoder bytes      : "
            f"{result['shared_decoder_bytes']}"
        )
        print(
            f"Amortized bytes/instance  : "
            f"{result['amortized_bytes_per_instance']:.1f}"
        )

    # ---------------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        f"{'Latent':>8} | "
        f"{'RMSE':>12} | "
        f"{'Future Loss':>12} | "
        f"{'Behavior':>10} | "
        f"{'Latent B':>10} | "
        f"{'Shared B':>12}"
    )
    print("-" * 78)

    for r in results:
        print(
            f"{r['latent_dim']:>8} | "
            f"{r['state_rmse']:>12.6f} | "
            f"{r['future_loss']:>12.6f} | "
            f"{r['prediction_agreement']*100:>9.2f}% | "
            f"{r['per_instance_latent_bytes']:>10} | "
            f"{r['shared_decoder_bytes']:>12}"
        )

    print()
    print("Full final-state size:")
    print(f"    {STATE_DIM * 8} bytes per instance (float64)")

    print()
    print("=" * 78)
    print("WHAT THIS TEST ACTUALLY ESTABLISHES")
    print("=" * 78)
    print(
        "A result is interesting if a compressor trained on OTHER trajectories "
        "can encode an unseen trajectory into a tiny latent while preserving "
        "future behavior."
    )
    print()
    print(
        "State RMSE can be non-zero. The stronger test is whether the "
        "reconstructed state continues to behave like the original state "
        "when exposed to future inputs it did not see during compression."
    )
    print()
    print(
        "The shared decoder is reported separately from per-instance storage "
        "so we do not hide the cost of the reconstruction machinery."
    )
    print()
    print("Held-out test-state SHA-256:")
    for i, t in enumerate(test_traj):
        print(f"    test {i:02d}: {sha256_array(t['final_state'])}")


if __name__ == "__main__":
    main()
