#!/usr/bin/env python3
"""
Experiment 6: Process -> Compact State -> Future Behavior

This returns to the mechanism of the user's actual neural compressor.

Core idea:
    fixed initial model + observed prefix
                  |
                  v
          evolving model state
                  |
                  v
             compact Z
                  |
                  v
        reconstructed model state
                  |
                  v
          unseen future input
                  |
                  v
             predictions

Unlike Experiment 5:
- EVERY trajectory starts from the SAME initial model.
- We do NOT ask a generic autoencoder to represent arbitrary random models.
- We test whether the information needed for future behavior can be
  represented compactly after processing a prefix.

We also include two important baselines:
1. Full state.
2. "Predictive state" trained directly against future behavior.

The main question is functional:
    Can Z preserve what the state DOES, rather than every number in the state?

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

PREFIX = 700
FUTURE = 500

TRAIN_STREAMS = 60
TEST_STREAMS = 15

LATENT_SIZES = [2, 4, 8, 16]

MODEL_LR = 0.025
ENCODER_HIDDEN = 64
TRAIN_STEPS = 1800
PREDICTIVE_SAMPLES = 64


def make_stream(seed, length):
    """
    Different data streams, but ALL are processed by the same initial model.
    """
    rng = np.random.default_rng(seed)

    x = np.zeros(length, dtype=np.int64)
    x[:CONTEXT] = rng.integers(0, VOCAB, size=CONTEXT)

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
    """
    Same online predictor used in the earlier experiments.
    """

    def __init__(self, rng):
        self.E = rng.normal(0, 0.20, (VOCAB, EMBED))
        self.W = rng.normal(0, 0.15, (EMBED * CONTEXT, HIDDEN))
        self.b = np.zeros(HIDDEN)
        self.V = rng.normal(0, 0.15, (HIDDEN, VOCAB))
        self.c = np.zeros(VOCAB)

    def copy(self):
        out = object.__new__(OnlinePredictor)
        out.E = self.E.copy()
        out.W = self.W.copy()
        out.b = self.b.copy()
        out.V = self.V.copy()
        out.c = self.c.copy()
        return out

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

    def predict(self, ctx):
        emb = self.E[ctx].reshape(-1)
        h = np.tanh(emb @ self.W + self.b)
        return h @ self.V + self.c

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


def make_initial_model():
    """
    One fixed initial model. Every stream begins here.
    """
    return OnlinePredictor(np.random.default_rng(SEED))


def process_prefix(initial_model, stream):
    model = initial_model.copy()
    states = []

    for i in range(CONTEXT, PREFIX):
        ctx = stream[i-CONTEXT:i]
        model.step(ctx, int(stream[i]))
        states.append(model.state_vector())

    return model.state_vector(), np.array(states)


def evaluate_future(state, stream):
    model = make_initial_model()
    model.load_state_vector(state)

    losses = []
    predictions = []

    for i in range(PREFIX, PREFIX + FUTURE):
        ctx = stream[i-CONTEXT:i]
        target = int(stream[i])

        loss, logits = model.step(ctx, target)

        losses.append(loss)
        predictions.append(int(np.argmax(logits)))

    return np.array(losses), np.array(predictions)


class PredictiveStateEncoder:
    """
    Encoder/decoder shared across streams.

    It learns a compact representation of the PREFIX's resulting state.

    Training objective:
        reconstruct the final state

    But evaluation is functional:
        reconstructed state -> unseen future -> behavior
    """

    def __init__(self, latent_dim, rng):
        self.W1 = rng.normal(0, 0.01, (STATE_DIM, ENCODER_HIDDEN))
        self.b1 = np.zeros(ENCODER_HIDDEN)

        self.W2 = rng.normal(
            0, 0.01, (ENCODER_HIDDEN, latent_dim)
        )
        self.b2 = np.zeros(latent_dim)

        self.W3 = rng.normal(
            0, 0.01, (latent_dim, ENCODER_HIDDEN)
        )
        self.b3 = np.zeros(ENCODER_HIDDEN)

        self.W4 = rng.normal(
            0, 0.01, (ENCODER_HIDDEN, STATE_DIM)
        )
        self.b4 = np.zeros(STATE_DIM)

    def encode(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decode(self, z):
        h = np.tanh(z @ self.W3 + self.b3)
        return h @ self.W4 + self.b4

    def parameter_bytes(self):
        return sum(
            x.nbytes
            for x in [
                self.W1, self.b1,
                self.W2, self.b2,
                self.W3, self.b3,
                self.W4, self.b4,
            ]
        )


def train_encoder(states, latent_dim, rng):
    """
    Train on many PREFIX->STATE examples, all generated from the SAME
    initial model.

    Global normalization is shared reconstruction machinery.
    """
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-6
    x = (states - mean) / std

    model = PredictiveStateEncoder(latent_dim, rng)
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

        model.W4 -= 0.01 * dW4
        model.b4 -= 0.01 * db4
        model.W3 -= 0.01 * dW3
        model.b3 -= 0.01 * db3
        model.W2 -= 0.01 * dW2
        model.b2 -= 0.01 * db2
        model.W1 -= 0.01 * dW1
        model.b1 -= 0.01 * db1

        if step % 450 == 0 or step == TRAIN_STEPS - 1:
            loss = float(np.mean((y - x) ** 2))
            print(
                f"    step {step:4d} | state reconstruction MSE {loss:.8f}"
            )

    return model, mean, std


def reconstruct(model, mean, std, state):
    x = (state - mean) / std
    z = model.encode(x.reshape(1, -1))[0]
    y = model.decode(z.reshape(1, -1))[0]
    return z, y * std + mean


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sha256_array(x):
    return hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("EXPERIMENT 6: PROCESS -> COMPACT STATE -> FUTURE BEHAVIOR")
    print("=" * 78)
    print(f"State dimension       : {STATE_DIM}")
    print(f"Training streams      : {TRAIN_STREAMS}")
    print(f"Held-out streams      : {TEST_STREAMS}")
    print(f"Latent sizes          : {LATENT_SIZES}")
    print(f"ONE fixed initial model used for ALL streams")
    print()

    # ---------------------------------------------------------------
    # Same initial model, different data streams.
    # ---------------------------------------------------------------
    initial_model = make_initial_model()

    print("[1] Processing streams from the SAME initial model...")

    train_examples = []
    test_examples = []

    for i in range(TRAIN_STREAMS + TEST_STREAMS):
        stream = make_stream(SEED + 1000 + i, PREFIX + FUTURE)
        final_state, trajectory = process_prefix(
            initial_model,
            stream,
        )

        example = {
            "stream": stream,
            "final_state": final_state,
            "trajectory": trajectory,
        }

        if i < TRAIN_STREAMS:
            train_examples.append(example)
        else:
            test_examples.append(example)

    train_states = np.stack(
        [x["final_state"] for x in train_examples]
    )

    print(f"Training state matrix : {train_states.shape}")
    print(f"Initial state hash     : {sha256_array(initial_model.state_vector())}")
    print()

    # ---------------------------------------------------------------
    # Full-state behavioral baseline.
    # ---------------------------------------------------------------
    print("[2] Full-state future behavior baseline...")

    full_losses = []
    full_predictions = []

    for x in test_examples:
        losses, predictions = evaluate_future(
            x["final_state"],
            x["stream"],
        )
        full_losses.append(losses)
        full_predictions.append(predictions)

    full_loss = float(np.mean([x.mean() for x in full_losses]))

    print(f"Full-state future loss: {full_loss:.8f}")
    print()

    # ---------------------------------------------------------------
    # Train shared compact representations.
    # ---------------------------------------------------------------
    print("[3] Training compact representations...")
    results = []

    for latent_dim in LATENT_SIZES:
        print()
        print("-" * 78)
        print(f"LATENT DIMENSION = {latent_dim}")
        print("-" * 78)

        local_rng = np.random.default_rng(
            SEED + 5000 + latent_dim
        )

        encoder, mean, std = train_encoder(
            train_states,
            latent_dim,
            local_rng,
        )

        state_errors = []
        future_losses = []
        behavior_agreements = []

        for idx, example in enumerate(test_examples):
            z, reconstructed_state = reconstruct(
                encoder,
                mean,
                std,
                example["final_state"],
            )

            state_errors.append(
                rmse(
                    reconstructed_state,
                    example["final_state"],
                )
            )

            losses, predictions = evaluate_future(
                reconstructed_state,
                example["stream"],
            )

            future_losses.append(losses.mean())

            behavior_agreements.append(
                np.mean(
                    predictions == full_predictions[idx]
                )
            )

        shared_bytes = (
            encoder.parameter_bytes()
            + mean.nbytes
            + std.nbytes
        )

        per_instance_bytes = latent_dim * 8

        amortized = (
            shared_bytes / TEST_STREAMS
            + per_instance_bytes
        )

        results.append(
            {
                "latent": latent_dim,
                "rmse": float(np.mean(state_errors)),
                "future_loss": float(np.mean(future_losses)),
                "agreement": float(np.mean(behavior_agreements)),
                "shared_bytes": int(shared_bytes),
                "instance_bytes": int(per_instance_bytes),
                "amortized_bytes": float(amortized),
            }
        )

        print(f"Held-out state RMSE       : {results[-1]['rmse']:.8f}")
        print(f"Held-out future loss      : {results[-1]['future_loss']:.8f}")
        print(
            f"Future prediction match  : "
            f"{results[-1]['agreement'] * 100:.2f}%"
        )
        print(f"Shared reconstruction bytes: {shared_bytes}")
        print(f"Per-instance latent bytes : {per_instance_bytes}")
        print(f"Amortized bytes/instance  : {amortized:.1f}")

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # IMPORTANT SECOND TEST:
    #
    # Instead of reconstructing weights, test whether the latent can
    # predict FUTURE OUTPUTS directly.
    #
    # This is the functional version of the hypothesis.
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("FUNCTIONAL TEST: LATENT -> FUTURE BEHAVIOR")
    print("=" * 78)
    print(
        "This section deliberately does NOT require reconstructing the "
        "928-value model state."
    )
    print(
        "The compact representation is judged by whether it preserves "
        "future predictions."
    )

    # A simple direct functional baseline:
    # store a compressed snapshot of the model's output logits over a set
    # of probe contexts. This is NOT the proposed solution; it establishes
    # how much behavioral information can be represented directly.
    probe_rng = np.random.default_rng(SEED + 9000)
    probe_contexts = probe_rng.integers(
        0,
        VOCAB,
        size=(PREDICTIVE_SAMPLES, CONTEXT),
    )

    # Gather behavior signatures for held-out states.
    behavior_signatures = []

    for example in test_examples:
        model = make_initial_model()
        model.load_state_vector(example["final_state"])

        logits = np.stack(
            [
                model.predict(ctx)
                for ctx in probe_contexts
            ]
        )

        behavior_signatures.append(logits)

    behavior_signatures = np.array(behavior_signatures)

    # PCA/SVD gives a non-neural, honest lower-dimensional behavioral
    # representation baseline.
    flat_behavior = behavior_signatures.reshape(
        TEST_STREAMS, -1
    )

    mean_behavior = flat_behavior.mean(axis=0)
    centered = flat_behavior - mean_behavior

    U, S, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    print()
    print("Behavior-signature dimensionality:")
    print(f"  Raw probe values: {flat_behavior.shape[1]}")
    print(
        "  This provides a useful control for the question: "
        "how low-dimensional is FUTURE BEHAVIOR itself?"
    )

    for k in [2, 4, 8, 16]:
        k = min(k, len(S))

        projected = centered @ Vt[:k].T
        reconstructed = projected @ Vt[:k] + mean_behavior

        behavior_rmse = rmse(
            flat_behavior,
            reconstructed,
        )

        explained = (
            np.sum(S[:k] ** 2)
            / max(np.sum(S ** 2), 1e-15)
        )

        print(
            f"  {k:2d} dimensions -> "
            f"behavior RMSE {behavior_rmse:.6f}, "
            f"variance retained {explained * 100:.2f}%"
        )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        f"{'Latent':>8} | "
        f"{'State RMSE':>12} | "
        f"{'Future Loss':>12} | "
        f"{'Behavior':>10} | "
        f"{'Instance B':>10} | "
        f"{'Shared B':>12}"
    )
    print("-" * 78)

    for r in results:
        print(
            f"{r['latent']:>8} | "
            f"{r['rmse']:>12.6f} | "
            f"{r['future_loss']:>12.6f} | "
            f"{r['agreement']*100:>9.2f}% | "
            f"{r['instance_bytes']:>10} | "
            f"{r['shared_bytes']:>12}"
        )

    print()
    print("Full final-state storage:")
    print(f"    {STATE_DIM * 8} bytes")

    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(
        "The critical comparison is not latent -> exact weights."
    )
    print(
        "It is: PREFIX -> tiny representation -> FUTURE BEHAVIOR."
    )
    print()
    print(
        "If a tiny latent preserves future behavior on held-out streams "
        "while using dramatically less per-instance information, that is "
        "evidence that the useful information about the processed prefix "
        "can live in a compact functional state rather than the full "
        "numerical model state."
    )
    print()
    print(
        "If it fails, compare it against the behavior-signature SVD. "
        "If future behavior itself is low-dimensional but our latent fails, "
        "the problem is our learned representation—not necessarily the "
        "underlying hypothesis."
    )


if __name__ == "__main__":
    main()
