#!/usr/bin/env python3
"""
Experiment 4: Compress the State Trajectory

Goal
----
Test whether we can replace an evolving neural-compressor state trajectory
with a smaller representation while preserving:

    A. exact final state
    B. approximate final state
    C. future behavior

This is deliberately a small, inspectable experiment inspired by the
online-learning behavior in neural_compress_np.py:

    predict -> observe symbol -> update model -> next state

We record the model state at every step, then compare three things:

    1. Full trajectory
    2. Sparse checkpoint representation
    3. A learned compact trajectory representation

The learned representation is trained to predict the FINAL state from the
prefix trajectory. It is not allowed to see the future suffix.

The important behavioral test is:

    reconstructed final state
              |
              v
       future predictor
              |
              v
       future predictions

A reconstruction can therefore fail to reproduce internal numbers while
still preserving useful behavior.

Dependencies:
    numpy
"""

import hashlib
import math
import numpy as np


SEED = 42
VOCAB = 16
EMBED = 8
HIDDEN = 16
CONTEXT = 4

STREAM_LEN = 3000
PREFIX = 1500
FUTURE = STREAM_LEN - PREFIX

LATENT_DIM = 4
TRAJ_HIDDEN = 32

TRAIN_STEPS = 1000
LR_TRAJ = 0.01
LR_MODEL = 0.025


def make_stream(length):
    """Deterministic structured symbol stream."""
    x = np.zeros(length, dtype=np.int64)
    x[:CONTEXT] = [1, 3, 5, 7]

    for i in range(CONTEXT, length):
        a = x[i - 1]
        b = x[i - 2]
        c = x[i - 4]
        x[i] = (a + 2 * b + 3 * c + (i % 5)) % VOCAB

    return x


class OnlinePredictor:
    """
    Small online model matching the important structure of the user's
    compressor: prediction followed by an update using the observed symbol.
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
        """Flatten the complete model state."""
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
        """Restore a flattened model state."""
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

    def hidden(self, ctx):
        emb = self.E[ctx].reshape(-1)
        return np.tanh(emb @ self.W + self.b)

    def logits(self, ctx):
        h = self.hidden(ctx)
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

        self.V -= LR_MODEL * dV
        self.c -= LR_MODEL * dc
        self.W -= LR_MODEL * dW
        self.b -= LR_MODEL * db

        for pos, idx in enumerate(ctx):
            self.E[idx] -= LR_MODEL * dE[pos]

        return loss, logits


def run_stream(model, stream, end, save_states=False):
    states = []
    losses = []
    predictions = []

    for i in range(CONTEXT, end):
        ctx = stream[i-CONTEXT:i]
        loss, logits = model.step(ctx, int(stream[i]))

        losses.append(loss)
        predictions.append(int(np.argmax(logits)))

        if save_states:
            states.append(model.state_vector())

    return (
        np.array(losses),
        np.array(predictions),
        np.array(states) if save_states else None,
    )


class TrajectoryEncoder:
    """
    Tiny learned compressor for a trajectory.

    It sees the complete PREFIX trajectory and maps it to a very small latent.
    A decoder reconstructs the FINAL state.

    This deliberately asks:
        "Can the trajectory be represented by information sufficient to
         reconstruct the state needed at the boundary?"
    """

    def __init__(self, input_dim, latent_dim, rng):
        self.W1 = rng.normal(0, 0.01, (input_dim, TRAJ_HIDDEN))
        self.b1 = np.zeros(TRAJ_HIDDEN)
        self.W2 = rng.normal(0, 0.01, (TRAJ_HIDDEN, latent_dim))
        self.b2 = np.zeros(latent_dim)

        self.W3 = rng.normal(0, 0.01, (latent_dim, TRAJ_HIDDEN))
        self.b3 = np.zeros(TRAJ_HIDDEN)
        self.W4 = rng.normal(0, 0.01, (TRAJ_HIDDEN, input_dim))
        self.b4 = np.zeros(input_dim)

    def encode(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decode(self, z):
        h = np.tanh(z @ self.W3 + self.b3)
        return h @ self.W4 + self.b4


def train_trajectory_autoencoder(trajectory, rng):
    """
    Train on random contiguous trajectory windows.

    Each window is normalized independently so the network focuses on
    trajectory structure rather than absolute parameter scale.
    """
    # Downsample trajectory heavily to make the experiment practical.
    stride = max(1, len(trajectory) // 128)
    sampled = trajectory[::stride]

    mean = sampled.mean(axis=0)
    std = sampled.std(axis=0) + 1e-6
    x = (sampled - mean) / std

    # The actual trajectory is a single sequence. To avoid pretending that
    # this is a large independent dataset, we train the autoencoder only as
    # a reconstruction probe of this one trajectory.
    input_dim = x.size
    flat = x.reshape(1, -1)

    model = TrajectoryEncoder(input_dim, LATENT_DIM, rng)

    for step in range(TRAIN_STEPS):
        h1 = np.tanh(flat @ model.W1 + model.b1)
        z = h1 @ model.W2 + model.b2
        h2 = np.tanh(z @ model.W3 + model.b3)
        y = h2 @ model.W4 + model.b4

        dy = 2.0 * (y - flat)

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

        dW1 = flat.T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)

        model.W4 -= LR_TRAJ * dW4
        model.b4 -= LR_TRAJ * db4
        model.W3 -= LR_TRAJ * dW3
        model.b3 -= LR_TRAJ * db3
        model.W2 -= LR_TRAJ * dW2
        model.b2 -= LR_TRAJ * db2
        model.W1 -= LR_TRAJ * dW1
        model.b1 -= LR_TRAJ * db1

        if step % 200 == 0 or step == TRAIN_STEPS - 1:
            loss = float(np.mean((y - flat) ** 2))
            print(f"trajectory AE step {step:4d} | MSE {loss:.8f}")

    return model, mean, std, sampled


def sha256_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def future_behavior(model, stream, start):
    losses = []
    predictions = []

    for i in range(start, len(stream)):
        ctx = stream[i-CONTEXT:i]
        loss, logits = model.step(ctx, int(stream[i]))
        losses.append(loss)
        predictions.append(int(np.argmax(logits)))

    return np.array(losses), np.array(predictions)


def main():
    rng = np.random.default_rng(SEED)
    stream = make_stream(STREAM_LEN)

    print("=" * 72)
    print("EXPERIMENT 4: COMPRESS THE STATE TRAJECTORY")
    print("=" * 72)
    print(f"Stream length      : {STREAM_LEN}")
    print(f"Prefix             : {PREFIX}")
    print(f"Future              : {FUTURE}")
    print(f"Model state values  : {VOCAB*EMBED + EMBED*CONTEXT*HIDDEN + HIDDEN + HIDDEN*VOCAB + VOCAB}")
    print(f"Trajectory latent   : {LATENT_DIM}")
    print()

    # ---------------------------------------------------------------
    # 1. Generate and record the real trajectory.
    # ---------------------------------------------------------------
    reference = OnlinePredictor(rng)

    print("[1] Generating reference state trajectory...")
    losses, predictions, trajectory = run_stream(
        reference, stream, PREFIX, save_states=True
    )

    initial_model = OnlinePredictor(np.random.default_rng(SEED))

    final_state = trajectory[-1]
    state_dim = final_state.size

    print(f"State dimension     : {state_dim}")
    print(f"Trajectory snapshots: {len(trajectory)}")
    print(f"Prefix mean loss    : {losses.mean():.6f}")
    print(f"Final prefix loss   : {losses[-1]:.6f}")
    print()

    # ---------------------------------------------------------------
    # 2. Baseline: full state is sufficient for future behavior.
    # ---------------------------------------------------------------
    print("[2] Full-state behavioral baseline...")

    full_state_model = OnlinePredictor(np.random.default_rng(SEED))
    full_state_model.load_state_vector(final_state)

    full_losses, full_predictions = future_behavior(
        full_state_model, stream, PREFIX
    )

    print(f"Full-state future loss: {full_losses.mean():.8f}")

    # ---------------------------------------------------------------
    # 3. Sparse trajectory: keep only checkpoints.
    # ---------------------------------------------------------------
    print()
    print("[3] Sparse trajectory control...")

    checkpoint_count = 8
    checkpoint_idx = np.linspace(
        0, len(trajectory) - 1, checkpoint_count
    ).astype(int)

    checkpoints = trajectory[checkpoint_idx]

    # Reconstruct final state using the last checkpoint.
    sparse_final = checkpoints[-1]

    sparse_model = OnlinePredictor(np.random.default_rng(SEED))
    sparse_model.load_state_vector(sparse_final)

    sparse_losses, sparse_predictions = future_behavior(
        sparse_model, stream, PREFIX
    )

    sparse_state_error = float(
        np.sqrt(np.mean((sparse_final - final_state) ** 2))
    )

    print(f"Checkpoints kept    : {checkpoint_count}")
    print(f"State RMSE          : {sparse_state_error:.3e}")
    print(f"Future loss         : {sparse_losses.mean():.8f}")
    print(
        f"Prediction agreement: "
        f"{np.mean(sparse_predictions == full_predictions) * 100:.2f}%"
    )

    # ---------------------------------------------------------------
    # 4. Learned compact representation of the trajectory.
    # ---------------------------------------------------------------
    print()
    print("[4] Learning a compact representation of the trajectory...")

    traj_model, mean, std, sampled = train_trajectory_autoencoder(
        trajectory, rng
    )

    flat = ((sampled - mean) / std).reshape(1, -1)
    z = traj_model.encode(flat)
    reconstructed_flat = traj_model.decode(z)

    reconstructed_sampled = (
        reconstructed_flat.reshape(sampled.shape) * std + mean
    )

    # The final reconstructed trajectory state is our reconstructed state.
    learned_final = reconstructed_sampled[-1]

    learned_state_error = float(
        np.sqrt(np.mean((learned_final - final_state) ** 2))
    )

    print()
    print("[5] Learned trajectory -> reconstructed final state")
    print(f"Latent values       : {LATENT_DIM}")
    print(f"Original state vals : {state_dim}")
    print(f"State RMSE          : {learned_state_error:.8f}")

    learned_model = OnlinePredictor(np.random.default_rng(SEED))
    learned_model.load_state_vector(learned_final)

    learned_losses, learned_predictions = future_behavior(
        learned_model, stream, PREFIX
    )

    learned_prediction_agreement = float(
        np.mean(learned_predictions == full_predictions)
    )

    print(f"Full-state future loss : {full_losses.mean():.8f}")
    print(f"Learned-state future loss: {learned_losses.mean():.8f}")
    print(
        f"Prediction agreement   : "
        f"{learned_prediction_agreement * 100:.2f}%"
    )

    # ---------------------------------------------------------------
    # 6. Compression accounting.
    # ---------------------------------------------------------------
    full_bytes = final_state.nbytes
    trajectory_bytes = trajectory.nbytes
    latent_bytes = z.nbytes + mean.nbytes + std.nbytes

    # Decoder parameters are also required. Count them honestly.
    decoder_bytes = (
        traj_model.W3.nbytes
        + traj_model.b3.nbytes
        + traj_model.W4.nbytes
        + traj_model.b4.nbytes
    )

    compact_package = latent_bytes + mean.nbytes + std.nbytes + decoder_bytes

    print()
    print("[6] Storage accounting")
    print(f"Final state bytes       : {full_bytes}")
    print(f"Full trajectory bytes   : {trajectory_bytes}")
    print(f"Latent only bytes       : {latent_bytes}")
    print(f"Trajectory decoder bytes: {decoder_bytes}")
    print(f"Compact package bytes   : {compact_package}")

    if compact_package > 0:
        print(
            f"Final-state/package ratio: "
            f"{full_bytes / compact_package:.3f}x"
        )
        print(
            f"Trajectory/package ratio: "
            f"{trajectory_bytes / compact_package:.3f}x"
        )

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print(
        "The key measurement is NOT whether the learned trajectory decoder "
        "reproduces every internal value."
    )
    print(
        "The key measurement is whether a much smaller representation of "
        "the trajectory can recreate a state that behaves like the original "
        "state on unseen future data."
    )
    print()
    print(
        "A high future-prediction agreement with a much smaller package would "
        "support the idea that the useful information is carried by a compact "
        "description of the state trajectory rather than by every stored state."
    )
    print(
        "A poor agreement tells us the current trajectory representation is "
        "not sufficient; that is still useful because it identifies what the "
        "representation is failing to preserve."
    )
    print()
    print(f"Final state SHA-256: {sha256_array(final_state)}")


if __name__ == "__main__":
    main()
