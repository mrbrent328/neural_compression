#!/usr/bin/env python3
"""
Experiment 2: Learned Latent "Rotation" Test

Question:
    Does a trained decoder depend on the absolute coordinates of its latent
    representation, or can it reconstruct the same data when the latent
    representation is viewed through a different orthogonal basis?

Pipeline:
    data -> encoder -> Z -> transformation(Z) -> frozen decoder -> reconstruction

Transformations tested:
    - original latent
    - small random orthogonal rotations
    - 15, 30, 45, 90 degree rotations in a random latent plane
    - random axis permutation
    - sign flips

This is deliberately a small synthetic experiment. It is NOT a compression
benchmark. It tests representation invariance.

Dependencies:
    numpy
"""

import hashlib
import math
import numpy as np


SEED = 42
LATENT_DIM = 8
INPUT_DIM = 16
HIDDEN_DIM = 32
SAMPLES = 4096
EPOCHS = 1200
LR = 0.01


def make_data():
    """Structured data with several correlated dimensions."""
    rng = np.random.default_rng(SEED)

    t = rng.uniform(-math.pi, math.pi, SAMPLES)
    x = np.stack(
        [
            np.sin(t),
            np.cos(t),
            np.sin(2 * t),
            np.cos(2 * t),
            np.sin(3 * t),
            np.cos(3 * t),
            t / math.pi,
            (t / math.pi) ** 2,
            np.sin(t) * np.cos(2 * t),
            np.cos(t) * np.sin(2 * t),
            np.sin(4 * t),
            np.cos(4 * t),
            np.sin(t + 0.3),
            np.cos(t - 0.2),
            np.sin(2 * t + 0.5),
            np.cos(2 * t - 0.5),
        ],
        axis=1,
    )

    # Keep the exact test set separate.
    split = int(SAMPLES * 0.8)
    return x[:split], x[split:]


class Autoencoder:
    """
    Tiny hand-rolled autoencoder.

    Encoder: 16 -> 32 -> 8
    Decoder: 8 -> 32 -> 16

    We use tanh hidden layers to keep the experiment dependency-free.
    """

    def __init__(self, rng):
        self.W1 = rng.normal(0, 0.15, (INPUT_DIM, HIDDEN_DIM))
        self.b1 = np.zeros(HIDDEN_DIM)

        self.W2 = rng.normal(0, 0.15, (HIDDEN_DIM, LATENT_DIM))
        self.b2 = np.zeros(LATENT_DIM)

        self.W3 = rng.normal(0, 0.15, (LATENT_DIM, HIDDEN_DIM))
        self.b3 = np.zeros(HIDDEN_DIM)

        self.W4 = rng.normal(0, 0.15, (HIDDEN_DIM, INPUT_DIM))
        self.b4 = np.zeros(INPUT_DIM)

    def encode(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        return z

    def decode(self, z):
        h = np.tanh(z @ self.W3 + self.b3)
        return h @ self.W4 + self.b4

    def forward(self, x):
        h1 = np.tanh(x @ self.W1 + self.b1)
        z = h1 @ self.W2 + self.b2
        h2 = np.tanh(z @ self.W3 + self.b3)
        y = h2 @ self.W4 + self.b4
        return h1, z, h2, y


def train(model, x_train):
    n = len(x_train)

    for epoch in range(EPOCHS):
        h1, z, h2, y = model.forward(x_train)
        dy = (2.0 / n) * (y - x_train)

        # Decoder gradients.
        dW4 = h2.T @ dy
        db4 = dy.sum(axis=0)

        dh2 = dy @ model.W4.T
        dh2_pre = dh2 * (1.0 - h2 * h2)

        dW3 = z.T @ dh2_pre
        db3 = dh2_pre.sum(axis=0)

        dz = dh2_pre @ model.W3.T

        # Encoder gradients.
        dW2 = h1.T @ dz
        db2 = dz.sum(axis=0)

        dh1 = dz @ model.W2.T
        dh1_pre = dh1 * (1.0 - h1 * h1)

        dW1 = x_train.T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)

        model.W4 -= LR * dW4
        model.b4 -= LR * db4
        model.W3 -= LR * dW3
        model.b3 -= LR * db3
        model.W2 -= LR * dW2
        model.b2 -= LR * db2
        model.W1 -= LR * dW1
        model.b1 -= LR * db1

        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            loss = np.mean((y - x_train) ** 2)
            print(f"epoch {epoch:4d} | training MSE {loss:.8f}")


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def make_plane_rotation(dim, i, j, degrees):
    """Rotate two latent axes while leaving all other axes untouched."""
    r = np.eye(dim)
    theta = math.radians(degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    r[i, i] = c
    r[j, j] = c
    r[i, j] = -s
    r[j, i] = s
    return r


def random_orthogonal(dim, rng):
    """Generate a true orthogonal matrix R where R.T @ R = I."""
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)

    # Normalize signs so the result is deterministic for a given RNG.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q @ np.diag(signs)
    return q


def sha256_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    rng = np.random.default_rng(SEED)

    x_train, x_test = make_data()

    print("=" * 68)
    print("EXPERIMENT 2: LEARNED LATENT 'ROTATION' TEST")
    print("=" * 68)
    print(f"Input dimension : {INPUT_DIM}")
    print(f"Latent dimension: {LATENT_DIM}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples    : {len(x_test)}")
    print()

    model = Autoencoder(rng)

    print("[1] Training autoencoder...")
    train(model, x_train)

    # Freeze the network conceptually by doing no further updates.
    z = model.encode(x_test)
    baseline = model.decode(z)
    baseline_error = mse(x_test, baseline)

    print()
    print("[2] Frozen-decoder baseline")
    print(f"Baseline MSE: {baseline_error:.8f}")

    print()
    print("[3] Testing alternate views of the same latent state")
    print("-" * 68)

    results = []

    def test(name, z_variant):
        y = model.decode(z_variant)
        error = mse(x_test, y)
        ratio = error / max(baseline_error, 1e-15)
        results.append((name, error, ratio))
        print(f"{name:<38} MSE={error:.8f}  ({ratio:10.2f}x baseline)")

    test("Original Z", z)

    for degrees in [1, 5, 15, 30, 45, 90, 180]:
        R = make_plane_rotation(LATENT_DIM, 0, 1, degrees)
        test(f"Plane rotation {degrees:>3} degrees", z @ R)

    R = random_orthogonal(LATENT_DIM, rng)
    test("Random orthogonal rotation", z @ R)

    permutation = rng.permutation(LATENT_DIM)
    test("Random axis permutation", z[:, permutation])

    signs = rng.choice([-1.0, 1.0], size=LATENT_DIM)
    test("Random axis sign flips", z * signs)

    # A deliberately destructive control.
    shuffled = z.copy()
    rng.shuffle(shuffled, axis=0)
    test("Shuffled samples (destructive control)", shuffled)

    print()
    print("[4] Representation identity checks")
    print("-" * 68)

    R = random_orthogonal(LATENT_DIM, rng)
    orthogonality_error = np.max(np.abs(R.T @ R - np.eye(LATENT_DIM)))

    print(f"Original Z SHA-256          : {sha256_array(z)}")
    print(f"Rotated Z SHA-256           : {sha256_array(z @ R)}")
    print(f"Orthogonality max error     : {orthogonality_error:.3e}")
    print(f"Original == rotated bytes?  : {np.array_equal(z, z @ R)}")

    print()
    print("=" * 68)
    print("INTERPRETATION")
    print("=" * 68)
    print(
        "If rotations produce nearly the same reconstruction error as the "
        "original latent, the decoder is substantially invariant to that "
        "change of representation."
    )
    print(
        "If even small rotations cause a large error increase, the decoder "
        "depends strongly on the learned latent coordinate system."
    )
    print(
        "The shuffled-sample control should fail. It checks that we are not "
        "mistaking arbitrary latent perturbation for useful invariance."
    )


if __name__ == "__main__":
    main()
