#!/usr/bin/env python3
import hashlib
import numpy as np

rng = np.random.default_rng(42)

data = np.array(
    ([0, 1, 2, 3, 4, 5, 4, 3, 2, 1] * 100)
    + ([10, 20, 30, 40, 30, 20] * 100),
    dtype=np.uint8,
)

block_size = 10
X = data.reshape(-1, block_size).astype(np.float64)

# Reversible "rotation".
A = rng.normal(size=(block_size, block_size))
Q, _ = np.linalg.qr(A)
rotated = X @ Q

# View the representation from the original angle again.
reconstructed = rotated @ Q.T
max_error = np.max(np.abs(X - reconstructed))

# A second representation: rotate, then flip several axes.
axis_flip = np.ones(block_size)
axis_flip[[1, 4, 7]] = -1
rotated_alt = rotated * axis_flip

# Decode the second representation.
alt_reconstructed = (rotated_alt * axis_flip) @ Q.T
alt_error = np.max(np.abs(X - alt_reconstructed))

print("=" * 60)
print("REPRESENTATION ROTATION BASELINE")
print("=" * 60)
print(f"Original bytes:              {data.nbytes}")
print(f"Representation bytes:       {rotated.nbytes}")
print(f"Alternative representation: {rotated_alt.nbytes}")
print()
print("Original SHA-256:")
print(hashlib.sha256(data.tobytes()).hexdigest())
print()
print(f"Rotation max error:          {max_error:.3e}")
print(f"Alternate max error:         {alt_error:.3e}")

if max_error < 1e-10 and alt_error < 1e-10:
    print()
    print("PASS: both representations reconstruct the same state.")
else:
    print()
    print("FAIL: reconstruction changed the state.")
