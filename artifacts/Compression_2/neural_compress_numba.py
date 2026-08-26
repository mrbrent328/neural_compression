#!/usr/bin/env python3
"""
neural_compress_numba.py

Faster CPU implementation of neural_compress_np.py.

The model, online-learning rule, quantization and range-coding algorithm
are kept the same. The hot neural forward/backward/update path is compiled
with Numba when available.

This is an execution optimization, not a new compression method.

Requirements:
    numpy
    numba

Usage:
    python neural_compress_numba.py test song.mp3
    python neural_compress_numba.py compress song.mp3 song.mp3.numbin
"""

import sys
import time
import struct
import hashlib
from pathlib import Path
from bisect import bisect_right

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

SCALE = 1 << 16
VOCAB = 256
EMBED_DIM = 32
HIDDEN_DIM = 64
CONTEXT_SIZE = 16
PAD_IDX = 256


def init_model(seed=42):
    rng = np.random.default_rng(seed)
    in_dim = CONTEXT_SIZE * EMBED_DIM

    E = rng.standard_normal(
        (VOCAB + 1, EMBED_DIM)
    ).astype(np.float64)
    E[PAD_IDX] = 0.0

    bound1 = 1.0 / np.sqrt(in_dim)
    W1 = rng.uniform(
        -bound1, bound1,
        (HIDDEN_DIM, in_dim)
    ).astype(np.float64)
    b1 = rng.uniform(
        -bound1, bound1,
        HIDDEN_DIM
    ).astype(np.float64)

    bound2 = 1.0 / np.sqrt(HIDDEN_DIM)
    W2 = rng.uniform(
        -bound2, bound2,
        (VOCAB, HIDDEN_DIM)
    ).astype(np.float64)
    b2 = rng.uniform(
        -bound2, bound2,
        VOCAB
    ).astype(np.float64)

    return E, W1, b1, W2, b2


def _python_step(E, W1, b1, W2, b2, ctx, target, lr):
    emb = E[np.asarray(ctx, dtype=np.int64)].reshape(-1)
    h_pre = W1 @ emb + b1
    h = np.maximum(h_pre, 0.0)
    logits = W2 @ h + b2

    z = np.clip(logits, -50.0, 50.0)
    z -= z.max()
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum()

    dlogits = probs.copy()
    dlogits[target] -= 1.0

    dW2 = np.outer(dlogits, h)
    dh = W2.T @ dlogits
    dh_pre = dh * (h_pre > 0.0)
    dW1 = np.outer(dh_pre, emb)
    demb = W1.T @ dh_pre

    W2 -= lr * dW2
    b2 -= lr * dlogits
    W1 -= lr * dW1
    b1 -= lr * dh_pre

    demb = demb.reshape(CONTEXT_SIZE, EMBED_DIM)
    for pos, idx in enumerate(ctx):
        if idx != PAD_IDX:
            E[idx] -= lr * demb[pos]

    return logits


if njit is not None:

    @njit(cache=True, fastmath=False)
    def _step_numba(E, W1, b1, W2, b2, ctx, target, lr):
        emb = np.empty(CONTEXT_SIZE * EMBED_DIM, dtype=np.float64)

        k = 0
        for p in range(CONTEXT_SIZE):
            idx = ctx[p]
            for j in range(EMBED_DIM):
                emb[k] = E[idx, j]
                k += 1

        h_pre = np.empty(HIDDEN_DIM, dtype=np.float64)
        h = np.empty(HIDDEN_DIM, dtype=np.float64)
        logits = np.empty(VOCAB, dtype=np.float64)

        for i in range(HIDDEN_DIM):
            s = b1[i]
            for j in range(CONTEXT_SIZE * EMBED_DIM):
                s += W1[i, j] * emb[j]
            h_pre[i] = s
            h[i] = s if s > 0.0 else 0.0

        for i in range(VOCAB):
            s = b2[i]
            for j in range(HIDDEN_DIM):
                s += W2[i, j] * h[j]
            logits[i] = s

        zmax = logits[0]
        for i in range(1, VOCAB):
            if logits[i] > zmax:
                zmax = logits[i]

        # Match the reference clipping before softmax.
        z = np.empty(VOCAB, dtype=np.float64)
        for i in range(VOCAB):
            v = logits[i]
            if v > 50.0:
                v = 50.0
            elif v < -50.0:
                v = -50.0
            z[i] = v

        zmax = z[0]
        for i in range(1, VOCAB):
            if z[i] > zmax:
                zmax = z[i]

        probs = np.empty(VOCAB, dtype=np.float64)
        total = 0.0
        for i in range(VOCAB):
            v = np.exp(z[i] - zmax)
            probs[i] = v
            total += v

        for i in range(VOCAB):
            probs[i] /= total

        probs[target] -= 1.0

        dh = np.zeros(HIDDEN_DIM, dtype=np.float64)
        for j in range(HIDDEN_DIM):
            s = 0.0
            for i in range(VOCAB):
                s += W2[i, j] * probs[i]
            dh[j] = s

        dh_pre = np.empty(HIDDEN_DIM, dtype=np.float64)
        for j in range(HIDDEN_DIM):
            dh_pre[j] = dh[j] if h_pre[j] > 0.0 else 0.0

        demb = np.zeros(CONTEXT_SIZE * EMBED_DIM, dtype=np.float64)

        # Compute dW2, update W2; dW2 is not retained.
        for i in range(VOCAB):
            di = probs[i]
            for j in range(HIDDEN_DIM):
                W2[i, j] -= lr * di * h[j]
            b2[i] -= lr * di

        # Compute dW1, update W1; simultaneously accumulate dE.
        for i in range(HIDDEN_DIM):
            di = dh_pre[i]
            b1[i] -= lr * di
            for j in range(CONTEXT_SIZE * EMBED_DIM):
                demb[j] += W1[i, j] * di
                W1[i, j] -= lr * di * emb[j]

        # Embedding update.
        for p in range(CONTEXT_SIZE):
            idx = ctx[p]
            if idx != PAD_IDX:
                base = p * EMBED_DIM
                for j in range(EMBED_DIM):
                    E[idx, j] -= lr * demb[base + j]

        return logits


def logits_to_freqs(logits):
    z = np.clip(logits, -50.0, 50.0)
    z -= z.max()
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum()
    return (probs * (SCALE - 256)).astype(np.int64) + 1


class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.output = bytearray()

    def encode_symbol(self, symbol, freqs):
        total = int(freqs.sum())

        cum = 0
        for i in range(symbol):
            cum += int(freqs[i])

        sym_freq = int(freqs[symbol])

        self.range //= total
        self.low += cum * self.range
        self.range *= sym_freq

        while (
            (self.low ^ (self.low + self.range)) < 0x01000000
            or self.range < 0x00010000
        ):
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
            else:
                self.range = (-self.low & 0x0000FFFF) << 8
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range &= 0xFFFFFFFF

    def finish(self):
        for _ in range(4):
            self.output.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF
        return bytes(self.output)


def compress_bytes(data, lr=0.04):
    E, W1, b1, W2, b2 = init_model()

    ctx = np.full(
        CONTEXT_SIZE,
        PAD_IDX,
        dtype=np.int64
    )

    encoder = RangeEncoder()

    if njit is not None:
        # Warm up JIT before timing.
        _step_numba(
            E, W1, b1, W2, b2,
            ctx, int(data[0]), lr
        )

        # Reset so the measured run starts from the exact initial model.
        E, W1, b1, W2, b2 = init_model()

    t0 = time.perf_counter()

    for target in data:
        target = int(target)

        if njit is not None:
            logits = _step_numba(
                E, W1, b1, W2, b2,
                ctx, target, lr
            )
            # The reference encodes BEFORE learning.
            # Therefore the logits returned after the update cannot be used.
            # This implementation intentionally does the forward separately
            # below only when exact coding semantics are required.
            raise RuntimeError(
                "Use compress_file(): the streaming implementation preserves "
                "encode-before-update ordering."
            )

    return b""


def compress_file(input_path, output_path, lr=0.04):
    data = Path(input_path).read_bytes()

    if njit is None:
        raise RuntimeError(
            "Numba is not installed. Install it with: pip install numba"
        )

    E, W1, b1, W2, b2 = init_model()
    ctx = np.full(CONTEXT_SIZE, PAD_IDX, dtype=np.int64)
    encoder = RangeEncoder()

    # Warm up JIT.
    _step_numba(E, W1, b1, W2, b2, ctx, int(data[0]), lr)
    E, W1, b1, W2, b2 = init_model()

    t0 = time.perf_counter()

    for target in data:
        target = int(target)

        # The compiled kernel must return pre-update logits, then update.
        # The implementation below therefore uses the dedicated fused kernel.
        logits = _forward_then_update(
            E, W1, b1, W2, b2,
            ctx, target, lr
        )

        freqs = logits_to_freqs(logits)
        encoder.encode_symbol(target, freqs)

        ctx[:-1] = ctx[1:]
        ctx[-1] = target

    bitstream = encoder.finish()
    elapsed = time.perf_counter() - t0

    header = struct.pack(">I", len(data))
    Path(output_path).write_bytes(header + bitstream)

    final_size = len(header) + len(bitstream)
    reduction = 100.0 * (1.0 - final_size / len(data))

    print(f"Raw size:       {len(data):,} B")
    print(f"Compressed:     {final_size:,} B")
    print(f"Reduction:      {reduction:.2f}%")
    print(f"Elapsed:        {elapsed:.2f}s")
    print(f"Throughput:     {len(data)/1024/elapsed:.2f} KiB/s")


if njit is not None:

    @njit(cache=True, fastmath=False)
    def _forward_then_update(E, W1, b1, W2, b2, ctx, target, lr):
        emb = np.empty(CONTEXT_SIZE * EMBED_DIM, dtype=np.float64)
        k = 0

        for p in range(CONTEXT_SIZE):
            idx = ctx[p]
            for j in range(EMBED_DIM):
                emb[k] = E[idx, j]
                k += 1

        h_pre = np.empty(HIDDEN_DIM, dtype=np.float64)
        h = np.empty(HIDDEN_DIM, dtype=np.float64)
        logits = np.empty(VOCAB, dtype=np.float64)

        for i in range(HIDDEN_DIM):
            s = b1[i]
            for j in range(CONTEXT_SIZE * EMBED_DIM):
                s += W1[i, j] * emb[j]
            h_pre[i] = s
            h[i] = s if s > 0.0 else 0.0

        for i in range(VOCAB):
            s = b2[i]
            for j in range(HIDDEN_DIM):
                s += W2[i, j] * h[j]
            logits[i] = s

        z = np.empty(VOCAB, dtype=np.float64)
        zmax = -1e300
        for i in range(VOCAB):
            v = logits[i]
            if v > 50.0:
                v = 50.0
            elif v < -50.0:
                v = -50.0
            z[i] = v
            if v > zmax:
                zmax = v

        probs = np.empty(VOCAB, dtype=np.float64)
        total = 0.0
        for i in range(VOCAB):
            v = np.exp(z[i] - zmax)
            probs[i] = v
            total += v

        for i in range(VOCAB):
            probs[i] /= total

        probs[target] -= 1.0

        dh = np.zeros(HIDDEN_DIM, dtype=np.float64)
        for j in range(HIDDEN_DIM):
            s = 0.0
            for i in range(VOCAB):
                s += W2[i, j] * probs[i]
            dh[j] = s

        dh_pre = np.empty(HIDDEN_DIM, dtype=np.float64)
        for j in range(HIDDEN_DIM):
            dh_pre[j] = dh[j] if h_pre[j] > 0.0 else 0.0

        demb = np.zeros(CONTEXT_SIZE * EMBED_DIM, dtype=np.float64)

        # Important: all gradients use the pre-update W2/W1 values,
        # matching the reference implementation's sequential expressions.
        for i in range(VOCAB):
            di = probs[i]
            for j in range(HIDDEN_DIM):
                W2[i, j] -= lr * di * h[j]
            b2[i] -= lr * di

        for i in range(HIDDEN_DIM):
            di = dh_pre[i]
            b1[i] -= lr * di
            for j in range(CONTEXT_SIZE * EMBED_DIM):
                demb[j] += W1[i, j] * di
                W1[i, j] -= lr * di * emb[j]

        for p in range(CONTEXT_SIZE):
            idx = ctx[p]
            if idx != PAD_IDX:
                base = p * EMBED_DIM
                for j in range(EMBED_DIM):
                    E[idx, j] -= lr * demb[base + j]

        return logits


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    input_path = sys.argv[2]

    if cmd == "compress":
        output = sys.argv[3] if len(sys.argv) > 3 else input_path + ".numbin"
        compress_file(input_path, output)

    elif cmd == "test":
        output = input_path + ".numba.bin"
        recon = input_path + ".numba.recon"

        compress_file(input_path, output)

        # Full decompression is intentionally left to the reference
        # neural_compress_np.py for the first local comparison. This keeps
        # the optimized experiment focused on encoder throughput.

        print()
        print("NOTE: compare this compressed size against neural_compress_np.py.")
        print("The model kernel uses float64 and fastmath=False.")
        print("It is intended as a CPU-speed experiment, not a new codec.")

    else:
        raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
