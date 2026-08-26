#!/usr/bin/env python3
"""
config_benchmark.py - Evaluates Neural Compression Configurations

Compares:
1. IDLE / Untrained (lr = 0.0): Raw baseline with static random weights.
2. Default Configuration (lr = 0.04, ctx = 16, hidden = 64)
3. "Good" / Optimized Configuration (lr = 0.08, ctx = 32, hidden = 128)
"""

import sys
import os
import time
import struct
import numpy as np
from itertools import accumulate

os.environ["OMP_NUM_THREADS"] = "1"

SCALE = 1 << 16
VOCAB = 256
EMBED_DIM = 32
PAD_IDX = 256

class ConfigurableNeuralCompressor:
    def __init__(self, context_size=16, hidden_dim=64, seed=42):
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        in_dim = context_size * EMBED_DIM

        self.E = rng.standard_normal((VOCAB + 1, EMBED_DIM)).astype(np.float64)
        self.E[PAD_IDX] = 0.0
        bound1 = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.uniform(-bound1, bound1, (hidden_dim, in_dim)).astype(np.float64)
        self.b1 = rng.uniform(-bound1, bound1, hidden_dim).astype(np.float64)
        bound2 = 1.0 / np.sqrt(hidden_dim)
        self.W2 = rng.uniform(-bound2, bound2, (VOCAB, hidden_dim)).astype(np.float64)
        self.b2 = rng.uniform(-bound2, bound2, VOCAB).astype(np.float64)

    def forward(self, ctx_idx):
        emb = self.E[ctx_idx].reshape(-1)
        h_pre = self.W1 @ emb + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = self.W2 @ h + self.b2
        return logits, (ctx_idx, emb, h_pre, h)

    def backward_and_step(self, cache, target, lr):
        if lr <= 0.0:
            return  # Idle / Baseline: Skip updates completely

        ctx_idx, emb, h_pre, h = cache
        z = self.W2 @ h + self.b2
        z = z - z.max()
        exp_z = np.exp(z)
        probs = exp_z / exp_z.sum()

        dlogits = probs.copy()
        dlogits[target] -= 1.0

        dW2 = np.outer(dlogits, h)
        db2 = dlogits
        dh = self.W2.T @ dlogits
        dh_pre = dh * (h_pre > 0.0)

        dW1 = np.outer(dh_pre, emb)
        db1 = dh_pre
        demb = self.W1.T @ dh_pre

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        demb = demb.reshape(self.context_size, EMBED_DIM)
        for pos, idx in enumerate(ctx_idx):
            if idx != PAD_IDX:
                self.E[idx] -= lr * demb[pos]


def logits_to_freqs(logits, scale=SCALE):
    z = np.clip(logits, -50.0, 50.0)
    z = z - z.max()
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum()
    freqs = (probs * (scale - 256)).astype(np.int64) + 1
    return freqs.tolist()


def evaluate_config(data, name, lr, ctx_size, hidden_dim, chunk_size=32768):
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    model = ConfigurableNeuralCompressor(context_size=ctx_size, hidden_dim=hidden_dim)
    ctx = [PAD_IDX] * ctx_size

    total_in = 0
    total_bits = 0.0
    t0 = time.time()

    print(f"\n--- Running Configuration: {name} ---")
    print(f"Params: Learning Rate={lr} | Context={ctx_size} bytes | Hidden Dim={hidden_dim}")

    for idx, chunk in enumerate(chunks):
        chunk_bits = 0.0
        for b in chunk:
            logits, cache = model.forward(ctx)
            freqs = logits_to_freqs(logits)

            # Theoretical cross-entropy bitrate estimation in bits
            prob = freqs[b] / sum(freqs)
            chunk_bits += -np.log2(prob)

            model.backward_and_step(cache, b, lr)
            ctx = ctx[1:] + [b]

        total_in += len(chunk)
        total_bits += chunk_bits
        bpb = chunk_bits / len(chunk)
        print(f"  chunk {idx:2d} | {len(chunk):6d} B | {bpb:5.2f} bits/byte")

    elapsed = time.time() - t0
    final_bpb = total_bits / total_in
    print(f"RESULT [{name}]: {final_bpb:.2f} bits/byte | Throughput: {total_in / 1024 / elapsed:.2f} KB/s")
    return final_bpb, elapsed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python config_benchmark.py <test_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, "rb") as f:
        test_data = f.read()

    print(f"Benchmarking Configurations on: '{file_path}' ({len(test_data)} bytes)")

    # 1. Idle (Untrained / Learning Disabled)
    evaluate_config(test_data, "IDLE (Baseline - No Learning)", lr=0.0, ctx_size=16, hidden_dim=64)

    # 2. Default Configuration
    evaluate_config(test_data, "DEFAULT (lr=0.04, ctx=16, hidden=64)", lr=0.04, ctx_size=16, hidden_dim=64)

    # 3. Optimized Configuration
    evaluate_config(test_data, "OPTIMIZED (lr=0.08, ctx=32, hidden=128)", lr=0.08, ctx_size=32, hidden_dim=128)
