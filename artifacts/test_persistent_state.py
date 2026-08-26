#!/usr/bin/env python3
"""
test_persistent_state.py - Persistent Neural State Compression Test

Evaluates whether retaining neural network model weights and context state
across fixed-size file chunks improves compression performance compared to
resetting the model on every chunk.
"""

import sys
import os
import time
import struct
import hashlib
from itertools import accumulate
from bisect import bisect_right

os.environ["OMP_NUM_THREADS"] = "1"  # Single-threaded BLAS for reproducibility
import numpy as np

SCALE = 1 << 16
VOCAB = 256
EMBED_DIM = 32
HIDDEN_DIM = 64
CONTEXT_SIZE = 16
PAD_IDX = 256


# =====================================================================
# Core Neural Engine & Entropy Coding (Unchanged from original implementation)
# =====================================================================

class NeuralCompressorNP:
    """Embedding(257,32) -> concat(512) -> Linear(512,64) -> ReLU -> Linear(64,256)"""

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        in_dim = CONTEXT_SIZE * EMBED_DIM
        self.E = rng.standard_normal((VOCAB + 1, EMBED_DIM)).astype(np.float64)
        self.E[PAD_IDX] = 0.0  # padding_idx rows stay zero
        bound1 = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.uniform(-bound1, bound1, (HIDDEN_DIM, in_dim)).astype(np.float64)
        self.b1 = rng.uniform(-bound1, bound1, HIDDEN_DIM).astype(np.float64)
        bound2 = 1.0 / np.sqrt(HIDDEN_DIM)
        self.W2 = rng.uniform(-bound2, bound2, (VOCAB, HIDDEN_DIM)).astype(np.float64)
        self.b2 = rng.uniform(-bound2, bound2, VOCAB).astype(np.float64)

    def forward(self, ctx_idx):
        emb = self.E[ctx_idx].reshape(-1)
        h_pre = self.W1 @ emb + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = self.W2 @ h + self.b2
        return logits, (ctx_idx, emb, h_pre, h)

    def backward_and_step(self, cache, target, lr):
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

        # SGD Update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        demb = demb.reshape(CONTEXT_SIZE, EMBED_DIM)
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


class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.output = bytearray()

    def encode_symbol(self, symbol, freqs, cum):
        total = cum[-1]
        cum_freq = cum[symbol]
        sym_freq = freqs[symbol]
        self.range //= total
        self.low += cum_freq * self.range
        self.range *= sym_freq
        while (self.low ^ (self.low + self.range)) < 0x01000000 or self.range < 0x00010000:
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
            else:
                self.range = (-self.low & 0x0000FFFF) << 8
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF

    def finish(self):
        for _ in range(4):
            self.output.append(self.low >> 24)
            self.low = (self.low << 8) & 0xFFFFFFFF
        return bytes(self.output)


class RangeDecoder:
    def __init__(self, stream):
        self.stream = stream
        self.ptr = 0
        self.low = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
            self.code = (self.code << 8) | b
            self.ptr += 1

    def decode_symbol(self, freqs, cum):
        total = cum[-1]
        count = (self.code - self.low) // (self.range // total)
        symbol = bisect_right(cum, count) - 1
        sym_freq = freqs[symbol]
        cum_freq = cum[symbol]
        self.range //= total
        self.low += cum_freq * self.range
        self.range *= sym_freq
        while (self.low ^ (self.low + self.range)) < 0x01000000 or self.range < 0x00010000:
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
                b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
                self.code = ((self.code << 8) | b) & 0xFFFFFFFF
                self.ptr += 1
            else:
                self.range = (-self.low & 0x0000FFFF) << 8
                self.low = (self.low << 8) & 0xFFFFFFFF
                b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
                self.code = ((self.code << 8) | b) & 0xFFFFFFFF
                self.ptr += 1
        return symbol


# =====================================================================
# Chunked Experiment Harness
# =====================================================================

def process_chunk(chunk_data, model, ctx, mode_encode=True, decoder_stream=None, lr=0.04):
    """Processes a single data chunk for either compression or decompression."""
    chunk_len = len(chunk_data) if mode_encode else None

    if mode_encode:
        encoder = RangeEncoder()
        t0 = time.time()
        for i in range(chunk_len):
            target_byte = chunk_data[i]
            logits, cache = model.forward(ctx)
            freqs = logits_to_freqs(logits)
            cum = list(accumulate(freqs, initial=0))
            encoder.encode_symbol(target_byte, freqs, cum)
            model.backward_and_step(cache, target_byte, lr)
            ctx = ctx[1:] + [target_byte]
        bitstream = encoder.finish()
        elapsed = time.time() - t0
        return bitstream, ctx, elapsed
    else:
        # Decompression mode
        decoder = decoder_stream
        decoded = bytearray()
        t0 = time.time()
        for _ in range(chunk_len):
            logits, cache = model.forward(ctx)
            freqs = logits_to_freqs(logits)
            cum = list(accumulate(freqs, initial=0))
            symbol = decoder.decode_symbol(freqs, cum)
            decoded.append(symbol)
            model.backward_and_step(cache, symbol, lr)
            ctx = ctx[1:] + [symbol]
        elapsed = time.time() - t0
        return bytes(decoded), ctx, elapsed


def run_experiment(input_path, chunk_size=65536, reset_interval=0, lr=0.04):
    """
    Runs chunked compression/decompression test.
    reset_interval = 1   => Experiment A (RESET every chunk)
    reset_interval = 0   => Experiment B (PERSISTENT model state)
    reset_interval = N   => Periodic resets every N chunks (Checkpointing test)
    """
    with open(input_path, "rb") as f:
        data = f.read()

    total_bytes = len(data)
    chunks = [data[i : i + chunk_size] for i in range(0, total_bytes, chunk_size)]

    mode_str = "RESET" if reset_interval == 1 else ("PERSISTENT" if reset_interval == 0 else f"RESET_EVERY_{reset_interval}")
    print(f"\n" + "=" * 70)
    print(f"RUNNING MODE: {mode_str} | File: '{input_path}' | Chunk Size: {chunk_size} B")
    print("=" * 70)

    # --- Compression Pass ---
    model = NeuralCompressorNP()
    ctx = [PAD_IDX] * CONTEXT_SIZE

    encoded_chunks = []
    chunk_stats = []

    for idx, chunk in enumerate(chunks):
        # Checkpoint reset logic
        if reset_interval > 0 and idx > 0 and idx % reset_interval == 0:
            model = NeuralCompressorNP()
            ctx = [PAD_IDX] * CONTEXT_SIZE

        bitstream, ctx, elapsed = process_chunk(chunk, model, ctx, mode_encode=True, lr=lr)
        encoded_chunks.append(bitstream)

        in_bytes = len(chunk)
        out_bytes = len(bitstream)
        bits_per_byte = (out_bytes * 8) / in_bytes if in_bytes > 0 else 0

        chunk_stats.append({
            "chunk": idx,
            "in_bytes": in_bytes,
            "out_bytes": out_bytes,
            "bpb": bits_per_byte,
            "time": elapsed
        })

        print(f"chunk {idx:3d} | {in_bytes:6d} bytes | {out_bytes:6d} compressed | {bits_per_byte:5.2f} bits/byte | {elapsed:5.2f}s")

    # Serialize overall payload
    # Header format: [Chunk Count: 4B] + List of [Chunk Raw Len: 4B][Bitstream Len: 4B][Bitstream]
    payload = bytearray(struct.pack(">I", len(chunks)))
    for orig_chunk, bstream in zip(chunks, encoded_chunks):
        payload.extend(struct.pack(">II", len(orig_chunk), len(bstream)))
        payload.extend(bstream)

    total_in = sum(s["in_bytes"] for s in chunk_stats)
    total_out = sum(s["out_bytes"] for s in chunk_stats)
    overall_bpb = (total_out * 8) / total_in if total_in > 0 else 0
    print("-" * 70)
    print(f"TOTAL: {total_in} bytes -> {total_out} compressed bytes ({overall_bpb:.2f} bits/byte)")

    # --- Verification Decompression Pass ---
    print("\n[*] Verifying bit-exact reconstruction...")
    dec_model = NeuralCompressorNP()
    dec_ctx = [PAD_IDX] * CONTEXT_SIZE

    reconstructed_bytes = bytearray()

    # Parse payload
    num_chunks = struct.unpack(">I", payload[:4])[0]
    ptr = 4

    for idx in range(num_chunks):
        if reset_interval > 0 and idx > 0 and idx % reset_interval == 0:
            dec_model = NeuralCompressorNP()
            dec_ctx = [PAD_IDX] * CONTEXT_SIZE

        orig_c_len, bstream_len = struct.unpack(">II", payload[ptr : ptr + 8])
        ptr += 8
        bstream = payload[ptr : ptr + bstream_len]
        ptr += bstream_len

        decoder = RangeDecoder(bstream)
        # Reconstruct chunk
        t0 = time.time()
        for _ in range(orig_c_len):
            logits, cache = dec_model.forward(dec_ctx)
            freqs = logits_to_freqs(logits)
            cum = list(accumulate(freqs, initial=0))
            symbol = decoder.decode_symbol(freqs, cum)
            reconstructed_bytes.append(symbol)
            dec_model.backward_and_step(cache, symbol, lr)
            dec_ctx = dec_ctx[1:] + [symbol]

    # Verification checks
    orig_hash = hashlib.sha256(data).hexdigest()
    recon_hash = hashlib.sha256(reconstructed_bytes).hexdigest()

    print(f"Original SHA-256     : {orig_hash}")
    print(f"Reconstructed SHA-256: {recon_hash}")
    assert orig_hash == recon_hash, "CRITICAL ERROR: Bit-exact verification failed!"
    print("VERIFICATION SUCCESS: Bit-exact parity verified across decoder/encoder.")

    return chunk_stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_persistent_state.py <input_file> [chunk_size_kb] [checkpoint_interval]")
        print("Example: python test_persistent_state.py sample.txt 64 100")
        sys.exit(1)

    input_file = sys.argv[1]
    chunk_size_kb = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    checkpoint_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    chunk_bytes = chunk_size_kb * 1024

    # Run Experiment A: RESET mode
    stats_reset = run_experiment(input_file, chunk_size=chunk_bytes, reset_interval=1)

    # Run Experiment B: PERSISTENT mode
    stats_persist = run_experiment(input_file, chunk_size=chunk_bytes, reset_interval=0)

    # Optional Experiment C: Checkpointed Mode
    if len(stats_persist) > checkpoint_interval:
        run_experiment(input_file, chunk_size=chunk_bytes, reset_interval=checkpoint_interval)


if __name__ == "__main__":
    main()
