#!/usr/bin/env python3
"""
neural_escape_compress.py - Bit-Exact 2-Class Neural Compression with Escape Symbols

Compresses data where one byte dominates (e.g. 99% 'A'). The neural model predicts
a binary distribution [Dominant, Escape]. When an Escape symbol is encoded, the
exact byte value is written as a literal.
"""

import sys
import os
import time
import struct
import hashlib
from itertools import accumulate
from bisect import bisect_right

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

SCALE = 1 << 16
CONTEXT_SIZE = 16
EMBED_DIM = 32
HIDDEN_DIM = 64
PAD_IDX = 256
DOMINANT_BYTE = ord('A')  # Target dominant byte (e.g., 0x41)


# =====================================================================
# 2-Class Neural Predictor & Range Coder
# =====================================================================

class NeuralCompressorEscapeNP:
    """Embedding(257,32) -> concat -> Linear(512,64) -> ReLU -> Linear(64,2)"""

    def __init__(self, dominant_byte=DOMINANT_BYTE, seed=42):
        self.dominant_byte = dominant_byte
        rng = np.random.default_rng(seed)
        in_dim = CONTEXT_SIZE * EMBED_DIM

        self.E = rng.standard_normal((257, EMBED_DIM)).astype(np.float64)
        self.E[PAD_IDX] = 0.0

        bound1 = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.uniform(-bound1, bound1, (HIDDEN_DIM, in_dim)).astype(np.float64)
        self.b1 = rng.uniform(-bound1, bound1, HIDDEN_DIM).astype(np.float64)

        bound2 = 1.0 / np.sqrt(HIDDEN_DIM)
        self.W2 = rng.uniform(-bound2, bound2, (2, HIDDEN_DIM)).astype(np.float64)
        self.b2 = rng.uniform(-bound2, bound2, 2).astype(np.float64)

    def forward(self, ctx_idx):
        emb = self.E[ctx_idx].reshape(-1)
        h_pre = self.W1 @ emb + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = self.W2 @ h + self.b2
        return logits, (ctx_idx, emb, h_pre, h)

    def backward_and_step(self, cache, is_escape: bool, lr: float):
        if lr <= 0.0:
            return

        ctx_idx, emb, h_pre, h = cache
        target = 1 if is_escape else 0

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

        demb = demb.reshape(CONTEXT_SIZE, EMBED_DIM)
        for pos, idx in enumerate(ctx_idx):
            if idx != PAD_IDX:
                self.E[idx] -= lr * demb[pos]


def binary_logits_to_freqs(logits, scale=SCALE):
    z = np.clip(logits, -50.0, 50.0)
    z = z - z.max()
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum()

    freq0 = int(probs[0] * (scale - 2)) + 1
    freq1 = scale - freq0
    return [freq0, freq1]


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
# Compression & Decompression Pipelines
# =====================================================================

def compress_escape(data: bytes, dominant_byte=DOMINANT_BYTE, lr=0.04):
    model = NeuralCompressorEscapeNP(dominant_byte=dominant_byte)
    encoder = RangeEncoder()
    ctx = [PAD_IDX] * CONTEXT_SIZE
    literals = bytearray()

    t0 = time.time()
    for target_byte in data:
        logits, cache = model.forward(ctx)
        freqs = binary_logits_to_freqs(logits)
        cum = [0, freqs[0], freqs[0] + freqs[1]]

        if target_byte == dominant_byte:
            # Symbol 0: Dominant Byte
            encoder.encode_symbol(0, freqs, cum)
            model.backward_and_step(cache, is_escape=False, lr=lr)
        else:
            # Symbol 1: Escape Symbol
            encoder.encode_symbol(1, freqs, cum)
            model.backward_and_step(cache, is_escape=True, lr=lr)
            literals.append(target_byte)

        ctx = ctx[1:] + [target_byte]

    bitstream = encoder.finish()
    elapsed = time.time() - t0

    # Payload Format: [Orig Len: 4B][Dominant Byte: 1B][Literals Len: 4B][Bitstream Len: 4B][Literals][Bitstream]
    header = struct.pack(">IBII", len(data), dominant_byte, len(literals), len(bitstream))
    packed_payload = header + bytes(literals) + bitstream
    return packed_payload, elapsed


def decompress_escape(packed_payload: bytes, lr=0.04):
    orig_len, dominant_byte, lit_len, bit_len = struct.unpack(">IBII", packed_payload[:13])
    ptr = 13

    literals = packed_payload[ptr : ptr + lit_len]
    ptr += lit_len
    bitstream = packed_payload[ptr : ptr + bit_len]

    model = NeuralCompressorEscapeNP(dominant_byte=dominant_byte)
    decoder = RangeDecoder(bitstream)
    ctx = [PAD_IDX] * CONTEXT_SIZE

    decoded_bytes = bytearray()
    lit_ptr = 0

    t0 = time.time()
    for _ in range(orig_len):
        logits, cache = model.forward(ctx)
        freqs = binary_logits_to_freqs(logits)
        cum = [0, freqs[0], freqs[0] + freqs[1]]

        symbol = decoder.decode_symbol(freqs, cum)

        if symbol == 0:
            actual_byte = dominant_byte
            model.backward_and_step(cache, is_escape=False, lr=lr)
        else:
            actual_byte = literals[lit_ptr]
            lit_ptr += 1
            model.backward_and_step(cache, is_escape=True, lr=lr)

        decoded_bytes.append(actual_byte)
        ctx = ctx[1:] + [actual_byte]

    elapsed = time.time() - t0
    return bytes(decoded_bytes), elapsed


# =====================================================================
# Verification Test
# =====================================================================

if __name__ == "__main__":
    # Generate Synthetic Data: 99% 'A', 0.5% 'B', 0.5% 'C'
    total_size = 100000
    print(f"[*] Generating {total_size} bytes of skewed data (99% 'A', 0.5% 'B', 0.5% 'C')...")
    rng = np.random.default_rng(42)
    choices = rng.choice([ord('A'), ord('B'), ord('C')], size=total_size, p=[0.99, 0.005, 0.005])
    test_data = bytes(choices)

    print("\n[*] Compressing with NeuralCompressorEscapeNP...")
    compressed_payload, c_time = compress_escape(test_data, dominant_byte=ord('A'))
    raw_size = len(test_data)
    comp_size = len(compressed_payload)
    bpb = (comp_size * 8) / raw_size

    print(f"[+] Compression Complete in {c_time:.2f}s")
    print(f"    Raw Size       : {raw_size} bytes")
    print(f"    Compressed Size: {comp_size} bytes")
    print(f"    Bitrate        : {bpb:.3f} bits/byte")

    print("\n[*] Decompressing payload...")
    decompressed_data, d_time = decompress_escape(compressed_payload)
    print(f"[+] Decompression Complete in {d_time:.2f}s")

    orig_hash = hashlib.sha256(test_data).hexdigest()
    recon_hash = hashlib.sha256(decompressed_data).hexdigest()

    print("-" * 65)
    print(f"Original SHA-256     : {orig_hash}")
    print(f"Reconstructed SHA-256: {recon_hash}")
    assert orig_hash == recon_hash, "CRITICAL ERROR: Mismatch between original and decoded data!"
    print("RESULT: EXACT BIT-EXACT MATCH (PASS)")
