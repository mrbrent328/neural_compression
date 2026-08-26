#!/usr/bin/env python3
"""
hybrid_persistent_compressor.py - Integrated Adaptive Neural Compressor

Combines:
1. A/B/C Structural Profiling (Fast Routing)
2. Mode A: 2-Class Neural Escape Coder (Sparse/Uniform Chunks)
3. Mode B: Persistent 256-Class Neural Compressor (Structured Chunks)
4. Mode C: Direct Passthrough (High-Entropy Chunks)
"""

import sys
import os
import time
import struct
import math
import hashlib
from collections import Counter
from itertools import accumulate
from bisect import bisect_right

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

SCALE = 1 << 16
CONTEXT_SIZE = 16
EMBED_DIM = 32
HIDDEN_DIM = 64
PAD_IDX = 256

# =====================================================================
# Core Neural Engines
# =====================================================================

class NeuralCompressorNP:
    """Full 256-Class Neural Predictor for Structured Data (Mode B)"""
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        in_dim = CONTEXT_SIZE * EMBED_DIM
        self.E = rng.standard_normal((257, EMBED_DIM)).astype(np.float64)
        self.E[PAD_IDX] = 0.0
        bound1 = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.uniform(-bound1, bound1, (HIDDEN_DIM, in_dim)).astype(np.float64)
        self.b1 = rng.uniform(-bound1, bound1, HIDDEN_DIM).astype(np.float64)
        bound2 = 1.0 / np.sqrt(HIDDEN_DIM)
        self.W2 = rng.uniform(-bound2, bound2, (256, HIDDEN_DIM)).astype(np.float64)
        self.b2 = rng.uniform(-bound2, bound2, 256).astype(np.float64)

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

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        demb = demb.reshape(CONTEXT_SIZE, EMBED_DIM)
        for pos, idx in enumerate(ctx_idx):
            if idx != PAD_IDX:
                self.E[idx] -= lr * demb[pos]


class NeuralCompressorEscapeNP:
    """Fast 2-Class Neural Predictor for Uniform/Sparse Data (Mode A)"""
    def __init__(self, dominant_byte, seed=42):
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


# =====================================================================
# Utilities & Entropy Coding
# =====================================================================

def logits_to_freqs(logits, scale=SCALE):
    z = np.clip(logits, -50.0, 50.0)
    z = z - z.max()
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum()
    freqs = (probs * (scale - 256)).astype(np.int64) + 1
    return freqs.tolist()

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
# Integrated Pipeline Execution
# =====================================================================

def profile_chunk(chunk: bytes):
    if not chunk:
        return 'A', 0, 0.0
    counts = Counter(chunk)
    dom_byte, freq = counts.most_common(1)[0]
    dom_ratio = freq / len(chunk)

    entropy = 0.0
    for count in counts.values():
        p = count / len(chunk)
        entropy -= p * math.log2(p)

    if dom_ratio > 0.70 or entropy < 2.5:
        return 'A', dom_byte, entropy
    elif entropy > 6.2:
        return 'C', dom_byte, entropy
    else:
        return 'B', dom_byte, entropy

def hybrid_compress(data: bytes, chunk_size=32768, lr=0.04):
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Persistent Model State surviving across Type-B Chunks
    persistent_b_model = NeuralCompressorNP()
    persistent_ctx = [PAD_IDX] * CONTEXT_SIZE

    chunk_payloads = []

    print(f"[*] Hybrid Compressing {len(data)} bytes ({len(chunks)} chunks)...")
    t0 = time.time()

    for idx, chunk in enumerate(chunks):
        mode, dom_byte, entropy = profile_chunk(chunk)

        if mode == 'A':
            # Fast Escape Mode
            model_a = NeuralCompressorEscapeNP(dominant_byte=dom_byte)
            encoder = RangeEncoder()
            ctx_a = [PAD_IDX] * CONTEXT_SIZE
            literals = bytearray()

            for b in chunk:
                logits, cache = model_a.forward(ctx_a)
                freqs = binary_logits_to_freqs(logits)
                cum = [0, freqs[0], freqs[0] + freqs[1]]

                if b == dom_byte:
                    encoder.encode_symbol(0, freqs, cum)
                    model_a.backward_and_step(cache, is_escape=False, lr=lr)
                else:
                    encoder.encode_symbol(1, freqs, cum)
                    model_a.backward_and_step(cache, is_escape=True, lr=lr)
                    literals.append(b)
                ctx_a = ctx_a[1:] + [b]

            bitstream = encoder.finish()
            # Header: Mode 'A' (1B), RawLen (4B), DomByte (1B), LitLen (4B), BitLen (4B)
            header = struct.pack(">BIIBI", ord('A'), len(chunk), len(literals), dom_byte, len(bitstream))
            c_payload = header + bytes(literals) + bitstream

        elif mode == 'B':
            # Persistent Neural Mode
            encoder = RangeEncoder()
            for b in chunk:
                logits, cache = persistent_b_model.forward(persistent_ctx)
                freqs = logits_to_freqs(logits)
                cum = list(accumulate(freqs, initial=0))
                encoder.encode_symbol(b, freqs, cum)
                persistent_b_model.backward_and_step(cache, b, lr)
                persistent_ctx = persistent_ctx[1:] + [b]

            bitstream = encoder.finish()
            # Header: Mode 'B' (1B), RawLen (4B), BitLen (4B)
            header = struct.pack(">BII", ord('B'), len(chunk), len(bitstream))
            c_payload = header + bitstream

        else: # Mode 'C'
            # Direct Passthrough Mode
            header = struct.pack(">BI", ord('C'), len(chunk))
            c_payload = header + chunk

        chunk_payloads.append(c_payload)
        bpb = (len(c_payload) * 8) / len(chunk)
        print(f"  Chunk {idx:2d} | Mode {mode} | Entropy: {entropy:4.2f} | Payload: {len(c_payload):6d} B | {bpb:5.2f} bits/byte")

    elapsed = time.time() - t0

    # Master Package
    master_header = struct.pack(">II", len(data), len(chunks))
    final_stream = master_header + b"".join(chunk_payloads)
    return final_stream, elapsed

def hybrid_decompress(packed_data: bytes, lr=0.04):
    orig_total_len, num_chunks = struct.unpack(">II", packed_data[:8])
    ptr = 8

    persistent_b_model = NeuralCompressorNP()
    persistent_ctx = [PAD_IDX] * CONTEXT_SIZE

    reconstructed = bytearray()
    t0 = time.time()

    for _ in range(num_chunks):
        mode_flag = chr(packed_data[ptr])
        ptr += 1

        if mode_flag == 'A':
            raw_len, lit_len, dom_byte, bit_len = struct.unpack(">IIBI", packed_data[ptr : ptr + 13])
            ptr += 13
            literals = packed_data[ptr : ptr + lit_len]
            ptr += lit_len
            bitstream = packed_data[ptr : ptr + bit_len]
            ptr += bit_len

            model_a = NeuralCompressorEscapeNP(dominant_byte=dom_byte)
            decoder = RangeDecoder(bitstream)
            ctx_a = [PAD_IDX] * CONTEXT_SIZE
            lit_ptr = 0

            for _ in range(raw_len):
                logits, cache = model_a.forward(ctx_a)
                freqs = binary_logits_to_freqs(logits)
                cum = [0, freqs[0], freqs[0] + freqs[1]]
                sym = decoder.decode_symbol(freqs, cum)

                if sym == 0:
                    b = dom_byte
                    model_a.backward_and_step(cache, is_escape=False, lr=lr)
                else:
                    b = literals[lit_ptr]
                    lit_ptr += 1
                    model_a.backward_and_step(cache, is_escape=True, lr=lr)

                reconstructed.append(b)
                ctx_a = ctx_a[1:] + [b]

        elif mode_flag == 'B':
            raw_len, bit_len = struct.unpack(">II", packed_data[ptr : ptr + 8])
            ptr += 8
            bitstream = packed_data[ptr : ptr + bit_len]
            ptr += bit_len

            decoder = RangeDecoder(bitstream)
            for _ in range(raw_len):
                logits, cache = persistent_b_model.forward(persistent_ctx)
                freqs = logits_to_freqs(logits)
                cum = list(accumulate(freqs, initial=0))
                sym = decoder.decode_symbol(freqs, cum)

                reconstructed.append(sym)
                persistent_b_model.backward_and_step(cache, sym, lr)
                persistent_ctx = persistent_ctx[1:] + [sym]

        elif mode_flag == 'C':
            raw_len = struct.unpack(">I", packed_data[ptr : ptr + 4])[0]
            ptr += 4
            raw_chunk = packed_data[ptr : ptr + raw_len]
            ptr += raw_len
            reconstructed.extend(raw_chunk)

    elapsed = time.time() - t0
    return bytes(reconstructed), elapsed

# =====================================================================
# Verification Routine
# =====================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hybrid_persistent_compressor.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, "rb") as f:
        original_data = f.read()

    # Compress
    compressed_payload, c_time = hybrid_compress(original_data)

    raw_size = len(original_data)
    comp_size = len(compressed_payload)
    ratio = (1.0 - (comp_size / raw_size)) * 100 if raw_size > 0 else 0
    final_bpb = (comp_size * 8) / raw_size if raw_size > 0 else 0

    print("\n" + "=" * 70)
    print(f"COMPRESSION SUMMARY:")
    print(f"  Raw Size       : {raw_size} bytes")
    print(f"  Compressed Size: {comp_size} bytes")
    print(f"  Reduction      : {ratio:.2f}% ({final_bpb:.2f} bits/byte)")
    print(f"  Encoding Time  : {c_time:.2f}s ({raw_size/1024/c_time:.2f} KB/s)")

    # Decompress
    print("\n[*] Running Verification Decompression...")
    decompressed_data, d_time = hybrid_decompress(compressed_payload)
    print(f"[+] Decoding Time  : {d_time:.2f}s")

    # Hash Verification
    orig_hash = hashlib.sha256(original_data).hexdigest()
    recon_hash = hashlib.sha256(decompressed_data).hexdigest()

    print("-" * 70)
    print(f"Original SHA-256     : {orig_hash}")
    print(f"Reconstructed SHA-256: {recon_hash}")
    assert orig_hash == recon_hash, "CRITICAL ERROR: Bit-exact parity verification failed!"
    print("VERIFICATION SUCCESS: Exact bit-exact match across decoder/encoder.")
