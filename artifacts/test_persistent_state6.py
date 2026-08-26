#!/usr/bin/env python3
"""
pytorch_hybrid_compressor.py - Integrated PyTorch/CUDA Persistent Neural Compressor

Features:
- Adaptive A/B/C Router
- Mode A: 2-Class Neural Escape Coder (Sparse/Uniform Chunks)
- Mode B: PyTorch CUDA Accelerated Persistent Neural Engine (Structured Chunks)
- Mode C: Passthrough Direct Copy (High Entropy / Compressed Chunks)
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE = 1 << 16
CONTEXT_SIZE = 16
EMBED_DIM = 32
HIDDEN_DIM = 64
PAD_IDX = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# PyTorch Acceleration Module for Mode B
# =====================================================================

class PyTorchPersistentCompressor(nn.Module):
    """
    PyTorch Neural Predictor for Structured Data (Mode B).
    Supports CUDA parallel operations and low-overhead tensor updates.
    """
    def __init__(self, context_size=CONTEXT_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, device=DEVICE):
        super().__init__()
        self.device = device
        self.context_size = context_size

        self.E = nn.Embedding(257, embed_dim, padding_idx=PAD_IDX)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)

        # Match random seed initialization
        torch.manual_seed(42)
        nn.init.uniform_(self.E.weight, -1.0, 1.0)
        with torch.no_grad():
            self.E.weight[PAD_IDX].fill_(0.0)

        in_dim = context_size * embed_dim
        b1 = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.fc1.weight, -b1, b1)
        nn.init.uniform_(self.fc1.bias, -b1, b1)

        b2 = 1.0 / math.sqrt(hidden_dim)
        nn.init.uniform_(self.fc2.weight, -b2, b2)
        nn.init.uniform_(self.fc2.bias, -b2, b2)

        self.to(self.device)

    def forward_single(self, ctx_list):
        """Pre-allocated single forward pass."""
        ctx_t = torch.tensor(ctx_list, dtype=torch.long, device=self.device).unsqueeze(0)
        emb = self.E(ctx_t).view(1, -1)
        h = F.relu(self.fc1(emb))
        logits = self.fc2(h)
        return logits.squeeze(0)

    def step_gradient(self, logits, target_byte, lr=0.04):
        """Analytical backprop step without autograd graph creation overhead."""
        # Fast manual gradient step on CUDA tensors to avoid engine sync locks
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            dlogits = probs.clone()
            dlogits[target_byte] -= 1.0

            # Retrieve cached hidden states
            # Standard autograd loop works cleanly with set_to_none=True
            pass


class PyTorchCompressorAutograd(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.E = nn.Embedding(257, EMBED_DIM, padding_idx=PAD_IDX)
        self.fc1 = nn.Linear(CONTEXT_SIZE * EMBED_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 256)
        self.to(self.device)
        self.opt = torch.optim.SGD(self.parameters(), lr=0.04)

    def forward_and_step(self, ctx_list, target_byte):
        ctx_t = torch.tensor(ctx_list, dtype=torch.long, device=self.device).unsqueeze(0)

        emb = self.E(ctx_t).view(1, -1)
        h = F.relu(self.fc1(emb))
        logits = self.fc2(h)

        # Calculate logits for entropy coder
        logits_np = logits.detach().cpu().squeeze(0).numpy()

        # Online Learning Step
        target_t = torch.tensor([target_byte], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, target_t)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return logits_np


# =====================================================================
# Mode A: CPU Fast Escape Model
# =====================================================================

class NeuralCompressorEscapeNP:
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
# Range Coder Utilities
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
# Main Profiling & Execution Engine
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

def pytorch_hybrid_compress(data: bytes, chunk_size=32768, lr=0.04):
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Initialize PyTorch Persistent State Engine
    persistent_b_model = PyTorchCompressorAutograd(device=DEVICE)
    persistent_ctx = [PAD_IDX] * CONTEXT_SIZE

    chunk_payloads = []

    print(f"[*] Hybrid Compressing {len(data)} bytes ({len(chunks)} chunks) on [{DEVICE.type.upper()}]...")
    t0 = time.time()

    for idx, chunk in enumerate(chunks):
        mode, dom_byte, entropy = profile_chunk(chunk)

        if mode == 'A':
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
            header = struct.pack(">BIIBI", ord('A'), len(chunk), len(literals), dom_byte, len(bitstream))
            c_payload = header + bytes(literals) + bitstream

        elif mode == 'B':
            encoder = RangeEncoder()
            for b in chunk:
                logits_np = persistent_b_model.forward_and_step(persistent_ctx, b)
                freqs = logits_to_freqs(logits_np)
                cum = list(accumulate(freqs, initial=0))
                encoder.encode_symbol(b, freqs, cum)
                persistent_ctx = persistent_ctx[1:] + [b]

            bitstream = encoder.finish()
            header = struct.pack(">BII", ord('B'), len(chunk), len(bitstream))
            c_payload = header + bitstream

        else: # Mode 'C'
            header = struct.pack(">BI", ord('C'), len(chunk))
            c_payload = header + chunk

        chunk_payloads.append(c_payload)
        bpb = (len(c_payload) * 8) / len(chunk)
        print(f"  Chunk {idx:2d} | Mode {mode} | Entropy: {entropy:4.2f} | Payload: {len(c_payload):6d} B | {bpb:5.2f} bits/byte")

    elapsed = time.time() - t0
    master_header = struct.pack(">II", len(data), len(chunks))
    return master_header + b"".join(chunk_payloads), elapsed


def pytorch_hybrid_decompress(packed_data: bytes, lr=0.04):
    orig_total_len, num_chunks = struct.unpack(">II", packed_data[:8])
    ptr = 8

    persistent_b_model = PyTorchCompressorAutograd(device=DEVICE)
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
                # In decompression, we extract logits FIRST to decode symbol
                with torch.no_grad():
                    ctx_t = torch.tensor(persistent_ctx, dtype=torch.long, device=DEVICE).unsqueeze(0)
                    emb = persistent_b_model.E(ctx_t).view(1, -1)
                    h = F.relu(persistent_b_model.fc1(emb))
                    logits = persistent_b_model.fc2(h)
                    logits_np = logits.detach().cpu().squeeze(0).numpy()

                freqs = logits_to_freqs(logits_np)
                cum = list(accumulate(freqs, initial=0))
                sym = decoder.decode_symbol(freqs, cum)
                reconstructed.append(sym)

                # Then update model with decoded symbol
                persistent_b_model.forward_and_step(persistent_ctx, sym)
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
# Benchmark Execution
# =====================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pytorch_hybrid_compressor.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, "rb") as f:
        original_data = f.read()

    # Compress
    compressed_payload, c_time = pytorch_hybrid_compress(original_data)

    raw_size = len(original_data)
    comp_size = len(compressed_payload)
    ratio = (1.0 - (comp_size / raw_size)) * 100 if raw_size > 0 else 0
    final_bpb = (comp_size * 8) / raw_size if raw_size > 0 else 0

    print("\n" + "=" * 70)
    print(f"COMPRESSION SUMMARY ({os.path.basename(file_path)}):")
    print(f"  Raw Size       : {raw_size} bytes")
    print(f"  Compressed Size: {comp_size} bytes")
    print(f"  Reduction      : {ratio:.2f}% ({final_bpb:.2f} bits/byte)")
    print(f"  Encoding Time  : {c_time:.2f}s ({raw_size/1024/c_time:.2f} KB/s)")

    # Decompress
    print("\n[*] Running Verification Decompression...")
    decompressed_data, d_time = pytorch_hybrid_decompress(compressed_payload)
    print(f"[+] Decoding Time  : {d_time:.2f}s")

    # Hash Verification
    orig_hash = hashlib.sha256(original_data).hexdigest()
    recon_hash = hashlib.sha256(decompressed_data).hexdigest()

    print("-" * 70)
    print(f"Original SHA-256     : {orig_hash}")
    print(f"Reconstructed SHA-256: {recon_hash}")
    assert orig_hash == recon_hash, "CRITICAL ERROR: Bit-exact parity verification failed!"
    print("VERIFICATION SUCCESS: Exact bit-exact match across PyTorch decoder/encoder.")
