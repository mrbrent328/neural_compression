#!/usr/bin/env python3
"""
neural_compress_np.py - NumPy-only Bit-Exact Online Neural Range Compressor

Same architecture and algorithm as neural_compress_opt.py, but the model's
forward and backward pass are hand-rolled in NumPy instead of PyTorch.
Rationale: the network is tiny (512 -> 64 -> 256, ~50K params) and every
PyTorch call pays fixed dispatcher/autograd-graph overhead that dwarfs the
actual FLOPs at this scale. Removing the framework entirely should win here.

No torch dependency at all -- only numpy + stdlib.
"""

import sys
import os
import time
import struct
import hashlib
from itertools import accumulate
from bisect import bisect_right

os.environ["OMP_NUM_THREADS"] = "1"  # single-threaded BLAS for reproducibility
import numpy as np

SCALE = 1 << 16
VOCAB = 256
EMBED_DIM = 32
HIDDEN_DIM = 64
CONTEXT_SIZE = 16
PAD_IDX = 256


class NeuralCompressorNP:
    """Embedding(257,32) -> concat(512) -> Linear(512,64) -> ReLU -> Linear(64,256)"""

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        in_dim = CONTEXT_SIZE * EMBED_DIM
        # match torch's default init scale roughly (kaiming-uniform-ish / embedding N(0,1))
        self.E = rng.standard_normal((VOCAB + 1, EMBED_DIM)).astype(np.float64)
        self.E[PAD_IDX] = 0.0  # padding_idx rows stay zero, like torch
        bound1 = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.uniform(-bound1, bound1, (HIDDEN_DIM, in_dim)).astype(np.float64)
        self.b1 = rng.uniform(-bound1, bound1, HIDDEN_DIM).astype(np.float64)
        bound2 = 1.0 / np.sqrt(HIDDEN_DIM)
        self.W2 = rng.uniform(-bound2, bound2, (VOCAB, HIDDEN_DIM)).astype(np.float64)
        self.b2 = rng.uniform(-bound2, bound2, VOCAB).astype(np.float64)

    def forward(self, ctx_idx):
        """ctx_idx: list[int] length CONTEXT_SIZE. Returns (logits, cache-for-backward)."""
        emb = self.E[ctx_idx].reshape(-1)          # (512,)
        h_pre = self.W1 @ emb + self.b1             # (64,)
        h = np.maximum(h_pre, 0.0)                   # ReLU
        logits = self.W2 @ h + self.b2                # (256,)
        return logits, (ctx_idx, emb, h_pre, h)

    def backward_and_step(self, cache, target, lr):
        ctx_idx, emb, h_pre, h = cache
        logits, _ = self.forward(ctx_idx) if False else (None, None)  # unused, kept explicit below

        # recompute softmax probs from the same logits path for the gradient
        # (cheap: only needed once, no autograd graph bookkeeping)
        z = self.W2 @ h + self.b2
        z = z - z.max()
        exp_z = np.exp(z)
        probs = exp_z / exp_z.sum()

        dlogits = probs.copy()
        dlogits[target] -= 1.0                      # dL/dlogits for softmax-CE

        dW2 = np.outer(dlogits, h)
        db2 = dlogits
        dh = self.W2.T @ dlogits
        dh_pre = dh * (h_pre > 0.0)

        dW1 = np.outer(dh_pre, emb)
        db1 = dh_pre
        demb = self.W1.T @ dh_pre                    # (512,)

        # SGD update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        demb = demb.reshape(CONTEXT_SIZE, EMBED_DIM)
        for pos, idx in enumerate(ctx_idx):
            if idx != PAD_IDX:                        # padding_idx rows never trained, like torch
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


def compress_file(input_path, output_bin_path, lr=0.04):
    with open(input_path, "rb") as f:
        data = f.read()
    orig_len = len(data)
    print(f"[*] Encoding '{input_path}' ({orig_len} bytes)...")
    t0 = time.time()

    model = NeuralCompressorNP()
    encoder = RangeEncoder()
    ctx = [PAD_IDX] * CONTEXT_SIZE

    for i in range(orig_len):
        target_byte = data[i]
        logits, cache = model.forward(ctx)
        freqs = logits_to_freqs(logits)
        cum = list(accumulate(freqs, initial=0))
        encoder.encode_symbol(target_byte, freqs, cum)
        model.backward_and_step(cache, target_byte, lr)
        ctx = ctx[1:] + [target_byte]

    bitstream = encoder.finish()
    elapsed = time.time() - t0

    header = struct.pack(">I", orig_len)
    with open(output_bin_path, "wb") as f:
        f.write(header + bitstream)

    final_size = len(header) + len(bitstream)
    ratio = (1.0 - (final_size / orig_len)) * 100 if orig_len > 0 else 0
    print(f"[+] Compressed output saved to '{output_bin_path}'")
    print(f"    Raw Size: {orig_len} B | Binary Size: {final_size} B | Reduction: {ratio:.2f}%")
    print(f"    Elapsed: {elapsed:.2f}s ({orig_len/1024/elapsed:.2f} KB/s)")


def decompress_file(input_bin_path, output_path, lr=0.04):
    with open(input_bin_path, "rb") as f:
        raw_bin = f.read()
    orig_len = struct.unpack(">I", raw_bin[:4])[0]
    bitstream = raw_bin[4:]
    print(f"[*] Decompressing '{input_bin_path}' ({len(raw_bin)} bytes)...")
    t0 = time.time()

    model = NeuralCompressorNP()
    decoder = RangeDecoder(bitstream)
    ctx = [PAD_IDX] * CONTEXT_SIZE
    decoded_bytes = bytearray()

    for i in range(orig_len):
        logits, cache = model.forward(ctx)
        freqs = logits_to_freqs(logits)
        cum = list(accumulate(freqs, initial=0))
        symbol = decoder.decode_symbol(freqs, cum)
        decoded_bytes.append(symbol)
        model.backward_and_step(cache, symbol, lr)
        ctx = ctx[1:] + [symbol]

    elapsed = time.time() - t0
    with open(output_path, "wb") as f:
        f.write(decoded_bytes)
    print(f"[+] Decompressed output written to '{output_path}' ({len(decoded_bytes)} bytes)")
    print(f"    Elapsed: {elapsed:.2f}s")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1].lower()

    if cmd == "compress":
        input_file = sys.argv[2]
        output_bin = sys.argv[3] if len(sys.argv) > 3 else input_file + ".bin"
        compress_file(input_file, output_bin)
    elif cmd == "decompress":
        input_bin = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "reconstructed_file"
        decompress_file(input_bin, output_file)
    elif cmd == "test":
        input_file = sys.argv[2]
        bin_file = input_file + ".tmp.bin"
        recon_file = input_file + ".tmp.recon"
        compress_file(input_file, bin_file)
        print("-" * 65)
        decompress_file(bin_file, recon_file)
        with open(input_file, "rb") as f:
            orig_hash = hashlib.sha256(f.read()).hexdigest()
        with open(recon_file, "rb") as f:
            recon_hash = hashlib.sha256(f.read()).hexdigest()
        print("-" * 65)
        print(f"Original SHA-256     : {orig_hash}")
        print(f"Reconstructed SHA-256: {recon_hash}")
        assert orig_hash == recon_hash, "CRITICAL ERROR: Reconstruction mismatch!"
        print("RESULT: EXACT BIT-EXACT MATCH (PASS)")
        if os.path.exists(bin_file):
            os.remove(bin_file)
        if os.path.exists(recon_file):
            os.remove(recon_file)
    else:
        print(f"Unknown command '{cmd}'")
        print(__doc__)


if __name__ == "__main__":
    main()
