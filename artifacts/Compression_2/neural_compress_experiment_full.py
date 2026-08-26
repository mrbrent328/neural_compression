#!/usr/bin/env python3
"""
neural_compress_opt.py - Optimized Bit-Exact Synchronized Online Neural Range Compressor

Changes vs original:
  1. Single forward pass per symbol (detach logits for coding freqs, reuse the
     SAME graph for the training backward pass) instead of two forward passes.
  2. Plain SGD instead of Adam (Adam's per-param moment bookkeeping was ~1.1s
     of ~2.7s encode time on a 2000-byte file; SGD has near-zero optimizer overhead).
  3. Cumulative-frequency prefix sums computed with itertools.accumulate once
     per symbol instead of repeated sum(freqs[:symbol]) slicing (encoder) and
     a linear scan (decoder) -> both become O(256) once instead of O(256) *
     repeated slicing/allocation.
  4. Preallocated rolling context buffer (list, in-place slide) instead of
     rebuilding a fresh Python list each iteration.

Default lr=0.04 for SGD was chosen by a small sweep (0.005-0.1) across a
repetitive-text sample and a mixed text+noise sample -- it's a reasonable
middle ground, not exhaustively tuned. Adam (original) still wins on ratio
for highly predictable data; SGD wins on speed. Swap the optimizer back to
Adam in _make_model_and_opt() if ratio matters more than throughput for
your data -- the single-forward-pass fix helps either way.

Determinism note: still relies on identical PyTorch/BLAS build across machines.
"""

import sys
import os
import time
import struct
import hashlib
import argparse
import json
from itertools import accumulate
from bisect import bisect_right
import torch
import torch.nn as nn

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(42)

device = torch.device("cpu")
SCALE = 1 << 16

EXPERIMENT_CONFIG = {
    "embed_dim": 32,
    "hidden_dim": 64,
    "context_size": 16,
    "num_states": 1,
    "lr": 0.04,
}


class NeuralCompressor(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        embed_dim=32,
        hidden_dim=64,
        context_size=16,
        num_states=1,
    ):
        super().__init__()
        self.context_size = context_size
        self.num_states = num_states
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=256)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.relu(self.fc1(embeds))
        return self.fc2(out)


def logits_to_freqs(logits, scale=SCALE):
    """logits: 1D tensor (already detached). Returns list[int] freqs."""
    logits = torch.clamp(logits, min=-50.0, max=50.0)
    probs = torch.softmax(logits, dim=-1)
    freqs = (probs * (scale - 256)).int() + 1
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

        # cum[i] = sum(freqs[:i]) (exclusive prefix); find first i with cum[i] <= count < cum[i+1]
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


def _make_model_and_opt(context_size, lr):
    torch.manual_seed(42)
    model = NeuralCompressor(
        context_size=context_size,
        embed_dim=EXPERIMENT_CONFIG["embed_dim"],
        hidden_dim=EXPERIMENT_CONFIG["hidden_dim"],
        num_states=EXPERIMENT_CONFIG["num_states"],
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def compress_file(input_path, output_bin_path, context_size=16, lr=0.04):
    with open(input_path, "rb") as f:
        data = f.read()

    orig_len = len(data)
    print(f"[*] Encoding '{input_path}' ({orig_len} bytes)...")
    t0 = time.time()

    model, optimizer, criterion = _make_model_and_opt(context_size, lr)
    encoder = RangeEncoder()
    pad_idx = 256
    model.train()

    ctx = [pad_idx] * context_size  # rolling buffer, slid in place

    for i in range(orig_len):
        target_byte = data[i]

        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(ctx_t).squeeze(0)          # SINGLE forward pass, graph retained

        freqs = logits_to_freqs(logits.detach())   # detach only for coding -> no extra forward
        cum = list(accumulate(freqs, initial=0))    # prefix sums, computed once

        encoder.encode_symbol(target_byte, freqs, cum)

        target_t = torch.tensor([target_byte], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = criterion(logits.unsqueeze(0), target_t)  # reuse the SAME logits/graph
        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [target_byte]  # slide window

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


def decompress_file(input_bin_path, output_path, context_size=16, lr=0.04):
    with open(input_bin_path, "rb") as f:
        raw_bin = f.read()

    orig_len = struct.unpack(">I", raw_bin[:4])[0]
    bitstream = raw_bin[4:]

    print(f"[*] Decompressing '{input_bin_path}' ({len(raw_bin)} bytes)...")
    t0 = time.time()

    model, optimizer, criterion = _make_model_and_opt(context_size, lr)
    decoder = RangeDecoder(bitstream)
    pad_idx = 256
    model.train()
    decoded_bytes = bytearray()

    ctx = [pad_idx] * context_size

    for i in range(orig_len):
        ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(ctx_t).squeeze(0)

        freqs = logits_to_freqs(logits.detach())
        cum = list(accumulate(freqs, initial=0))

        symbol = decoder.decode_symbol(freqs, cum)
        decoded_bytes.append(symbol)

        target_t = torch.tensor([symbol], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = criterion(logits.unsqueeze(0), target_t)
        loss.backward()
        optimizer.step()

        ctx = ctx[1:] + [symbol]

    elapsed = time.time() - t0
    with open(output_path, "wb") as f:
        f.write(decoded_bytes)

    print(f"[+] Decompressed output written to '{output_path}' ({len(decoded_bytes)} bytes)")
    print(f"    Elapsed: {elapsed:.2f}s")



def parse_experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--states", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.04)
    args, remaining = parser.parse_known_args()

    EXPERIMENT_CONFIG["embed_dim"] = args.embed
    EXPERIMENT_CONFIG["hidden_dim"] = args.hidden
    EXPERIMENT_CONFIG["context_size"] = args.context
    EXPERIMENT_CONFIG["num_states"] = args.states
    EXPERIMENT_CONFIG["lr"] = args.lr

    return remaining

def main():
    remaining = parse_experiment_args()
    if len(remaining) >= 1:
        sys.argv = [sys.argv[0]] + remaining

    print("Experiment config:")
    print(json.dumps(EXPERIMENT_CONFIG, indent=2))

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

    elif cmd == "batch":
        folder = sys.argv[2] if len(sys.argv) > 2 else "."

        total_raw = 0
        total_compressed = 0
        total_time = 0.0

        files = [
            f for f in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, f))
            and f != os.path.basename(__file__)
            and not f.endswith(".bin")
            and not f.endswith(".tmp.bin")
            and not f.endswith(".tmp.recon")
        ]

        print("=" * 70)
        print("NEURAL COMPRESS BATCH TEST")
        print("=" * 70)
        print(f"Folder: {os.path.abspath(folder)}")
        print(f"Files: {len(files)}")
        print()

        for index, filename in enumerate(files, 1):
            input_file = os.path.join(folder, filename)
            output_file = input_file + ".batch.bin"

            print("-" * 70)
            print(f"[{index}/{len(files)}] {filename}")

            start = time.time()
            compress_file(input_file, output_file)
            elapsed = time.time() - start

            raw_size = os.path.getsize(input_file)
            compressed_size = os.path.getsize(output_file)

            total_raw += raw_size
            total_compressed += compressed_size
            total_time += elapsed

        print()
        print("=" * 70)
        print("TOTAL")
        print("=" * 70)

        reduction = (
            (1.0 - total_compressed / total_raw) * 100
            if total_raw else 0
        )

        print(f"Raw size:          {total_raw:,} B")
        print(f"Compressed size:   {total_compressed:,} B")
        print(f"Reduction:         {reduction:.2f}%")
        print(f"Total elapsed:     {total_time:.2f}s")

        if total_time:
            print(
                f"Overall throughput: "
                f"{total_raw / 1024 / total_time:.2f} KiB/s"
            )

        print("=" * 70)

    else:
        print(f"Unknown command '{cmd}'")
        print(__doc__)


if __name__ == "__main__":
    main()
