#!/usr/bin/env python3
"""
neural_compress.py - Fast Deterministic Neural Range Compressor
=============================================================================
Usage:
    python3 neural_compress.py compress <input_file> <output_bin>
    python3 neural_compress.py decompress <input_bin> <output_file>
    python3 neural_compress.py test <input_file>
"""

import sys
import os
import time
import struct
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim

# Environment determinism overrides
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(42)

# Auto-detect GPU/ROCm device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 1 << 16
BLOCK_SIZE = 8192  # Batch chunk size for high-speed GPU execution


# =============================================================================
# 1. NEURAL PREDICTOR MODEL
# =============================================================================
class NeuralCompressor(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, hidden_dim=64, context_size=16):
        super().__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=256)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x).view(x.size(0), -1)
        out = self.relu(self.fc1(embeds))
        return self.fc2(out)


def get_quantized_frequencies_batch(model, contexts_tensor, scale=SCALE):
    """Calculates deterministic quantized frequency distributions in batch."""
    with torch.no_grad():
        logits = model(contexts_tensor)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        probs = torch.softmax(logits, dim=-1)
        freqs = (probs * (scale - 256)).int() + 1
        return freqs.cpu().tolist()


# =============================================================================
# 2. RANGE ENCODER & DECODER
# =============================================================================
class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.output = bytearray()

    def encode_symbol(self, symbol, freqs):
        total = sum(freqs)
        cum_freq = sum(freqs[:symbol])
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

    def decode_symbol(self, freqs):
        total = sum(freqs)
        count = (self.code - self.low) // (self.range // total)

        cum = 0
        symbol = 0
        for i, f in enumerate(freqs):
            if cum + f > count:
                symbol = i
                break
            cum += f

        sym_freq = freqs[symbol]
        cum_freq = cum

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


# =============================================================================
# 3. PIPELINE IMPLEMENTATION (ENCODE & DECODE)
# =============================================================================
def compress_file(input_path, output_bin_path, context_size=16, lr=0.002):
    with open(input_path, "rb") as f:
        data = f.read()

    orig_len = len(data)
    print(f"[*] Encoding '{input_path}' ({orig_len} bytes) using device: {device}...")
    t0 = time.time()

    torch.manual_seed(42)
    model = NeuralCompressor(context_size=context_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    encoder = RangeEncoder()

    pad_idx = 256

    for block_start in range(0, orig_len, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, orig_len)
        block_len = block_end - block_start

        # Build context batch
        ctx_list = []
        target_list = []
        for i in range(block_start, block_end):
            if i < context_size:
                ctx = [pad_idx] * (context_size - i) + list(data[0:i])
            else:
                ctx = list(data[i - context_size : i])
            ctx_list.append(ctx)
            target_list.append(data[i])

        ctx_t = torch.tensor(ctx_list, dtype=torch.long, device=device)
        target_t = torch.tensor(target_list, dtype=torch.long, device=device)

        # Vectorized predictions
        model.eval()
        freqs_batch = get_quantized_frequencies_batch(model, ctx_t)

        # Range encode symbols
        for i, target_byte in enumerate(target_list):
            encoder.encode_symbol(target_byte, freqs_batch[i])

        # Single batch gradient step
        model.train()
        optimizer.zero_grad()
        logits = model(ctx_t)
        loss = criterion(logits, target_t)
        loss.backward()
        optimizer.step()

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


def decompress_file(input_bin_path, output_path, context_size=16, lr=0.002):
    with open(input_bin_path, "rb") as f:
        raw_bin = f.read()

    orig_len = struct.unpack(">I", raw_bin[:4])[0]
    bitstream = raw_bin[4:]

    print(f"[*] Decompressing '{input_bin_path}' ({len(raw_bin)} bytes) using device: {device}...")
    t0 = time.time()

    torch.manual_seed(42)
    model = NeuralCompressor(context_size=context_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    decoder = RangeDecoder(bitstream)

    pad_idx = 256
    decoded_bytes = bytearray()

    for block_start in range(0, orig_len, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, orig_len)

        # Decode block sequentially symbol-by-symbol while evaluating model
        block_decoded = []
        ctx_list = []

        for i in range(block_start, block_end):
            if i < context_size:
                ctx = [pad_idx] * (context_size - i) + list(decoded_bytes[0:i])
            else:
                ctx = list(decoded_bytes[i - context_size : i])

            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
            model.eval()
            freqs = get_quantized_frequencies_batch(model, ctx_t)[0]

            symbol = decoder.decode_symbol(freqs)
            decoded_bytes.append(symbol)
            block_decoded.append(symbol)
            ctx_list.append(ctx)

        # Update weights at block boundaries
        ctx_t = torch.tensor(ctx_list, dtype=torch.long, device=device)
        target_t = torch.tensor(block_decoded, dtype=torch.long, device=device)

        model.train()
        optimizer.zero_grad()
        logits = model(ctx_t)
        loss = criterion(logits, target_t)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    with open(output_path, "wb") as f:
        f.write(decoded_bytes)

    print(f"[+] Decompressed output written to '{output_path}' ({len(decoded_bytes)} bytes)")
    print(f"    Elapsed: {elapsed:.2f}s")


# =============================================================================
# 4. CLI COMMAND HANDLER
# =============================================================================
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