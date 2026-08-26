#!/usr/bin/env python3
"""
fCompress.py - Bit-Exact Synchronized Online Neural Range Compressor
====================================================================

Usage:
    python3 fCompress.py compress <input_file> <output_bin>
    python3 fCompress.py decompress <input_bin> <output_file>
    python3 fCompress.py test <input_file>

This version fixes the context-construction bug in the original script and
keeps the encoder/decoder model updates synchronized.

Important causal rule:
    To predict byte N, the model receives ONLY bytes before N.
    The current byte is never placed in its own context.
"""

import sys
import os
import time
import struct
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# DETERMINISM / DEVICE
# =============================================================================

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(42)

# CPU keeps the online optimizer/model updates bit-reproducible.
device = torch.device("cpu")

SCALE = 1 << 16
BLOCK_SIZE = 2048
PAD_IDX = 256


# =============================================================================
# 1. NEURAL PREDICTOR MODEL
# =============================================================================

class NeuralCompressor(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32,
                 hidden_dim=64, context_size=16):
        super().__init__()

        self.context_size = context_size

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embed_dim,
            padding_idx=PAD_IDX,
        )

        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # Accept either:
        #   [batch, context_size]
        # or
        #   [context_size]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        embeds = self.embedding(x).reshape(x.size(0), -1)
        out = self.relu(self.fc1(embeds))
        return self.fc2(out)


def get_quantized_frequencies(model, contexts_tensor, scale=SCALE):
    """Return one frequency table per context."""

    with torch.no_grad():
        logits = model(contexts_tensor)

        logits = torch.clamp(logits, min=-50.0, max=50.0)
        probs = torch.softmax(logits, dim=-1)

        # Every symbol receives at least frequency 1.
        freqs = (probs * (scale - 256)).to(torch.int64) + 1

        return freqs.cpu().tolist()


# =============================================================================
# 2. CAUSAL CONTEXT CONSTRUCTION
# =============================================================================

def make_context(history, context_size):
    """
    Build exactly context_size input symbols from already-known history.

    Example with context_size=4:

        history = [A, B, C]
        context = [PAD, A, B, C]

    For the first byte:

        history = []
        context = [PAD, PAD, PAD, PAD]

    The predicted/current byte is NEVER included in its own context.
    """
    recent = history[-context_size:]

    if len(recent) < context_size:
        recent = [PAD_IDX] * (context_size - len(recent)) + recent

    return recent


def build_training_batch(history, context_size):
    """
    Build causal contexts/targets for a block.

    history contains the bytes through the end of the block.
    For each target, only bytes preceding that target are used as context.
    """
    contexts = []
    targets = []

    for pos, target in enumerate(history):
        contexts.append(make_context(history[:pos], context_size))
        targets.append(target)

    return contexts, targets


# =============================================================================
# 3. RANGE ENCODER / DECODER
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

        # Keep the arithmetic state in the same 32-bit domain as the decoder.
        self.low &= 0xFFFFFFFF
        self.range &= 0xFFFFFFFF

        while (
            (self.low ^ (self.low + self.range)) < 0x01000000
            or self.range < 0x00010000
        ):
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.output.append((self.low >> 24) & 0xFF)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
            else:
                self.range = ((-self.low) & 0x0000FFFF) << 8
                self.range &= 0xFFFFFFFF
                self.output.append((self.low >> 24) & 0xFF)
                self.low = (self.low << 8) & 0xFFFFFFFF

    def finish(self):
        for _ in range(4):
            self.output.append((self.low >> 24) & 0xFF)
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
            self.code = ((self.code << 8) | b) & 0xFFFFFFFF
            self.ptr += 1

    def decode_symbol(self, freqs):
        total = sum(freqs)

        unit = self.range // total
        count = (self.code - self.low) // unit

        # Defensive clamp. A valid stream should never require this.
        if count >= total:
            count = total - 1

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

        self.low &= 0xFFFFFFFF
        self.range &= 0xFFFFFFFF

        while (
            (self.low ^ (self.low + self.range)) < 0x01000000
            or self.range < 0x00010000
        ):
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF

                b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
                self.code = ((self.code << 8) | b) & 0xFFFFFFFF
                self.ptr += 1
            else:
                self.range = ((-self.low) & 0x0000FFFF) << 8
                self.range &= 0xFFFFFFFF
                self.low = (self.low << 8) & 0xFFFFFFFF

                b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
                self.code = ((self.code << 8) | b) & 0xFFFFFFFF
                self.ptr += 1

        return symbol


# =============================================================================
# 4. SYNCHRONIZED ONLINE MODEL UPDATE
# =============================================================================

def train_model_on_block(model, optimizer, criterion, history, context_size):
    """
    Apply exactly one optimizer update for the completed block.

    The encoder and decoder call this with identical byte history, so their
    model states remain synchronized.
    """
    if not history:
        return

    contexts, targets = build_training_batch(history, context_size)

    ctx_t = torch.tensor(
        contexts,
        dtype=torch.long,
        device=device,
    )

    target_t = torch.tensor(
        targets,
        dtype=torch.long,
        device=device,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(ctx_t)
    loss = criterion(logits, target_t)

    loss.backward()
    optimizer.step()


# =============================================================================
# 5. ENCODE
# =============================================================================

def compress_file(input_path, output_bin_path,
                  context_size=16, lr=0.002):

    with open(input_path, "rb") as f:
        data = f.read()

    orig_len = len(data)

    print(f"[*] Encoding '{input_path}' ({orig_len} bytes)...")
    t0 = time.time()

    torch.manual_seed(42)

    model = NeuralCompressor(
        context_size=context_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    encoder = RangeEncoder()

    # Bytes already encoded. This is the causal history.
    history = []

    model.eval()

    for block_start in range(0, orig_len, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, orig_len)
        block = data[block_start:block_end]

        # -------------------------------------------------------------
        # Predict and encode every byte using ONLY previous bytes.
        # -------------------------------------------------------------
        for target_byte in block:
            ctx = make_context(history, context_size)

            ctx_t = torch.tensor(
                [ctx],
                dtype=torch.long,
                device=device,
            )

            freqs = get_quantized_frequencies(model, ctx_t)[0]

            encoder.encode_symbol(target_byte, freqs)

            history.append(target_byte)

        # -------------------------------------------------------------
        # Train after the block has been encoded.
        # This update is mirrored by the decoder.
        # -------------------------------------------------------------
        train_model_on_block(
            model,
            optimizer,
            criterion,
            history,
            context_size,
        )

    bitstream = encoder.finish()
    elapsed = time.time() - t0

    header = struct.pack(">I", orig_len)

    with open(output_bin_path, "wb") as f:
        f.write(header + bitstream)

    final_size = len(header) + len(bitstream)
    ratio = (
        (1.0 - (final_size / orig_len)) * 100
        if orig_len > 0 else 0
    )

    speed = orig_len / 1024 / elapsed if elapsed > 0 else 0

    print(f"[+] Compressed output saved to '{output_bin_path}'")
    print(
        f"    Raw Size: {orig_len} B | "
        f"Binary Size: {final_size} B | "
        f"Reduction: {ratio:.2f}%"
    )
    print(f"    Elapsed: {elapsed:.2f}s ({speed:.2f} KB/s)")


# =============================================================================
# 6. DECODE
# =============================================================================

def decompress_file(input_bin_path, output_path,
                    context_size=16, lr=0.002):

    with open(input_bin_path, "rb") as f:
        raw_bin = f.read()

    if len(raw_bin) < 4:
        raise ValueError("Compressed file is too short to contain a header.")

    orig_len = struct.unpack(">I", raw_bin[:4])[0]
    bitstream = raw_bin[4:]

    print(
        f"[*] Decompressing '{input_bin_path}' "
        f"({len(raw_bin)} bytes)..."
    )

    t0 = time.time()

    torch.manual_seed(42)

    model = NeuralCompressor(
        context_size=context_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    decoder = RangeDecoder(bitstream)

    history = []

    model.eval()

    while len(history) < orig_len:
        block_end = min(
            len(history) + BLOCK_SIZE,
            orig_len,
        )

        # -------------------------------------------------------------
        # Decode the next block causally.
        # -------------------------------------------------------------
        while len(history) < block_end:
            ctx = make_context(history, context_size)

            ctx_t = torch.tensor(
                [ctx],
                dtype=torch.long,
                device=device,
            )

            freqs = get_quantized_frequencies(model, ctx_t)[0]

            symbol = decoder.decode_symbol(freqs)

            history.append(symbol)

        # -------------------------------------------------------------
        # Mirror the encoder's model update exactly.
        # -------------------------------------------------------------
        train_model_on_block(
            model,
            optimizer,
            criterion,
            history,
            context_size,
        )

    decoded_bytes = bytes(history)

    with open(output_path, "wb") as f:
        f.write(decoded_bytes)

    elapsed = time.time() - t0

    print(
        f"[+] Decompressed output written to "
        f"'{output_path}' ({len(decoded_bytes)} bytes)"
    )
    print(f"    Elapsed: {elapsed:.2f}s")


# =============================================================================
# 7. TEST HARNESS
# =============================================================================

def test_file(input_file):

    bin_file = input_file + ".tmp.bin"
    recon_file = input_file + ".tmp.recon"

    try:
        compress_file(input_file, bin_file)

        print("-" * 65)

        decompress_file(bin_file, recon_file)

        with open(input_file, "rb") as f:
            original = f.read()

        with open(recon_file, "rb") as f:
            reconstructed = f.read()

        orig_hash = hashlib.sha256(original).hexdigest()
        recon_hash = hashlib.sha256(reconstructed).hexdigest()

        print("-" * 65)
        print(f"Original SHA-256     : {orig_hash}")
        print(f"Reconstructed SHA-256: {recon_hash}")

        if original != reconstructed:
            print("RESULT: RECONSTRUCTION MISMATCH")
            raise AssertionError(
                "CRITICAL ERROR: Reconstruction mismatch!"
            )

        print("RESULT: EXACT BIT-EXACT MATCH (PASS)")

    finally:
        if os.path.exists(bin_file):
            os.remove(bin_file)

        if os.path.exists(recon_file):
            os.remove(recon_file)


# =============================================================================
# 8. CLI
# =============================================================================

def main():

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "compress":
        input_file = sys.argv[2]
        output_bin = (
            sys.argv[3]
            if len(sys.argv) > 3
            else input_file + ".bin"
        )

        compress_file(input_file, output_bin)

    elif cmd == "decompress":
        input_bin = sys.argv[2]
        output_file = (
            sys.argv[3]
            if len(sys.argv) > 3
            else "reconstructed_file"
        )

        decompress_file(input_bin, output_file)

    elif cmd == "test":
        input_file = sys.argv[2]
        test_file(input_file)

    else:
        print(f"Unknown command '{cmd}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
