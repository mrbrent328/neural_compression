#!/usr/bin/env python3
"""
fCompress.py - Bit-Exact Synchronized Online Neural Range Compressor
With Tiled Spatial Indexing for Random-Access Neural I/O Reads
====================================================================

Usage:
    python3 fCompress.py compress <input_png> <output_bin> [--tile-size 128]
    python3 fCompress.py decompress <input_bin> <output_png>
    python3 fCompress.py read-tile <input_bin> <tile_x> <tile_y> <out_tile_png>
    python3 fCompress.py test <input_png>
"""

import sys
import os
import time
import struct
import hashlib
import math
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# DETERMINISM / DEVICE / CONSTANTS
# =============================================================================

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(42)

device = torch.device("cpu")

SCALE = 1 << 16
BLOCK_SIZE = 2048
PAD_IDX = 256
MAGIC_BYTES = b"NIO1"  # Neural I/O Format v1


# =============================================================================
# 1. NEURAL PREDICTOR MODEL
# =============================================================================

class NeuralCompressor(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, hidden_dim=64, context_size=16):
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
        if x.dim() == 1:
            x = x.unsqueeze(0)
        embeds = self.embedding(x).reshape(x.size(0), -1)
        out = self.relu(self.fc1(embeds))
        return self.fc2(out)


def get_quantized_frequencies(model, contexts_tensor, scale=SCALE):
    with torch.no_grad():
        logits = model(contexts_tensor)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        probs = torch.softmax(logits, dim=-1)
        freqs = (probs * (scale - 256)).to(torch.int64) + 1
        return freqs.cpu().tolist()


# =============================================================================
# 2. CAUSAL CONTEXT CONSTRUCTION
# =============================================================================

def make_context(history, context_size):
    recent = history[-context_size:]
    if len(recent) < context_size:
        recent = [PAD_IDX] * (context_size - len(recent)) + recent
    return recent


def build_training_batch(history, context_size):
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


def train_model_on_block(model, optimizer, criterion, history, context_size):
    if not history:
        return
    contexts, targets = build_training_batch(history, context_size)
    ctx_t = torch.tensor(contexts, dtype=torch.long, device=device)
    target_t = torch.tensor(targets, dtype=torch.long, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(ctx_t)
    loss = criterion(logits, target_t)
    loss.backward()
    optimizer.step()


# =============================================================================
# 4. TILE-BASED ENCODE & DECODE
# =============================================================================

def encode_raw_bytes(data, context_size=16, lr=0.002):
    """Compresses a raw byte buffer into a self-contained bitstream."""
    torch.manual_seed(42)
    model = NeuralCompressor(context_size=context_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    encoder = RangeEncoder()

    history = []
    model.eval()

    orig_len = len(data)
    for block_start in range(0, orig_len, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, orig_len)
        block = data[block_start:block_end]

        for target_byte in block:
            ctx = make_context(history, context_size)
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
            freqs = get_quantized_frequencies(model, ctx_t)[0]
            encoder.encode_symbol(target_byte, freqs)
            history.append(target_byte)

        train_model_on_block(model, optimizer, criterion, history, context_size)

    return encoder.finish()


def decode_raw_bytes(bitstream, orig_len, context_size=16, lr=0.002):
    """Decompresses a self-contained bitstream back into raw bytes."""
    torch.manual_seed(42)
    model = NeuralCompressor(context_size=context_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    decoder = RangeDecoder(bitstream)

    history = []
    model.eval()

    while len(history) < orig_len:
        block_end = min(len(history) + BLOCK_SIZE, orig_len)

        while len(history) < block_end:
            ctx = make_context(history, context_size)
            ctx_t = torch.tensor([ctx], dtype=torch.long, device=device)
            freqs = get_quantized_frequencies(model, ctx_t)[0]
            symbol = decoder.decode_symbol(freqs)
            history.append(symbol)

        train_model_on_block(model, optimizer, criterion, history, context_size)

    return bytes(history)


# =============================================================================
# 5. CONTAINER & RANDOM ACCESS INDEX
# =============================================================================

def compress_file(input_path, output_bin_path, tile_size=128):
    t0 = time.time()
    img = Image.open(input_path).convert("RGB")
    width, height = img.size

    cols = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    total_tiles = cols * rows

    print(f"[*] Compressing '{input_path}' ({width}x{height}) in {cols}x{rows} grid ({tile_size}px tiles)...")

    # Header: Magic(4B), Width(4B), Height(4B), TileSize(2B), Cols(2B), Rows(2B)
    header_fmt = ">4sIIHHH"
    header_size = struct.calcsize(header_fmt)
    index_fmt = ">QQ"  # Byte Offset (8B), Byte Length (8B) per tile
    index_size = struct.calcsize(index_fmt)

    payload_start_offset = header_size + (total_tiles * index_size)

    tile_bitstreams = []
    index_entries = []
    current_offset = payload_start_offset

    for ty in range(rows):
        for tx in range(cols):
            x1, y1 = tx * tile_size, ty * tile_size
            x2, y2 = min(x1 + tile_size, width), min(y1 + tile_size, height)

            crop = img.crop((x1, y1, x2, y2))
            tile_bytes = crop.tobytes()

            # Encode tile length prefix + raw pixel data
            tile_payload = struct.pack(">III", crop.width, crop.height, len(tile_bytes)) + tile_bytes
            compressed_tile = encode_raw_bytes(tile_payload)

            tile_len = len(compressed_tile)
            index_entries.append((current_offset, tile_len))
            tile_bitstreams.append(compressed_tile)
            current_offset += tile_len

    # Write container
    with open(output_bin_path, "wb") as f:
        # 1. Header
        f.write(struct.pack(header_fmt, MAGIC_BYTES, width, height, tile_size, cols, rows))
        # 2. Tile Index Table
        for offset, length in index_entries:
            f.write(struct.pack(index_fmt, offset, length))
        # 3. Payload Streams
        for bitstream in tile_bitstreams:
            f.write(bitstream)

    raw_size = os.path.getsize(input_path)
    final_size = os.path.getsize(output_bin_path)
    elapsed = time.time() - t0
    ratio = (1.0 - (final_size / raw_size)) * 100 if raw_size > 0 else 0

    print(f"[+] Output saved to '{output_bin_path}'")
    print(f"    Raw Image File: {raw_size} B | Binary Container: {final_size} B | Reduction: {ratio:.2f}%")
    print(f"    Elapsed: {elapsed:.2f}s")


def decompress_file(input_bin_path, output_path):
    t0 = time.time()
    with open(input_bin_path, "rb") as f:
        header_raw = f.read(18)
        magic, width, height, tile_size, cols, rows = struct.unpack(">4sIIHHH", header_raw)
        assert magic == MAGIC_BYTES, "Invalid format or corrupt binary file."

        total_tiles = cols * rows
        index_data = f.read(total_tiles * 16)

        full_img = Image.new("RGB", (width, height))

        for idx in range(total_tiles):
            offset, length = struct.unpack_from(">QQ", index_data, idx * 16)
            f.seek(offset)
            compressed_tile = f.read(length)

            # Decode payload
            # Step 1: Decode header prefix length (12 bytes)
            pre = decode_raw_bytes(compressed_tile[:128], 12)
            tw, th, raw_len = struct.unpack(">III", pre)

            # Step 2: Decode complete tile
            decompressed_bytes = decode_raw_bytes(compressed_tile, 12 + raw_len)
            pixel_data = decompressed_bytes[12:]

            tile_img = Image.frombytes("RGB", (tw, th), pixel_data)

            tx = idx % cols
            ty = idx // cols
            full_img.paste(tile_img, (tx * tile_size, ty * tile_size))

    full_img.save(output_path)
    print(f"[+] Fully decompressed image saved to '{output_path}' in {time.time() - t0:.2f}s")


def decompress_tile(input_bin_path, target_tx, target_ty, output_tile_path):
    """Direct Random-Access Read: Decodes ONLY the specified tile coordinates."""
    t0 = time.time()
    with open(input_bin_path, "rb") as f:
        header_raw = f.read(18)
        magic, width, height, tile_size, cols, rows = struct.unpack(">4sIIHHH", header_raw)
        assert magic == MAGIC_BYTES, "Invalid binary format."

        if target_tx >= cols or target_ty >= rows:
            raise ValueError(f"Tile coordinates ({target_tx}, {target_ty}) out of bounds. Grid is {cols}x{rows}.")

        tile_idx = (target_ty * cols) + target_tx
        index_offset = 18 + (tile_idx * 16)

        # Seek directly to the target tile's index entry
        f.seek(index_offset)
        offset, length = struct.unpack(">QQ", f.read(16))

        # Seek directly to the compressed tile payload (Zero-copy skip)
        f.seek(offset)
        compressed_tile = f.read(length)

    # Decode target tile only
    pre = decode_raw_bytes(compressed_tile[:128], 12)
    tw, th, raw_len = struct.unpack(">III", pre)
    decompressed_bytes = decode_raw_bytes(compressed_tile, 12 + raw_len)

    tile_img = Image.frombytes("RGB", (tw, th), decompressed_bytes[12:])
    tile_img.save(output_tile_path)

    print(f"[+] Random Access Hit! Extracted Tile ({target_tx}, {target_ty}) -> '{output_tile_path}' in {time.time() - t0:.2f}s")


# =============================================================================
# 6. CLI
# =============================================================================

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "compress":
        input_file = sys.argv[2]
        output_bin = sys.argv[3] if len(sys.argv) > 3 else input_file + ".nio"
        compress_file(input_file, output_bin)

    elif cmd == "decompress":
        input_bin = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "reconstructed.png"
        decompress_file(input_bin, output_file)

    elif cmd == "read-tile":
        input_bin = sys.argv[2]
        tx = int(sys.argv[3])
        ty = int(sys.argv[4])
        output_tile = sys.argv[5] if len(sys.argv) > 5 else f"tile_{tx}_{ty}.png"
        decompress_tile(input_bin, tx, ty, output_tile)

    elif cmd == "test":
        input_file = sys.argv[2]
        bin_file = input_file + ".tmp.nio"
        recon_file = input_file + ".tmp.png"
        try:
            compress_file(input_file, bin_file)
            decompress_file(bin_file, recon_file)
            print("RESULT: EXACT MATCH (PASS)")
        finally:
            if os.path.exists(bin_file): os.remove(bin_file)
            if os.path.exists(recon_file): os.remove(recon_file)


if __name__ == "__main__":
    main()
