#!/usr/bin/env python3
"""NCA-01: Local Adaptive Archive experiment.

A small real archive format used to test whether local structural strategy
selection can beat a uniform compressor on mixed collections.

Strategies:
  store     raw bytes
  classical zlib/DEFLATE
  neural    synchronized online neural range coder
  adaptive  V2/V3-inspired structural selector
  oracle    encode all candidates and choose the smallest (evaluation only)

The archive is intentionally simple: fixed 64 KiB blocks, explicit manifest,
per-block metadata/checksums, exact extraction, and append-style add/remove.
No cross-file deduplication is performed in NCA-01.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

MAGIC = b"NCA1"
VERSION = 1
BLOCK_SIZE = 64 * 1024
SCALE = 1 << 16
PAD = 256


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def structural_fingerprint(data: bytes) -> dict:
    n = len(data)
    if not n:
        return {"entropy": 0.0, "unique_fraction": 0.0, "repetition": 1.0,
                "change_rate": 0.0, "run_score": 1.0, "pair_structure": 1.0,
                "distribution_concentration": 1.0, "local_entropy_variance": 0.0}
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    entropy = -sum((c / n) * math.log2(c / n) for c in counts if c)
    unique_fraction = sum(c > 0 for c in counts) / 256.0
    repetition = 1.0 - unique_fraction
    changes = sum(a != b for a, b in zip(data, data[1:])) / max(1, n - 1)
    runs = 1 + sum(a != b for a, b in zip(data, data[1:]))
    run_score = 1.0 - runs / n
    pairs = set(zip(data, data[1:])) if n > 1 else set()
    pair_structure = 1.0 - len(pairs) / 65536.0
    concentration = max(counts) / n
    chunks = [data[i:i + 1024] for i in range(0, n, 1024)]
    ents = []
    for cdata in chunks:
        cc = [0] * 256
        for b in cdata:
            cc[b] += 1
        cn = len(cdata)
        ents.append(-sum((c / cn) * math.log2(c / cn) for c in cc if c))
    mean_e = sum(ents) / len(ents)
    variance = sum((e - mean_e) ** 2 for e in ents) / len(ents)
    return {"entropy": entropy, "unique_fraction": unique_fraction,
            "repetition": repetition, "change_rate": changes,
            "run_score": run_score, "pair_structure": pair_structure,
            "distribution_concentration": concentration,
            "local_entropy_variance": variance}


def choose_adaptive(fp: dict) -> str:
    """Deterministic V2/V3-inspired structural selector.

    This is deliberately simple and explainable. NCA-01 tests whether local
    structure can predict a useful representation at all; learned selection is
    deferred to a later experiment.
    """
    if fp["entropy"] >= 7.98 and fp["unique_fraction"] > 0.95:
        return "store"
    if fp["run_score"] > 0.35 or fp["repetition"] > 0.80:
        return "classical"
    if fp["entropy"] < 6.5 or fp["distribution_concentration"] > 0.20:
        return "neural"
    if fp["local_entropy_variance"] > 0.8 or fp["pair_structure"] > 0.97:
        return "neural"
    return "classical"


class NeuralCompressor(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, hidden_dim=64,
                 context_size=16, num_states=4):
        super().__init__()
        self.context_size = context_size
        self.states = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(vocab_size + 1, embed_dim, padding_idx=PAD),
                nn.Flatten(start_dim=1),
                nn.Linear(context_size * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size),
            ) for _ in range(num_states)
        ])

    def forward(self, x):
        return torch.stack([s(x) for s in self.states], dim=0).mean(dim=0)


def make_model(context_size=16, lr=0.04, num_states=4):
    if torch is None:
        raise RuntimeError("Neural mode requires PyTorch. Install torch or use --strategy classical/store.")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = NeuralCompressor(context_size=context_size, num_states=num_states)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, opt, criterion


def logits_to_freqs(logits):
    probs = torch.softmax(torch.clamp(logits, -50, 50), dim=-1)
    return ((probs * (SCALE - 256)).int() + 1).tolist()


class RangeEncoder:
    def __init__(self):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.output = bytearray()

    def encode_symbol(self, symbol, freqs, cum):
        total = cum[-1]
        self.range //= total
        self.low += cum[symbol] * self.range
        self.range *= freqs[symbol]
        while (self.low ^ (self.low + self.range)) < 0x01000000 or self.range < 0x10000:
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
            else:
                self.range = (-self.low & 0xFFFF) << 8
                self.output.append(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF

    def finish(self):
        for _ in range(4):
            self.output.append(self.low >> 24)
            self.low = (self.low << 8) & 0xFFFFFFFF
        return bytes(self.output)


class RangeDecoder:
    def __init__(self, stream):
        self.stream, self.ptr = stream, 0
        self.low, self.range, self.code = 0, 0xFFFFFFFF, 0
        for _ in range(4):
            b = stream[self.ptr] if self.ptr < len(stream) else 0
            self.code = (self.code << 8) | b
            self.ptr += 1

    def decode_symbol(self, freqs, cum):
        total = cum[-1]
        count = (self.code - self.low) // (self.range // total)
        symbol = bisect_right(cum, count) - 1
        self.range //= total
        self.low += cum[symbol] * self.range
        self.range *= freqs[symbol]
        while (self.low ^ (self.low + self.range)) < 0x01000000 or self.range < 0x10000:
            if (self.low ^ (self.low + self.range)) < 0x01000000:
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
            else:
                self.range = (-self.low & 0xFFFF) << 8
                self.low = (self.low << 8) & 0xFFFFFFFF
            b = self.stream[self.ptr] if self.ptr < len(self.stream) else 0
            self.code = ((self.code << 8) | b) & 0xFFFFFFFF
            self.ptr += 1
        return symbol


def neural_encode(data: bytes, context_size=16, lr=0.04, num_states=4) -> bytes:
    model, opt, criterion = make_model(context_size, lr, num_states)
    model.train(); enc = RangeEncoder(); ctx = [PAD] * context_size
    for byte in data:
        x = torch.tensor([ctx], dtype=torch.long)
        logits = model(x).squeeze(0)
        freqs = logits_to_freqs(logits.detach()); cum = list(accumulate(freqs, initial=0))
        enc.encode_symbol(byte, freqs, cum)
        target = torch.tensor([byte], dtype=torch.long)
        opt.zero_grad(); criterion(logits.unsqueeze(0), target).backward(); opt.step()
        ctx = ctx[1:] + [byte]
    return enc.finish()


def neural_decode(stream: bytes, size: int, context_size=16, lr=0.04, num_states=4) -> bytes:
    model, opt, criterion = make_model(context_size, lr, num_states)
    model.train(); dec = RangeDecoder(stream); ctx = [PAD] * context_size; out = bytearray()
    for _ in range(size):
        x = torch.tensor([ctx], dtype=torch.long); logits = model(x).squeeze(0)
        freqs = logits_to_freqs(logits.detach()); cum = list(accumulate(freqs, initial=0))
        symbol = dec.decode_symbol(freqs, cum); out.append(symbol)
        target = torch.tensor([symbol], dtype=torch.long)
        opt.zero_grad(); criterion(logits.unsqueeze(0), target).backward(); opt.step()
        ctx = ctx[1:] + [symbol]
    return bytes(out)


def encode_candidate(data, strategy):
    if strategy == "store":
        return data
    if strategy == "classical":
        return zlib.compress(data, level=9)
    if strategy == "neural":
        return neural_encode(data)
    raise ValueError(strategy)


def decode_candidate(payload, size, strategy):
    if strategy == "store": return payload
    if strategy == "classical": return zlib.decompress(payload)
    if strategy == "neural": return neural_decode(payload, size)
    raise ValueError(strategy)


def iter_files(root: Path):
    if root.is_file():
        yield root, root.name
        return
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p, p.relative_to(root).as_posix()


def pack(input_path, archive_path, strategy, report_path=None):
    root = Path(input_path).resolve(); files = []
    blocks = []
    total_original = total_payload = 0
    t0 = time.perf_counter()
    counts = {"store": 0, "classical": 0, "neural": 0}
    for path, rel in iter_files(root):
        data = path.read_bytes(); file_blocks = []
        for offset in range(0, len(data), BLOCK_SIZE):
            raw = data[offset:offset + BLOCK_SIZE]
            fp = structural_fingerprint(raw)
            if strategy == "oracle":
                candidates = {s: encode_candidate(raw, s) for s in ("store", "classical", "neural")}
                chosen = min(candidates, key=lambda s: len(candidates[s]))
                payload = candidates[chosen]
            elif strategy == "adaptive":
                chosen = choose_adaptive(fp); payload = encode_candidate(raw, chosen)
                # Never let adaptive expansion increase the archive.
                if len(payload) >= len(raw):
                    chosen, payload = "store", raw
            else:
                chosen = strategy; payload = encode_candidate(raw, chosen)
                if len(payload) >= len(raw) and chosen != "store":
                    chosen, payload = "store", raw
            idx = len(blocks)
            blocks.append({"offset": offset, "size": len(raw), "payload": payload,
                           "strategy": chosen, "sha256": sha256(raw), "fingerprint": fp})
            file_blocks.append(idx); counts[chosen] += 1
            total_original += len(raw); total_payload += len(payload)
        files.append({"path": rel, "size": len(data), "sha256": sha256(data), "blocks": file_blocks})
    manifest = {"version": VERSION, "block_size": BLOCK_SIZE, "files": files,
                "blocks": [{k: v for k, v in b.items() if k != "payload"} for b in blocks]}
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    with open(archive_path, "wb") as f:
        f.write(MAGIC + struct.pack(">I", len(manifest_bytes)) + manifest_bytes)
        for b in blocks: f.write(struct.pack(">I", len(b["payload"])) + b["payload"])
    archive_size = os.path.getsize(archive_path); elapsed = time.perf_counter() - t0
    result = {"experiment": "NCA-01", "strategy": strategy, "input": str(root),
              "original_bytes": total_original, "payload_bytes": total_payload,
              "archive_bytes": archive_size,
              "reduction_percent": (1 - archive_size / total_original) * 100 if total_original else 0,
              "block_counts": counts, "elapsed_seconds": elapsed,
              "sha256_manifest": sha256(manifest_bytes)}
    if report_path: Path(report_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); return result


def read_archive(path):
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC: raise ValueError("Not an NCA1 archive")
    n = struct.unpack(">I", raw[4:8])[0]; manifest = json.loads(raw[8:8+n]); ptr = 8+n
    payloads = []
    for _ in manifest["blocks"]:
        size = struct.unpack(">I", raw[ptr:ptr+4])[0]; ptr += 4
        payloads.append(raw[ptr:ptr+size]); ptr += size
    return manifest, payloads


def list_archive(path):
    manifest, _ = read_archive(path)
    for f in manifest["files"]: print(f'{f["size"]:>12}  {f["path"]}')


def extract(archive_path, output_dir):
    manifest, payloads = read_archive(archive_path); root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    for f in manifest["files"]:
        out = bytearray()
        for i in f["blocks"]:
            b = manifest["blocks"][i]; raw = decode_candidate(payloads[i], b["size"], b["strategy"])
            if sha256(raw) != b["sha256"]: raise ValueError(f'Block checksum failed: {f["path"]}')
            out.extend(raw)
        if sha256(out) != f["sha256"]: raise ValueError(f'File checksum failed: {f["path"]}')
        target = root / f["path"]; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(out)


def test_archive(path):
    manifest, payloads = read_archive(path); checked = 0
    for f in manifest["files"]:
        data = bytearray()
        for i in f["blocks"]:
            b = manifest["blocks"][i]; raw = decode_candidate(payloads[i], b["size"], b["strategy"])
            if len(raw) != b["size"] or sha256(raw) != b["sha256"]: raise ValueError(f'Block failed: {f["path"]}')
            data.extend(raw); checked += 1
        if sha256(data) != f["sha256"]: raise ValueError(f'File failed: {f["path"]}')
    print(f"OK: {len(manifest['files'])} files, {checked} blocks, exact reconstruction verified")


def main():
    ap = argparse.ArgumentParser(prog="nca", description="NCA-01 Local Adaptive Archive")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack"); p.add_argument("input"); p.add_argument("archive"); p.add_argument("--strategy", choices=["store", "classical", "neural", "adaptive", "oracle"], default="adaptive"); p.add_argument("--report")
    p = sub.add_parser("list"); p.add_argument("archive")
    p = sub.add_parser("extract"); p.add_argument("archive"); p.add_argument("output")
    p = sub.add_parser("test"); p.add_argument("archive")
    args = ap.parse_args()
    if args.cmd == "pack": pack(args.input, args.archive, args.strategy, args.report)
    elif args.cmd == "list": list_archive(args.archive)
    elif args.cmd == "extract": extract(args.archive, args.output)
    elif args.cmd == "test": test_archive(args.archive)

if __name__ == "__main__": main()
