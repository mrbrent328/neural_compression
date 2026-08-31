#!/usr/bin/env python3
"""NCA-02: Strong Neural Archive experiment.

NCA-02 keeps the NCA-01 archive machinery and adaptive selector unchanged,
but replaces the small neural predictor with the stronger Series-4-inspired
configuration identified in prior experiments:

  embedding = 48
  hidden    = 300
  context   = 10 bytes
  states    = 3
  optimizer = SGD(lr=0.04)

Unlike NCA-01's averaged predictor states, NCA-02 selects one of the three
states deterministically by minimum predictive entropy. The choice depends
only on shared model state and decoded context, so encoder and decoder make
the same decision without transmitting selector bits. Only the selected state
is updated after each symbol.

All other archive behavior is delegated to NCA-01: 64 KiB blocks, structural
fingerprinting/selection, range coding, manifest, checksums, exact testing,
and no cross-file deduplication.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "NCA-01_local_adaptive_archive.py"

if not BASE_PATH.exists():
    raise FileNotFoundError(
        "NCA-02 requires NCA-01_local_adaptive_archive.py in the same directory."
    )

spec = importlib.util.spec_from_file_location("nca01_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load NCA-01 base module")
nca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nca)

EXPERIMENT = "NCA-02"
MAGIC = b"NCA2"
VERSION = 2
EMBED_DIM = 48
HIDDEN_DIM = 300
CONTEXT_SIZE = 10
NUM_STATES = 3
LEARNING_RATE = 0.04
SEED = 42


class StrongNeuralCompressor(nn.Module):
    """Three independent Series-4-inspired predictors."""

    def __init__(self, vocab_size=256, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, context_size=CONTEXT_SIZE,
                 num_states=NUM_STATES):
        super().__init__()
        self.context_size = context_size
        self.num_states = num_states
        self.states = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(vocab_size + 1, embed_dim, padding_idx=nca.PAD),
                nn.Flatten(start_dim=1),
                nn.Linear(context_size * embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size),
            ) for _ in range(num_states)
        ])


def make_model():
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = StrongNeuralCompressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def select_state(logits_by_state):
    """Choose the most confident state using predictive entropy.

    This is deterministic from the shared model/context and therefore requires
    no selector bits in the archive.
    """
    entropies = []
    for logits in logits_by_state:
        probs = torch.softmax(torch.clamp(logits.detach(), -50, 50), dim=-1)
        entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum().item()
        entropies.append(entropy)
    return min(range(len(entropies)), key=lambda i: (entropies[i], i))


def predict_selected(model, x):
    logits_by_state = [state(x).squeeze(0) for state in model.states]
    chosen = select_state(logits_by_state)
    return logits_by_state[chosen], chosen


def train_selected(model, optimizer, criterion, logits, state_index, target):
    optimizer.zero_grad()
    criterion(logits.unsqueeze(0), target).backward()
    # Only the selected predictor state participates in this update. Gradients
    # for the other states are absent because their logits were not used.
    optimizer.step()


def neural_encode(data: bytes) -> bytes:
    model, optimizer, criterion = make_model()
    model.train()
    enc = nca.RangeEncoder()
    ctx = [nca.PAD] * CONTEXT_SIZE
    for byte in data:
        x = torch.tensor([ctx], dtype=torch.long)
        logits, state_index = predict_selected(model, x)
        freqs = nca.logits_to_freqs(logits.detach())
        cum = list(nca.accumulate(freqs, initial=0))
        enc.encode_symbol(byte, freqs, cum)
        target = torch.tensor([byte], dtype=torch.long)
        train_selected(model, optimizer, criterion, logits, state_index, target)
        ctx = ctx[1:] + [byte]
    return enc.finish()


def neural_decode(stream: bytes, size: int) -> bytes:
    model, optimizer, criterion = make_model()
    model.train()
    dec = nca.RangeDecoder(stream)
    ctx = [nca.PAD] * CONTEXT_SIZE
    out = bytearray()
    for _ in range(size):
        x = torch.tensor([ctx], dtype=torch.long)
        logits, state_index = predict_selected(model, x)
        freqs = nca.logits_to_freqs(logits.detach())
        cum = list(nca.accumulate(freqs, initial=0))
        symbol = dec.decode_symbol(freqs, cum)
        out.append(symbol)
        target = torch.tensor([symbol], dtype=torch.long)
        train_selected(model, optimizer, criterion, logits, state_index, target)
        ctx = ctx[1:] + [symbol]
    return bytes(out)


def encode_candidate(data, strategy):
    if strategy == "neural":
        return neural_encode(data)
    return nca.encode_candidate(data, strategy)


def decode_candidate(payload, size, strategy):
    if strategy == "neural":
        return neural_decode(payload, size)
    return nca.decode_candidate(payload, size, strategy)


def pack(input_path, archive_path, strategy, report_path=None):
    # Patch only the candidate codec used by NCA-01's pack implementation.
    original_encode = nca.encode_candidate
    original_magic = nca.MAGIC
    original_version = nca.VERSION
    try:
        nca.encode_candidate = encode_candidate
        nca.MAGIC = MAGIC
        nca.VERSION = VERSION
        result = nca.pack(input_path, archive_path, strategy, None)
    finally:
        nca.encode_candidate = original_encode
        nca.MAGIC = original_magic
        nca.VERSION = original_version

    result["experiment"] = EXPERIMENT
    result["neural_config"] = {
        "embedding": EMBED_DIM,
        "hidden": HIDDEN_DIM,
        "context": CONTEXT_SIZE,
        "states": NUM_STATES,
        "optimizer": "SGD",
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "state_selection": "minimum predictive entropy",
    }
    if report_path:
        Path(report_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def read_archive(path):
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError("Not an NCA2 archive")
    n = int.from_bytes(raw[4:8], "big")
    manifest = json.loads(raw[8:8 + n])
    ptr = 8 + n
    payloads = []
    for _ in manifest["blocks"]:
        size = int.from_bytes(raw[ptr:ptr + 4], "big")
        ptr += 4
        payloads.append(raw[ptr:ptr + size])
        ptr += size
    return manifest, payloads


def list_archive(path):
    manifest, _ = read_archive(path)
    for f in manifest["files"]:
        print(f'{f["size"]:>12}  {f["path"]}')


def test_archive(path):
    manifest, payloads = read_archive(path)
    checked = 0
    for f in manifest["files"]:
        data = bytearray()
        for i in f["blocks"]:
            b = manifest["blocks"][i]
            raw = decode_candidate(payloads[i], b["size"], b["strategy"])
            if len(raw) != b["size"] or nca.sha256(raw) != b["sha256"]:
                raise ValueError(f'Block failed: {f["path"]}')
            data.extend(raw)
            checked += 1
        if nca.sha256(data) != f["sha256"]:
            raise ValueError(f'File failed: {f["path"]}')
    print(f"OK: {len(manifest['files'])} files, {checked} blocks, exact reconstruction verified")


def extract(archive_path, output_dir):
    manifest, payloads = read_archive(archive_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for f in manifest["files"]:
        out = bytearray()
        for i in f["blocks"]:
            b = manifest["blocks"][i]
            raw = decode_candidate(payloads[i], b["size"], b["strategy"])
            if nca.sha256(raw) != b["sha256"]:
                raise ValueError(f'Block checksum failed: {f["path"]}')
            out.extend(raw)
        if nca.sha256(out) != f["sha256"]:
            raise ValueError(f'File checksum failed: {f["path"]}')
        target = root / f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out)


def main():
    ap = __import__("argparse").ArgumentParser(
        prog="nca-02", description="NCA-02 Strong Neural Archive"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("input")
    p.add_argument("archive")
    p.add_argument("--strategy", choices=["store", "classical", "neural", "adaptive", "oracle"], default="adaptive")
    p.add_argument("--report")
    p = sub.add_parser("list")
    p.add_argument("archive")
    p = sub.add_parser("extract")
    p.add_argument("archive")
    p.add_argument("output")
    p = sub.add_parser("test")
    p.add_argument("archive")
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.input, args.archive, args.strategy, args.report)
    elif args.cmd == "list":
        list_archive(args.archive)
    elif args.cmd == "extract":
        extract(args.archive, args.output)
    elif args.cmd == "test":
        test_archive(args.archive)


if __name__ == "__main__":
    main()
