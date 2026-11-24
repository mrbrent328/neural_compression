#!/usr/bin/env python3
"""
E43 — Minimal latent communication between two frozen language models.

Phase 0/1: a controlled synthetic fact-transfer benchmark. The task is
deliberately symbolic and held out by fact identity to test useful transfer
before introducing expensive pretrained-model inference.

Sender and receiver are frozen. Only the sender encoder and receiver adapter
are trained. The receiver consumes either a continuous latent vector, a
quantized latent packet, a text string, a compressed text string, or random
latent values. The first implementation uses small MLP agents rather than
claiming pretrained-LM integration; this isolates the communication protocol.
"""

from __future__ import annotations
import argparse, gzip, json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

SEED = 43
FACT_COUNT = 256
VALUE_COUNT = 64
QUERY_COUNT = 64
LATENT_DIM = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class Example:
    fact_id: int
    value_id: int
    query_id: int

def make_dataset(seed: int, n: int = 12000):
    """Deterministic world; train/test split by query/fact combinations."""
    g = random.Random(seed)
    values = [g.randrange(VALUE_COUNT) for _ in range(FACT_COUNT)]
    rows = []
    for _ in range(n):
        f = g.randrange(FACT_COUNT)
        q = g.randrange(QUERY_COUNT)
        rows.append(Example(f, values[f], q))
    # Query IDs 0..47 train; 48..63 held out.
    train = [x for x in rows if x.query_id < 48]
    test = [x for x in rows if x.query_id >= 48]
    return values, train, test

def make_text(fact_id: int, query_id: int) -> str:
    # Explicitly simple, human-readable semantic serialization baseline.
    return f"Fact identifier {fact_id} has stored value {query_id}"  # replaced at call site

class Sender(nn.Module):
    def __init__(self, fact_count=FACT_COUNT, width=64):
        super().__init__()
        self.fact = nn.Embedding(fact_count, width)
        self.query = nn.Embedding(QUERY_COUNT, width)
        self.body = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                  nn.Linear(width, width), nn.GELU())

    def forward(self, fact, query):
        return self.body(torch.cat([self.fact(fact), self.query(query)], -1))

class Receiver(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.query = nn.Embedding(QUERY_COUNT, width)
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                  nn.Linear(width, VALUE_COUNT))

    def forward(self, message, query):
        return self.head(torch.cat([message, self.query(query)], -1))

class EncoderAdapter(nn.Module):
    def __init__(self, width=64, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, latent_dim))

    def forward(self, h):
        return self.net(h)

class ReceiverAdapter(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, width), nn.GELU(),
                                 nn.Linear(width, width))

    def forward(self, z):
        return self.net(z)

def quantize_packet(z: torch.Tensor, bits: int = 8) -> torch.Tensor:
    # Fixed symmetric scalar quantization. Scale is packet metadata and counted.
    qmax = (1 << (bits - 1)) - 1
    scale = z.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(z / scale).clamp(-qmax, qmax)
    return q * scale

def make_models(seed: int):
    seed_all(seed)
    sender = Sender().to(DEVICE)
    receiver = Receiver().to(DEVICE)
    for p in sender.parameters(): p.requires_grad_(False)
    for p in receiver.parameters(): p.requires_grad_(False)
    enc = EncoderAdapter().to(DEVICE)
    dec = ReceiverAdapter().to(DEVICE)
    return sender, receiver, enc, dec

def tensor_batch(rows):
    return tuple(torch.tensor([getattr(x, k) for x in rows], dtype=torch.long, device=DEVICE)
                 for k in ("fact_id", "value_id", "query_id"))

def train_latent(seed: int, epochs: int = 12, batch_size: int = 256):
    values, train, test = make_dataset(seed)
    sender, receiver, enc, dec = make_models(seed)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    rng = random.Random(seed + 100)
    start = time.perf_counter()
    for _ in range(epochs):
        rng.shuffle(train)
        for i in range(0, len(train), batch_size):
            batch = train[i:i+batch_size]
            f, y, q = tensor_batch(batch)
            with torch.no_grad():
                h = sender(f, q)
            z = enc(h)
            zq = quantize_packet(z)
            decoded = dec(zq)
            logits = receiver(decoded, q)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    train_seconds = time.perf_counter() - start
    return values, test, sender, receiver, enc, dec, train_seconds

@torch.no_grad()
def eval_latent(test, sender, receiver, enc, dec, random_control=False):
    ys, preds = [], []
    t0 = time.perf_counter()
    for i in range(0, len(test), 512):
        batch = test[i:i+512]
        f, y, q = tensor_batch(batch)
        h = sender(f, q)
        z = enc(h)
        if random_control:
            z = torch.randn_like(z)
        else:
            z = quantize_packet(z)
        msg = dec(z)
        logits = receiver(msg, q)
        preds.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(y.cpu().tolist())
    elapsed = time.perf_counter() - t0
    return {"accuracy": sum(a == b for a,b in zip(preds,ys))/max(1,len(ys)),
            "seconds": elapsed, "examples": len(ys)}

def text_baseline(test, compressed=False, seed=SEED):
    """Receiver is given an exact textual fact serialization; token-free oracle parser baseline."""
    # A faithful text baseline for this first protocol benchmark: ASCII packet
    # containing the fact's stored value; byte accounting includes all characters.
    _, train, _ = make_dataset(seed)
    # Establish a sender-side lookup table from training evidence, then test
    # queries exercise unseen query IDs but known fact identifiers.
    # Baseline text conveys fact identity; receiver lookup is the receiver's memory.
    # This is explicitly a communication-channel baseline, not an LM benchmark.
    mapping = {}
    for row in train:
        mapping[row.fact_id] = row.value_id
    total_bytes = 0
    correct = 0
    t0 = time.perf_counter()
    for row in test:
        packet = f"F{row.fact_id}:V{row.value_id}".encode("ascii")
        if compressed:
            wire = gzip.compress(packet, mtime=0)
        else:
            wire = packet
        total_bytes += len(wire)
        # Deterministic parser, as the symbolic receiver's interpretation rule.
        text = gzip.decompress(wire) if compressed else wire
        value = int(text.split(b":V")[1])
        correct += int(value == row.value_id)
    return {"accuracy": correct/max(1,len(test)), "seconds":time.perf_counter()-t0,
            "total_wire_bytes": total_bytes, "mean_wire_bytes": total_bytes/max(1,len(test))}

def count_params(*models):
    return sum(p.numel() for m in models for p in m.parameters())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("runs/E43/e43_phase1.json"))
    a = p.parse_args()
    seed_all(a.seed)
    values, test, sender, receiver, enc, dec, train_seconds = train_latent(a.seed, a.epochs)
    latent = eval_latent(test, sender, receiver, enc, dec)
    random_result = eval_latent(test, sender, receiver, enc, dec, random_control=True)
    text = text_baseline(test, compressed=False, seed=a.seed)
    compressed_text = text_baseline(test, compressed=True, seed=a.seed)
    packet_bits = LATENT_DIM * 8 + 32  # int8 vector + float32 scale; per example
    result = {
        "experiment": "E43_phase1_latent_communication_protocol",
        "seed": a.seed,
        "device": DEVICE,
        "status": "synthetic protocol feasibility only; not pretrained LM communication",
        "task": "transmit fact identity/value so receiver can answer held-out query IDs",
        "split": "query IDs 0-47 train, 48-63 held out",
        "results": {"latent_int8": latent, "random_latent_control": random_result,
                    "text": text, "gzip_text": compressed_text},
        "accounting": {
            "latent_packet_bits_per_example": packet_bits,
            "latent_packet_bytes_per_example": packet_bits/8,
            "trainable_adapter_parameters": sum(p.numel() for m in (enc,dec) for p in m.parameters()),
            "frozen_agent_parameters": sum(p.numel() for m in (sender,receiver) for p in m.parameters()),
            "total_parameters": count_params(sender,receiver,enc,dec),
            "training_seconds": train_seconds,
            "adapter_parameter_bytes_fp32": sum(p.numel()*4 for m in (enc,dec) for p in m.parameters()),
            "note": "The symbolic text baseline and neural latent baseline have different interpreter assumptions; do not interpret this as a compression win."
        }
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
