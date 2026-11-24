#!/usr/bin/env python3
"""
E43 Phase 1 — Synthetic latent-channel harness.

This is a protocol smoke test, not a pretrained-LM experiment. Frozen sender
and receiver bases exchange information only through an explicit packet.
"""
from __future__ import annotations
import argparse, gzip, json, random, time
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

FACT_COUNT, VALUE_COUNT, QUERY_COUNT = 256, 64, 64
LATENT_DIM, WIDTH = 16, 64

@dataclass(frozen=True)
class Row:
    fact: int
    value: int
    query: int

def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def world(seed):
    rng = random.Random(seed)
    # Each fact has a value; data rows expose fact/value pairs to sender training.
    values = [rng.randrange(VALUE_COUNT) for _ in range(FACT_COUNT)]
    train, test = [], []
    for fact, value in enumerate(values):
        # Split by query condition while preserving fact identity across train/test.
        for query in range(QUERY_COUNT):
            row = Row(fact, value, query)
            (train if query < 48 else test).append(row)
    rng.shuffle(train); rng.shuffle(test)
    return values, train, test

class SenderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fact = nn.Embedding(FACT_COUNT, WIDTH)
        self.query = nn.Embedding(QUERY_COUNT, WIDTH)
        self.net = nn.Sequential(nn.Linear(WIDTH * 2, WIDTH), nn.GELU(),
                                 nn.Linear(WIDTH, WIDTH), nn.GELU())
    def forward(self, fact, query):
        return self.net(torch.cat((self.fact(fact), self.query(query)), dim=-1))

class ReceiverBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Embedding(QUERY_COUNT, WIDTH)
        self.head = nn.Linear(WIDTH * 2, VALUE_COUNT)
    def forward(self, message, query):
        return self.head(torch.cat((message, self.query(query)), dim=-1))

class SenderAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(),
                                 nn.Linear(WIDTH, LATENT_DIM))
    def forward(self, x): return self.net(x)

class ReceiverAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(LATENT_DIM, WIDTH), nn.GELU(),
                                 nn.Linear(WIDTH, WIDTH))
    def forward(self, x): return self.net(x)

def batch(rows, device):
    return (torch.tensor([r.fact for r in rows], device=device),
            torch.tensor([r.value for r in rows], device=device),
            torch.tensor([r.query for r in rows], device=device))

def quantize_int8(z):
    # Per-example symmetric quantization: packet = int8[D] + float32 scale.
    qmax = 127
    scale = z.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(z / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale.squeeze(-1)

def dequantize(packet):
    q, scale = packet
    return q.to(torch.float32) * scale.unsqueeze(-1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("runs/E43/e43_phase1.json"))
    a = p.parse_args()
    seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values, train, test = world(a.seed)

    sender, receiver = SenderBase().to(device), ReceiverBase().to(device)
    for model in (sender, receiver):
        for param in model.parameters(): param.requires_grad_(False)
    sender_adapter, receiver_adapter = SenderAdapter().to(device), ReceiverAdapter().to(device)
    opt = torch.optim.AdamW(list(sender_adapter.parameters()) +
                            list(receiver_adapter.parameters()), lr=2e-3)
    rng = random.Random(a.seed + 1)
    start = time.perf_counter()
    for epoch in range(a.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), 256):
            rows = train[i:i+256]
            f, y, q = batch(rows, device)
            with torch.no_grad(): sender_state = sender(f, q)
            z = sender_adapter(sender_state)
            packet = quantize_int8(z)
            received = receiver_adapter(dequantize(packet))
            loss = F.cross_entropy(receiver(received, q), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    train_seconds = time.perf_counter() - start

    @torch.no_grad()
    def evaluate(kind):
        correct, total = 0, 0
        t0 = time.perf_counter()
        for i in range(0, len(test), 512):
            rows = test[i:i+512]
            f, y, q = batch(rows, device)
            h = sender(f, q)
            z = sender_adapter(h)
            if kind == "random":
                z = torch.randn_like(z)
            elif kind == "zero":
                z = torch.zeros_like(z)
            msg = receiver_adapter(z)
            pred = receiver(msg, q).argmax(-1)
            correct += int((pred == y).sum()); total += y.numel()
        return {"accuracy": correct / max(total, 1),
                "seconds": time.perf_counter() - t0, "examples": total}

    def text_eval(compressed):
        # Fairly explicit symbolic baseline: message contains fact/value directly.
        total, correct, wire_bytes = 0, 0, 0
        start_text = time.perf_counter()
        for row in test:
            payload = f"fact={row.fact};value={row.value}".encode("ascii")
            wire = gzip.compress(payload, mtime=0) if compressed else payload
            wire_bytes += len(wire)
            parsed = gzip.decompress(wire) if compressed else wire
            decoded = int(parsed.split(b"value=")[1])
            correct += int(decoded == row.value); total += 1
        return {"accuracy": correct/max(total,1), "seconds":time.perf_counter()-start_text,
                "total_wire_bytes":wire_bytes, "mean_wire_bytes":wire_bytes/max(total,1)}

    packet_bytes = LATENT_DIM + 4  # D int8 values + one float32 scale, no framing yet.
    trainable = sum(p.numel() for m in (sender_adapter, receiver_adapter) for p in m.parameters())
    frozen = sum(p.numel() for m in (sender, receiver) for p in m.parameters())
    result = {
        "experiment": "E43_phase1_synthetic_protocol_smoke_test",
        "seed": a.seed, "device": str(device),
        "status": "synthetic test only; does not test pretrained LMs or establish compression",
        "dataset": {"facts": FACT_COUNT, "train_rows":len(train), "heldout_query_rows":len(test),
                    "split":"query ids 0-47 train; 48-63 held out; same fact identities"},
        "results": {"learned_int8_latent":evaluate("latent"),
                    "random_latent":evaluate("random"), "zero_latent":evaluate("zero"),
                    "text_oracle_serialization":text_eval(False),
                    "gzip_text_oracle_serialization":text_eval(True)},
        "accounting": {
            "latent_packet_bytes_per_message":packet_bytes,
            "latent_packet_bits_per_message":packet_bytes*8,
            "trainable_adapter_parameters":trainable,
            "frozen_base_parameters":frozen,
            "total_parameters":trainable+frozen,
            "adapter_checkpoint_bytes_fp32":trainable*4,
            "frozen_base_checkpoint_bytes_fp32":frozen*4,
            "training_seconds":train_seconds,
            "note":"Text uses an explicit symbolic parser/oracle and is not a matched neural baseline. Do not compare its accuracy/time to the latent model as a compression claim."
        }
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
