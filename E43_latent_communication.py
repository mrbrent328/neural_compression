#!/usr/bin/env python3
"""
E43 Phase 1 — Synthetic latent-channel harness.

This is a protocol smoke test, not a pretrained-LM experiment. Frozen sender
and receiver bases exchange information only through an explicit quantized
packet. Train/test fact identities are disjoint.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FACT_COUNT, VALUE_COUNT = 2048, 64
LATENT_DIM, WIDTH = 16, 64


@dataclass(frozen=True)
class Row:
    fact: int
    value: int


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def world(seed: int):
    """Balanced values with a strict fact-identity split.

    Every value occurs 32 times: 24 training facts and 8 held-out facts.
    The sender receives the fact's value as private evidence, plus a fact
    identity embedding. It never receives the receiver's answer through a
    side channel; the receiver sees only the serialized latent packet.
    """
    rng = random.Random(seed)
    values = [fact % VALUE_COUNT for fact in range(FACT_COUNT)]
    train = [Row(fact, values[fact]) for fact in range(FACT_COUNT * 3 // 4)]
    test = [Row(fact, values[fact]) for fact in range(FACT_COUNT * 3 // 4, FACT_COUNT)]
    rng.shuffle(train)
    rng.shuffle(test)
    return values, train, test


class SenderBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fact = nn.Embedding(FACT_COUNT, WIDTH)
        self.value = nn.Embedding(VALUE_COUNT, WIDTH)
        self.net = nn.Sequential(
            nn.Linear(WIDTH * 2, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
            nn.GELU(),
        )

    def forward(self, fact, value):
        return self.net(torch.cat((self.fact(fact), self.value(value)), dim=-1))


class ReceiverBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(WIDTH, VALUE_COUNT)

    def forward(self, message):
        return self.head(message)


class SenderAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WIDTH, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, LATENT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class ReceiverAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
        )

    def forward(self, x):
        return self.net(x)


def batch(rows, device):
    return (
        torch.tensor([r.fact for r in rows], dtype=torch.long, device=device),
        torch.tensor([r.value for r in rows], dtype=torch.long, device=device),
    )


def quantize_int8(z, straight_through: bool = False):
    """Return a wire packet and its dequantized representation.

    Packet format: D signed int8 values plus one per-example float32 scale.
    With straight_through=True, forward values are quantized while backward
    uses an identity gradient approximation for the sender adapter.
    """
    qmax = 127
    scale = z.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(z / scale).clamp(-qmax, qmax).to(torch.int8)
    packet = (q, scale.squeeze(-1))
    decoded = dequantize(packet)
    if straight_through:
        decoded = z + (decoded - z).detach()
    return packet, decoded


def dequantize(packet):
    q, scale = packet
    return q.to(torch.float32) * scale.unsqueeze(-1)


def packet_bytes(packet):
    q, scale = packet
    return q.numel() * q.element_size() + scale.numel() * scale.element_size()


def grad_l2(module):
    grads = [p.grad.detach().norm(2) ** 2 for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack(grads).sum().sqrt().item())


def parameter_delta_l2(module, initial):
    deltas = [(p.detach() - start).norm(2) ** 2
              for p, start in zip(module.parameters(), initial)]
    if not deltas:
        return 0.0
    return float(torch.stack(deltas).sum().sqrt().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("runs/E43/e43_phase1.json"))
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values, train, test = world(args.seed)

    sender = SenderBase().to(device)
    receiver = ReceiverBase().to(device)
    for model in (sender, receiver):
        for param in model.parameters():
            param.requires_grad_(False)

    sender_adapter = SenderAdapter().to(device)
    receiver_adapter = ReceiverAdapter().to(device)
    sender_initial = [p.detach().clone() for p in sender_adapter.parameters()]
    receiver_initial = [p.detach().clone() for p in receiver_adapter.parameters()]

    optimizer = torch.optim.AdamW(
        list(sender_adapter.parameters()) + list(receiver_adapter.parameters()),
        lr=2e-3,
    )
    rng = random.Random(args.seed + 1)
    sender_grad_first = None
    receiver_grad_first = None
    start = time.perf_counter()

    for epoch in range(args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), 256):
            rows = train[i:i + 256]
            fact, target = batch(rows, device)
            with torch.no_grad():
                sender_state = sender(fact, target)

            latent = sender_adapter(sender_state)
            _packet, channel_latent = quantize_int8(latent, straight_through=True)
            received = receiver_adapter(channel_latent)
            loss = F.cross_entropy(receiver(received), target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if sender_grad_first is None:
                sender_grad_first = grad_l2(sender_adapter)
                receiver_grad_first = grad_l2(receiver_adapter)
            optimizer.step()

    train_seconds = time.perf_counter() - start

    @torch.no_grad()
    def evaluate(kind):
        correct = 0
        total = 0
        t0 = time.perf_counter()

        for i in range(0, len(test), 512):
            rows = test[i:i + 512]
            fact, target = batch(rows, device)
            if kind == "no_message":
                message = torch.zeros((len(rows), WIDTH), device=device)
            else:
                sender_state = sender(fact, target)
                latent = sender_adapter(sender_state)

                if kind == "random":
                    latent = torch.randn_like(latent)
                elif kind == "zero":
                    latent = torch.zeros_like(latent)

                packet, _ = quantize_int8(latent)
                if kind == "shuffled":
                    order = torch.roll(torch.arange(packet[0].shape[0], device=device), shifts=1)
                    packet = (packet[0][order], packet[1][order])

                message = receiver_adapter(dequantize(packet))

            prediction = receiver(message).argmax(dim=-1)
            correct += int((prediction == target).sum().item())
            total += target.numel()

        return {
            "accuracy": correct / max(total, 1),
            "seconds": time.perf_counter() - t0,
            "examples": total,
        }

    def text_stream_eval(compressed: bool):
        """Oracle serialization diagnostic; one stream, not one gzip per row."""
        lines = [f"fact={row.fact};value={row.value}" for row in test]
        payload = ("\n".join(lines) + "\n").encode("ascii")
        wire = gzip.compress(payload, mtime=0) if compressed else payload
        decoded = gzip.decompress(wire) if compressed else wire
        decoded_values = [
            int(line.split(b";value=", 1)[1])
            for line in decoded.splitlines()
        ]
        correct = sum(
            int(pred == row.value)
            for pred, row in zip(decoded_values, test)
        )
        return {
            "accuracy": correct / max(len(test), 1),
            "total_wire_bytes": len(wire),
            "mean_wire_bytes_per_message": len(wire) / max(len(test), 1),
            "uncompressed_stream_bytes": len(payload),
        }

    test_counts = Counter(row.value for row in test)
    majority_accuracy = max(test_counts.values()) / len(test)
    packet_example, _ = quantize_int8(torch.zeros((1, LATENT_DIM), device=device))
    packet_bytes_per_message = packet_bytes(packet_example)

    trainable = sum(
        p.numel()
        for model in (sender_adapter, receiver_adapter)
        for p in model.parameters()
    )
    frozen = sum(
        p.numel()
        for model in (sender, receiver)
        for p in model.parameters()
    )

    result = {
        "experiment": "E43_phase1_synthetic_protocol_smoke_test_v2",
        "seed": args.seed,
        "device": str(device),
        "status": (
            "synthetic protocol test only; does not test pretrained LMs "
            "or establish a compression advantage"
        ),
        "dataset": {
            "facts": FACT_COUNT,
            "value_classes": VALUE_COUNT,
            "train_facts": FACT_COUNT * 3 // 4,
            "heldout_facts": FACT_COUNT // 4,
            "train_rows": len(train),
            "heldout_rows": len(test),
            "split": (
                "disjoint fact identities; 24 train facts and 8 held-out "
                "facts per value class"
            ),
            "value_balance": {
                "train_examples_per_value": FACT_COUNT * 3 // 4 // VALUE_COUNT,
                "test_examples_per_value": FACT_COUNT // 4 // VALUE_COUNT,
            },
        },
        "results": {
            "learned_quantized_latent": evaluate("learned"),
            "shuffled_quantized_latent": evaluate("shuffled"),
            "random_quantized_latent": evaluate("random"),
            "zero_quantized_latent": evaluate("zero"),
            "no_message": evaluate("no_message"),
            "test_majority_class": {"accuracy": majority_accuracy},
            "text_oracle_stream": text_stream_eval(False),
            "gzip_text_oracle_stream": text_stream_eval(True),
        },
        "learning_diagnostics": {
            "sender_adapter_first_step_grad_l2": sender_grad_first,
            "receiver_adapter_first_step_grad_l2": receiver_grad_first,
            "sender_adapter_parameter_delta_l2": parameter_delta_l2(
                sender_adapter, sender_initial
            ),
            "receiver_adapter_parameter_delta_l2": parameter_delta_l2(
                receiver_adapter, receiver_initial
            ),
        },
        "accounting": {
            "latent_packet_bytes_per_message": packet_bytes_per_message,
            "latent_packet_bits_per_message": packet_bytes_per_message * 8,
            "trainable_adapter_parameters": trainable,
            "frozen_base_parameters": frozen,
            "total_parameters": trainable + frozen,
            "adapter_checkpoint_bytes_fp32": trainable * 4,
            "frozen_base_checkpoint_bytes_fp32": frozen * 4,
            "training_seconds": train_seconds,
            "quantization": (
                "int8 forward pass with straight-through identity gradient "
                "approximation during training"
            ),
            "text_baseline_warning": (
                "Text serialization is a symbolic oracle diagnostic, not a "
                "matched neural sender/receiver baseline. Its accuracy and "
                "timing must not be used to claim a latent compression win."
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
