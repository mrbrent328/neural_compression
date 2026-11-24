#!/usr/bin/env python3
"""E45 — Persistent Latent Memory: first-stage substrate experiment.

This is deliberately a memory-layer prototype, not a claim that arbitrary
pretrained LMs can consume this representation without an adapter.

The experiment learns:
  * a writer that maps a key/value observation into a compact latent record;
  * a query encoder that addresses records from a key alone;
  * a reader that reconstructs the associated value.

Records are serialized to disk, the model process is recreated, and recall is
measured after reload. The same learned system is compared with shuffled-memory
and empty-memory controls. A raw-vector storage baseline reports the byte cost.

Run:
    python E45/e45_latent_memory.py
    python E45/e45_latent_memory.py --device cpu --seed 45
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MemoryCodec(nn.Module):
    def __init__(self, key_dim: int, value_dim: int, latent_dim: int):
        super().__init__()
        self.writer = nn.Sequential(
            nn.Linear(key_dim + value_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
            nn.Tanh(),
        )
        self.query = nn.Sequential(
            nn.Linear(key_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        self.reader = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, value_dim),
        )

    def write(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.writer(torch.cat((key, value), dim=-1))

    def address(self, key: torch.Tensor) -> torch.Tensor:
        return self.query(key)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.reader(latent)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(n: int, key_dim: int, value_dim: int, seed: int, device):
    """Generate repeatable key/value associations with a shared nonlinear rule."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(n, key_dim, generator=g), dim=-1)
    # A fixed nonlinear mapping creates reusable structure across examples.
    transform = torch.randn(key_dim, value_dim, generator=g) / key_dim**0.5
    values = torch.tanh(keys @ transform + 0.15 * torch.sin(keys @ transform * 2.0))
    values = F.normalize(values, dim=-1)
    return keys.to(device), values.to(device)


def retrieve(codec: MemoryCodec, memory: torch.Tensor, query_key: torch.Tensor):
    q = F.normalize(codec.address(query_key), dim=-1)
    addresses = F.normalize(memory, dim=-1)
    weights = (q @ addresses.T).softmax(dim=-1)
    return weights @ memory, weights


def train_codec(codec, keys, values, epochs: int, lr: float):
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
    codec.train()
    n = keys.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(n, device=keys.device)
        # Each mini-episode writes a subset, then queries those same records.
        for batch in order.split(32):
            k, v = keys[batch], values[batch]
            latent = codec.write(k, v)
            q = F.normalize(codec.address(k), dim=-1)
            scores = q @ F.normalize(latent, dim=-1).T
            targets = torch.arange(len(batch), device=keys.device)
            contrastive = F.cross_entropy(scores * 8.0, targets)
            reconstruction = F.mse_loss(codec.decode(latent), v)
            loss = contrastive + reconstruction
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch + 1:4d}/{epochs} loss={loss.item():.5f}")


@torch.no_grad()
def evaluate(codec, keys, values, memory, shuffle_seed: int):
    codec.eval()
    recalled, ranks = [], []
    for i in range(len(keys)):
        latent, weights = retrieve(codec, memory, keys[i:i+1])
        prediction = F.normalize(codec.decode(latent), dim=-1)
        recalled.append(F.cosine_similarity(prediction, values[i:i+1]).item())
        ranks.append(int(weights.argmax(dim=-1).item() == i))
    # Control: preserve all stored records but destroy key-to-record alignment.
    generator = torch.Generator(device=memory.device).manual_seed(shuffle_seed)
    perm = torch.randperm(len(memory), generator=generator, device=memory.device)
    shuffled = memory[perm]
    shuffled_scores = []
    for i in range(len(keys)):
        latent, _ = retrieve(codec, shuffled, keys[i:i+1])
        prediction = F.normalize(codec.decode(latent), dim=-1)
        shuffled_scores.append(F.cosine_similarity(prediction, values[i:i+1]).item())
    return {
        "mean_cosine_reconstruction": float(np.mean(recalled)),
        "exact_top1_address_rate": float(np.mean(ranks)),
        "shuffled_memory_mean_cosine": float(np.mean(shuffled_scores)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--records", type=int, default=256)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--value-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=Path("runs/E45"))
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    keys, values = make_dataset(
        args.records, args.key_dim, args.value_dim, args.seed + 1, device
    )
    codec = MemoryCodec(args.key_dim, args.value_dim, args.latent_dim).to(device)
    train_codec(codec, keys, values, args.epochs, args.lr)

    # Persistent memory is written once, then loaded into a fresh codec object.
    with torch.no_grad():
        latent_records = codec.write(keys, values).detach().cpu().half()
    memory_path = args.output / "latent_memory.pt"
    model_path = args.output / "memory_codec.pt"
    torch.save({"latent_records": latent_records}, memory_path)
    torch.save(codec.state_dict(), model_path)

    reloaded_codec = MemoryCodec(args.key_dim, args.value_dim, args.latent_dim).to(device)
    reloaded_codec.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    saved = torch.load(memory_path, map_location=device, weights_only=True)
    reloaded_memory = saved["latent_records"].to(device=device, dtype=torch.float32)

    result = evaluate(reloaded_codec, keys, values, reloaded_memory, args.seed + 2)
    latent_bytes = memory_path.stat().st_size
    raw_bytes = args.records * (args.key_dim + args.value_dim) * 4
    result.update({
        "experiment": "E45_persistent_latent_memory_stage1",
        "status": "synthetic memory substrate test; not pretrained-LM integration",
        "seed": args.seed,
        "device": str(device),
        "records": args.records,
        "dimensions": {
            "key": args.key_dim, "value": args.value_dim, "latent": args.latent_dim
        },
        "persistence": {
            "saved_and_reloaded_in_fresh_model_instance": True,
            "latent_record_tensor_bytes": int(reloaded_memory.numel() * reloaded_memory.element_size()),
            "memory_file_bytes_including_container_overhead": latent_bytes,
            "raw_key_plus_value_float32_bytes": raw_bytes,
            "codec_checkpoint_bytes": model_path.stat().st_size,
        },
        "limitations": [
            "Synthetic vector associations only; no natural language or pretrained model.",
            "The latent record is a learned code, but query addressing and decoding are task-trained.",
            "This first stage does not test continual updates, interference, forgetting, or cross-model portability.",
            "Compression claims require amortizing codec/checkpoint costs and comparing matched baselines."
        ],
    })
    result_path = args.output / "e45_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
