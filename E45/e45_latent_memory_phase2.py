#!/usr/bin/env python3
"""E45 Phase 2: random-association integrity, restart, and scaling test.

Values are independently randomized from keys. Each latent record stores a
learned address code plus a compressed payload code. Retrieval is evaluated
against the entire memory. The evaluation is launched in a separate Python
process after the writer process exits.

Run:
  python E45/e45_latent_memory_phase2.py --device cpu --smoke
  python E45/e45_latent_memory_phase2.py --device cpu
"""
from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LatentMemoryCodec(nn.Module):
    def __init__(self, key_dim, value_dim, address_dim, payload_dim):
        super().__init__()
        self.key_writer = nn.Sequential(nn.Linear(key_dim, 128), nn.GELU(),
                                        nn.Linear(128, address_dim))
        self.query_encoder = nn.Sequential(nn.Linear(key_dim, 128), nn.GELU(),
                                           nn.Linear(128, address_dim))
        self.value_writer = nn.Sequential(nn.Linear(value_dim, 128), nn.GELU(),
                                           nn.Linear(128, payload_dim), nn.Tanh())
        self.reader = nn.Sequential(nn.Linear(payload_dim, 128), nn.GELU(),
                                    nn.Linear(128, value_dim))

    def write(self, keys, values):
        address = F.normalize(self.key_writer(keys), dim=-1)
        return torch.cat((address, self.value_writer(values)), dim=-1)

    def query(self, keys):
        return F.normalize(self.query_encoder(keys), dim=-1)

    def decode(self, payload):
        return self.reader(payload)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_episode(n, key_dim, value_dim, factor_dim, seed, device):
    """Independent keys and per-record values with shared low-dimensional structure."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(n, key_dim, generator=g), dim=-1)
    factors = torch.randn(n, factor_dim, generator=g)
    projection = torch.randn(factor_dim, value_dim, generator=g) / factor_dim**0.5
    values = F.normalize(torch.tanh(factors @ projection), dim=-1)
    return keys.to(device), values.to(device)


def train(codec, keys, values, epochs, batch_size, lr):
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
    for epoch in range(epochs):
        order = torch.randperm(len(keys), device=keys.device)
        total, batches = 0.0, 0
        for ix in order.split(batch_size):
            k, v = keys[ix], values[ix]
            address = F.normalize(codec.key_writer(k), dim=-1)
            query = codec.query(k)
            payload = codec.value_writer(v)
            logits = query @ address.T * 12.0
            targets = torch.arange(len(ix), device=keys.device)
            loss = F.cross_entropy(logits, targets) + F.mse_loss(codec.decode(payload), v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch+1:4d}/{epochs} loss={total/max(1,batches):.5f}",
                  flush=True)


@torch.no_grad()
def evaluate(codec, keys, values, memory, address_dim, expected_indices=None):
    codec.eval()
    addresses = F.normalize(memory[:, :address_dim].float(), dim=-1)
    payloads = memory[:, address_dim:].float()
    if expected_indices is None:
        expected_indices = torch.arange(len(memory), device=keys.device)
    else:
        expected_indices = expected_indices.to(keys.device)
    hits, cosine = 0, []
    for start in range(0, len(keys), 256):
        stop = min(start + 256, len(keys))
        idx = (codec.query(keys[start:stop]) @ addresses.T).argmax(dim=-1)
        original_ids = expected_indices[idx]
        targets = torch.arange(start, stop, device=keys.device)
        hits += int((original_ids == targets).sum())
        decoded = F.normalize(codec.decode(payloads[idx]), dim=-1)
        cosine.extend(F.cosine_similarity(decoded, values[start:stop], dim=-1).cpu().tolist())
    return {"top1_address_accuracy": hits / len(keys),
            "mean_cosine_reconstruction": float(np.mean(cosine))}


def train_and_save(args, seed, n, device, folder):
    seed_all(seed)
    keys, values = make_episode(n, args.key_dim, args.value_dim,
                                args.value_factor_dim, seed + 100000 + n, device)
    codec = LatentMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    train(codec, keys, values, args.epochs, args.batch_size, args.lr)
    with torch.no_grad():
        records = codec.write(keys, values).detach().cpu().half()
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(codec.state_dict(), folder / "codec.pt")
    torch.save({"records": records}, folder / "memory.pt")
    torch.save({"keys": keys.cpu(), "values": values.cpu()}, folder / "episode.pt")


def evaluate_after_restart(args, seed, n, device, folder):
    codec = LatentMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    codec.load_state_dict(torch.load(folder / "codec.pt", map_location=device,
                                     weights_only=True))
    memory = torch.load(folder / "memory.pt", map_location="cpu",
                        weights_only=True)["records"].to(device=device, dtype=torch.float32)
    episode = torch.load(folder / "episode.pt", map_location=device, weights_only=True)
    keys, values = episode["keys"], episode["values"]
    aligned = evaluate(codec, keys, values, memory, args.address_dim)

    # Permuting rows must not change content-addressed retrieval.
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 7))
    reordered = evaluate(codec, keys, values, memory[perm], args.address_dim,
                         expected_indices=perm)

    # Keep addresses fixed but permute payloads: this destroys key/value binding.
    payload_order = torch.roll(torch.arange(n, device=device), shifts=1)
    mismatched = torch.cat((memory[:, :args.address_dim],
                            memory[:, args.address_dim:][payload_order]), dim=-1)
    mismatch = evaluate(codec, keys, values, mismatched, args.address_dim)

    raw_bytes = n * (args.key_dim + args.value_dim) * 4
    record_bytes = n * (args.address_dim + args.payload_dim) * 2
    checkpoint_bytes = (folder / "codec.pt").stat().st_size
    memory_file_bytes = (folder / "memory.pt").stat().st_size
    total_bytes = checkpoint_bytes + memory_file_bytes
    return {
        "seed": seed, "records": n,
        "after_separate_process_reload": aligned,
        "row_order_permutation_control": reordered,
        "key_payload_mismatch_control": mismatch,
        "storage": {
            "raw_key_value_bytes": raw_bytes,
            "latent_record_tensor_bytes": record_bytes,
            "latent_memory_file_bytes": memory_file_bytes,
            "codec_checkpoint_bytes": checkpoint_bytes,
            "total_latent_system_bytes": total_bytes,
            "total_bytes_per_record_including_codec": total_bytes / n,
            "raw_to_latent_total_ratio": raw_bytes / total_bytes,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("all", "train", "eval"), default="all")
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--seeds", type=int, nargs="+", default=[45, 46, 47])
    p.add_argument("--records", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--value-dim", type=int, default=32)
    p.add_argument("--value-factor-dim", type=int, default=8)
    p.add_argument("--address-dim", type=int, default=8)
    p.add_argument("--payload-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("runs/E45_phase2"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.records, args.epochs = [args.seed], [64], 5
        args.batch_size = 64
    device = torch.device(args.device)
    if args.phase == "eval":
        result = evaluate_after_restart(args, args.seed, args.records[0], device,
                                        args.output / f"seed{args.seed}_n{args.records[0]}")
        print(json.dumps(result, indent=2))
        return

    summaries = []
    for seed in args.seeds:
        for n in args.records:
            folder = args.output / f"seed{seed}_n{n}"
            train_and_save(args, seed, n, device, folder)
            if args.phase == "train":
                continue
            # A distinct interpreter proves reload after the writer process exits.
            cmd = [sys.executable, str(Path(__file__).resolve()), "--phase", "eval",
                   "--seed", str(seed), "--records", str(n),
                   "--key-dim", str(args.key_dim), "--value-dim", str(args.value_dim),
                   "--value-factor-dim", str(args.value_factor_dim),
                   "--address-dim", str(args.address_dim), "--payload-dim", str(args.payload_dim),
                   "--device", str(device), "--output", str(args.output)]
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            summaries.append(json.loads(proc.stdout))
            print(f"reloaded seed={seed} records={n}", flush=True)

    if args.phase == "all":
        result = {
            "experiment": "E45_persistent_latent_memory_phase2",
            "status": "random-association integrity, retrieval, restart, and scaling test",
            "device": str(device),
            "configuration": {
                "seeds": args.seeds, "record_counts": args.records, "epochs": args.epochs,
                "key_dim": args.key_dim, "value_dim": args.value_dim,
                "value_factor_dim": args.value_factor_dim, "address_dim": args.address_dim,
                "payload_dim": args.payload_dim, "batch_size": args.batch_size,
            },
            "runs": summaries,
            "limitations": [
                "Synthetic vectors, not natural language or pretrained-model memory.",
                "Values have low-dimensional structure; key/value associations are independently randomized.",
                "Codec portability across tasks and model families is not tested.",
                "Storage includes the codec checkpoint but excludes runtime and filesystem overhead.",
                "No continual writes, conflicts, deletion, interference, or forgetting are tested.",
            ],
        }
        args.output.mkdir(parents=True, exist_ok=True)
        out = args.output / "e45_phase2_summary.json"
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
