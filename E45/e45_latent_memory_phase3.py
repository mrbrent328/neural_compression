#!/usr/bin/env python3
"""E45 Phase 3A: one frozen codec, multiple independent persistent memories.

The codec is trained once on a dedicated training episode, frozen, and used
without updates to write multiple new stores. Each store contains new keys and
new values, but values come from the same declared low-dimensional domain.

The default runner starts a writer process, waits for it to exit, then starts
a separate evaluator process. Evaluation queries the pooled stores to measure
cross-store isolation as well as record retrieval.

Run:
  python E45/e45_latent_memory_phase3.py --device cpu --smoke
  python E45/e45_latent_memory_phase3.py --device cpu
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


class SharedMemoryCodec(nn.Module):
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
        payload = self.value_writer(values)
        return torch.cat((address, payload), dim=-1)

    def query(self, keys):
        return F.normalize(self.query_encoder(keys), dim=-1)

    def decode(self, payload):
        return self.reader(payload)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def value_projection(value_factor_dim, value_dim, domain_seed):
    """One fixed value-domain transform shared by train and all stores."""
    g = torch.Generator(device="cpu").manual_seed(domain_seed)
    return torch.randn(value_factor_dim, value_dim, generator=g) / value_factor_dim**0.5


def make_episode(n, key_dim, value_dim, value_factor_dim, seed, projection):
    """New random keys and values; key/value pairings are arbitrary."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(n, key_dim, generator=g), dim=-1)
    factors = torch.randn(n, value_factor_dim, generator=g)
    values = F.normalize(torch.tanh(factors @ projection), dim=-1)
    return keys, values


def train_codec(codec, keys, values, epochs, batch_size, lr):
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
    n = len(keys)
    for epoch in range(epochs):
        order = torch.randperm(n, device=keys.device)
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


def build_config(args):
    return {
        "seed": args.seed, "stores": args.stores, "records_per_store": args.records_per_store,
        "train_records": args.train_records, "epochs": args.epochs,
        "key_dim": args.key_dim, "value_dim": args.value_dim,
        "value_factor_dim": args.value_factor_dim, "address_dim": args.address_dim,
        "payload_dim": args.payload_dim, "batch_size": args.batch_size,
        "domain_seed": args.domain_seed,
    }


def write_phase(args, device):
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    projection = value_projection(args.value_factor_dim, args.value_dim, args.domain_seed)

    train_keys, train_values = make_episode(
        args.train_records, args.key_dim, args.value_dim, args.value_factor_dim,
        args.seed + 1000, projection
    )
    codec = SharedMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    train_codec(codec, train_keys.to(device), train_values.to(device),
                args.epochs, args.batch_size, args.lr)
    # The checkpoint is written once. No optimizer or training state is retained.
    torch.save(codec.state_dict(), args.output / "shared_codec.pt")
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    codec.eval()

    for store_id in range(args.stores):
        store_seed = args.seed + 10000 + store_id * 7919
        keys, values = make_episode(args.records_per_store, args.key_dim,
                                    args.value_dim, args.value_factor_dim,
                                    store_seed, projection)
        with torch.no_grad():
            records = codec.write(keys.to(device), values.to(device)).cpu().half()
        store_name = f"store_{store_id:03d}"
        torch.save({"records": records}, args.output / f"{store_name}.pt")
        truth = {"keys": keys, "values": values}
        torch.save(truth, args.output / f"{store_name}_eval_truth.pt")
        metadata = {
            "store_id": store_id, "records": args.records_per_store,
            "address_dim": args.address_dim, "payload_dim": args.payload_dim,
            "codec": "shared_codec.pt", "store_seed": store_seed,
        }
        (args.output / f"{store_name}.json").write_text(json.dumps(metadata, indent=2))


@torch.no_grad()
def evaluate(args, device):
    codec = SharedMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    codec.load_state_dict(torch.load(args.output / "shared_codec.pt",
                                     map_location=device, weights_only=True))
    codec.eval()
    memories, all_keys, all_values, store_labels, record_labels = [], [], [], [], []
    memory_file_bytes = 0
    metadata_bytes = 0
    for store_id in range(args.stores):
        store_name = f"store_{store_id:03d}"
        memory_path = args.output / f"{store_name}.pt"
        metadata_path = args.output / f"{store_name}.json"
        records = torch.load(memory_path, map_location="cpu",
                             weights_only=True)["records"].to(device=device, dtype=torch.float32)
        truth = torch.load(args.output / f"{store_name}_eval_truth.pt",
                           map_location="cpu", weights_only=True)
        keys, values = truth["keys"].to(device), truth["values"].to(device)
        memories.append(records)
        all_keys.append(keys)
        all_values.append(values)
        store_labels.extend([store_id] * len(keys))
        record_labels.extend(range(len(keys)))
        memory_file_bytes += memory_path.stat().st_size
        metadata_bytes += metadata_path.stat().st_size

    memory = torch.cat(memories, dim=0)
    keys = torch.cat(all_keys, dim=0)
    values = torch.cat(all_values, dim=0)
    addresses = F.normalize(memory[:, :args.address_dim], dim=-1)
    payloads = memory[:, args.address_dim:]
    predicted_indices = []
    cosine = []
    for start in range(0, len(keys), 256):
        stop = min(start + 256, len(keys))
        idx = (codec.query(keys[start:stop]) @ addresses.T).argmax(dim=-1)
        predicted_indices.append(idx)
        decoded = F.normalize(codec.decode(payloads[idx]), dim=-1)
        cosine.extend(F.cosine_similarity(decoded, values[start:stop], dim=-1).cpu().tolist())
    predicted_indices = torch.cat(predicted_indices)
    expected = torch.arange(len(keys), device=device)
    correct_record = predicted_indices == expected
    predicted_store = predicted_indices // args.records_per_store
    expected_store = torch.tensor(store_labels, device=device)
    store_accuracy = float((predicted_store == expected_store).float().mean().item())
    record_accuracy = float(correct_record.float().mean().item())

    codec_bytes = (args.output / "shared_codec.pt").stat().st_size
    latent_total = codec_bytes + memory_file_bytes + metadata_bytes
    raw_total = len(keys) * (args.key_dim + args.value_dim) * 4
    duplicated_codec_total = args.stores * codec_bytes + memory_file_bytes + metadata_bytes
    per_store = []
    offset = 0
    for store_id in range(args.stores):
        count = args.records_per_store
        local = correct_record[offset:offset + count]
        local_cos = cosine[offset:offset + count]
        per_store.append({
            "store_id": store_id,
            "top1_record_accuracy": float(local.float().mean().item()),
            "mean_cosine_reconstruction": float(np.mean(local_cos)),
        })
        offset += count

    return {
        "experiment": "E45_persistent_latent_memory_phase3A",
        "status": "one frozen codec reused across independent stores; fresh-process evaluation",
        "configuration": build_config(args),
        "pooled_retrieval": {
            "top1_record_accuracy": record_accuracy,
            "top1_store_accuracy": store_accuracy,
            "mean_cosine_reconstruction": float(np.mean(cosine)),
            "query_count": len(keys),
            "pooled_memory_records": len(memory),
        },
        "per_store": per_store,
        "storage": {
            "raw_key_value_bytes": raw_total,
            "shared_codec_checkpoint_bytes": codec_bytes,
            "all_memory_files_bytes": memory_file_bytes,
            "all_metadata_bytes": metadata_bytes,
            "shared_codec_total_bytes": latent_total,
            "duplicated_codec_accounting_bytes": duplicated_codec_total,
            "shared_vs_duplicated_codec_savings_bytes": duplicated_codec_total - latent_total,
            "raw_to_shared_latent_ratio": raw_total / latent_total,
            "shared_bytes_per_record": latent_total / len(keys),
            "duplicated_codec_bytes_per_record": duplicated_codec_total / len(keys),
        },
        "limitations": [
            "Synthetic vectors only; no natural language or pretrained model.",
            "Train and test stores share a declared low-dimensional value domain and fixed projection.",
            "The codec is tested on unseen random keys and associations, not arbitrary unseen value domains.",
            "The duplicated-codec comparison is storage accounting, not separately trained codec quality.",
            "Runtime, filesystem allocation, and indexing overhead beyond listed metadata are excluded.",
            "Continual writes, deletion, conflicts, interference, and forgetting are not tested.",
        ],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("all", "train", "eval"), default="all")
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--stores", type=int, default=8)
    p.add_argument("--records-per-store", type=int, default=1024)
    p.add_argument("--train-records", type=int, default=8192)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--value-dim", type=int, default=32)
    p.add_argument("--value-factor-dim", type=int, default=8)
    p.add_argument("--address-dim", type=int, default=8)
    p.add_argument("--payload-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--domain-seed", type=int, default=45001)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("runs/E45_phase3A"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.stores, args.records_per_store = 2, 64
        args.train_records, args.epochs, args.batch_size = 256, 5, 64
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.phase == "train":
        write_phase(args, device)
        return
    if args.phase == "eval":
        result = evaluate(args, device)
        print(json.dumps(result, indent=2))
        (args.output / "e45_phase3A_result.json").write_text(json.dumps(result, indent=2))
        return

    # Dedicated writer process; it exits before the evaluator is started.
    common = [
        "--seed", str(args.seed), "--stores", str(args.stores),
        "--records-per-store", str(args.records_per_store),
        "--train-records", str(args.train_records),
        "--key-dim", str(args.key_dim), "--value-dim", str(args.value_dim),
        "--value-factor-dim", str(args.value_factor_dim),
        "--address-dim", str(args.address_dim), "--payload-dim", str(args.payload_dim),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--domain-seed", str(args.domain_seed),
        "--device", str(device), "--output", str(args.output),
    ]
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--phase", "train", *common], check=True)
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                           "--phase", "eval", *common],
                          check=True, capture_output=True, text=True)
    result = json.loads(proc.stdout)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "e45_phase3A_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
