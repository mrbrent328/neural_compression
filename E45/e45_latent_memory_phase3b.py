#!/usr/bin/env python3
"""E45 Phase 3B: frozen shared codec across held-out value domains.

Train one codec on several independently generated value domains, freeze it,
then write/evaluate stores from both seen and entirely held-out projection
domains. The evaluator pools all stores after a genuine process restart.

Run:
  python E45/e45_latent_memory_phase3b.py --device cpu --smoke
  python E45/e45_latent_memory_phase3b.py --device cpu
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


def make_projection(value_factor_dim, value_dim, domain_seed):
    """Independent projection defines one synthetic value domain."""
    g = torch.Generator(device="cpu").manual_seed(domain_seed)
    return torch.randn(value_factor_dim, value_dim, generator=g) / value_factor_dim**0.5


def make_episode(n, key_dim, value_dim, factor_dim, seed, projection):
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(n, key_dim, generator=g), dim=-1)
    factors = torch.randn(n, factor_dim, generator=g)
    values = F.normalize(torch.tanh(factors @ projection), dim=-1)
    return keys, values


def train_codec(codec, episodes, epochs, batch_size, lr, device):
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
    # Mix domains before batching so every training batch contains a mixture
    # of projection families rather than learning one domain at a time.
    keys = torch.cat([x[0] for x in episodes]).to(device)
    values = torch.cat([x[1] for x in episodes]).to(device)
    n = len(keys)
    for epoch in range(epochs):
        order = torch.randperm(n, device=device)
        total, batches = 0.0, 0
        for ix in order.split(batch_size):
            k, v = keys[ix], values[ix]
            address = F.normalize(codec.key_writer(k), dim=-1)
            query = codec.query(k)
            payload = codec.value_writer(v)
            logits = query @ address.T * 12.0
            targets = torch.arange(len(ix), device=device)
            loss = F.cross_entropy(logits, targets) + F.mse_loss(codec.decode(payload), v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch {epoch+1:4d}/{epochs} loss={total/max(1,batches):.5f}",
                  flush=True)


def config(args):
    return {
        "seed": args.seed, "train_domains": args.train_domains,
        "test_domains": args.test_domains, "records_per_store": args.records_per_store,
        "train_records_per_domain": args.train_records_per_domain,
        "epochs": args.epochs, "key_dim": args.key_dim,
        "value_dim": args.value_dim, "value_factor_dim": args.value_factor_dim,
        "address_dim": args.address_dim, "payload_dim": args.payload_dim,
        "batch_size": args.batch_size, "domain_seed_base": args.domain_seed_base,
    }


def write_phase(args, device):
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train_episodes = []
    for domain_id in range(args.train_domains):
        projection = make_projection(args.value_factor_dim, args.value_dim,
                                     args.domain_seed_base + domain_id)
        train_episodes.append(make_episode(
            args.train_records_per_domain, args.key_dim, args.value_dim,
            args.value_factor_dim, args.seed + 1000 + domain_id * 7919, projection
        ))

    codec = SharedMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    train_codec(codec, train_episodes, args.epochs, args.batch_size, args.lr, device)
    torch.save(codec.state_dict(), args.output / "shared_codec.pt")
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    codec.eval()

    # First write stores from training-seen domains, then equally many stores
    # from new projection domains never presented to the codec during training.
    domains = [("seen", i) for i in range(args.train_domains)]
    domains += [("unseen", args.train_domains + i) for i in range(args.test_domains)]
    for store_index, (group, domain_id) in enumerate(domains):
        projection = make_projection(
            args.value_factor_dim, args.value_dim,
            args.domain_seed_base + domain_id
        )
        store_seed = args.seed + 20000 + domain_id * 7919
        keys, values = make_episode(args.records_per_store, args.key_dim,
                                    args.value_dim, args.value_factor_dim,
                                    store_seed, projection)
        with torch.no_grad():
            records = codec.write(keys.to(device), values.to(device)).cpu().half()
        name = f"store_{store_index:03d}"
        torch.save({"records": records}, args.output / f"{name}.pt")
        torch.save({"keys": keys, "values": values},
                   args.output / f"{name}_eval_truth.pt")
        metadata = {
            "store_index": store_index, "group": group, "domain_id": domain_id,
            "records": args.records_per_store, "codec": "shared_codec.pt",
            "address_dim": args.address_dim, "payload_dim": args.payload_dim,
            "store_seed": store_seed,
        }
        (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2))


@torch.no_grad()
def evaluate(args, device):
    codec = SharedMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    codec.load_state_dict(torch.load(args.output / "shared_codec.pt",
                                     map_location=device, weights_only=True))
    codec.eval()

    memories, keys_all, values_all, groups = [], [], [], []
    memory_bytes = metadata_bytes = 0
    total_stores = args.train_domains + args.test_domains
    for store_index in range(total_stores):
        name = f"store_{store_index:03d}"
        mp = args.output / f"{name}.pt"
        jp = args.output / f"{name}.json"
        records = torch.load(mp, map_location="cpu", weights_only=True)["records"]
        truth = torch.load(args.output / f"{name}_eval_truth.pt",
                           map_location="cpu", weights_only=True)
        memories.append(records.to(device=device, dtype=torch.float32))
        keys_all.append(truth["keys"].to(device))
        values_all.append(truth["values"].to(device))
        groups.append("seen" if store_index < args.train_domains else "unseen")
        memory_bytes += mp.stat().st_size
        metadata_bytes += jp.stat().st_size

    memory = torch.cat(memories)
    keys = torch.cat(keys_all)
    values = torch.cat(values_all)
    addresses = F.normalize(memory[:, :args.address_dim], dim=-1)
    payloads = memory[:, args.address_dim:]
    predictions, cosine = [], []
    for start in range(0, len(keys), 256):
        stop = min(start + 256, len(keys))
        idx = (codec.query(keys[start:stop]) @ addresses.T).argmax(dim=-1)
        predictions.append(idx)
        decoded = F.normalize(codec.decode(payloads[idx]), dim=-1)
        cosine.extend(F.cosine_similarity(decoded, values[start:stop], dim=-1).cpu().tolist())
    predictions = torch.cat(predictions)
    expected = torch.arange(len(keys), device=device)
    correct = predictions == expected
    predicted_store = predictions // args.records_per_store
    expected_store = torch.arange(total_stores, device=device).repeat_interleave(
        args.records_per_store
    )
    store_correct = predicted_store == expected_store

    per_store = []
    offset = 0
    for store_index in range(total_stores):
        sl = slice(offset, offset + args.records_per_store)
        per_store.append({
            "store_index": store_index,
            "domain_group": groups[store_index],
            "top1_record_accuracy": float(correct[sl].float().mean().item()),
            "top1_store_accuracy": float(store_correct[sl].float().mean().item()),
            "mean_cosine_reconstruction": float(np.mean(cosine[offset:offset + args.records_per_store])),
        })
        offset += args.records_per_store

    def group_metrics(group):
        indices = [i for i, g in enumerate(groups) if g == group]
        mask = torch.zeros(len(keys), dtype=torch.bool, device=device)
        for i in indices:
            mask[i * args.records_per_store:(i + 1) * args.records_per_store] = True
        vals = [per_store[i]["mean_cosine_reconstruction"] for i in indices]
        return {
            "stores": len(indices),
            "query_count": int(mask.sum().item()),
            "top1_record_accuracy": float(correct[mask].float().mean().item()),
            "top1_store_accuracy": float(store_correct[mask].float().mean().item()),
            "mean_cosine_reconstruction": float(np.mean(vals)),
        }

    codec_bytes = (args.output / "shared_codec.pt").stat().st_size
    total_records = len(keys)
    latent_total = codec_bytes + memory_bytes + metadata_bytes
    raw_total = total_records * (args.key_dim + args.value_dim) * 4
    return {
        "experiment": "E45_persistent_latent_memory_phase3B",
        "status": "one frozen codec trained on multiple projection domains; seen/unseen domains evaluated after fresh-process reload",
        "configuration": config(args),
        "pooled_retrieval": {
            "top1_record_accuracy": float(correct.float().mean().item()),
            "top1_store_accuracy": float(store_correct.float().mean().item()),
            "mean_cosine_reconstruction": float(np.mean(cosine)),
            "query_count": total_records, "pooled_memory_records": len(memory),
        },
        "domain_generalization": {
            "seen_domains": group_metrics("seen"),
            "unseen_domains": group_metrics("unseen"),
            "unseen_minus_seen_cosine": (
                group_metrics("unseen")["mean_cosine_reconstruction"]
                - group_metrics("seen")["mean_cosine_reconstruction"]
            ),
        },
        "per_store": per_store,
        "storage": {
            "raw_key_value_bytes": raw_total,
            "shared_codec_checkpoint_bytes": codec_bytes,
            "all_memory_files_bytes": memory_bytes,
            "all_metadata_bytes": metadata_bytes,
            "shared_codec_total_bytes": latent_total,
            "raw_to_shared_latent_ratio": raw_total / latent_total,
            "shared_bytes_per_record": latent_total / total_records,
        },
        "limitations": [
            "Synthetic vectors only; no natural language or pretrained model.",
            "Unseen domains are new random projections from the same generator family, not arbitrary real-world semantic domains.",
            "Seen-domain evaluation uses new keys, factors, and associations; no evaluation record is reused from codec training.",
            "The codec receives no domain identifier or projection matrix at write or read time.",
            "Storage accounting excludes runtime, filesystem allocation, and indexing overhead beyond listed metadata.",
            "Continual writes, deletion, conflicts, interference under updates, and forgetting are not tested.",
        ],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("all", "train", "eval"), default="all")
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--train-domains", type=int, default=8)
    p.add_argument("--test-domains", type=int, default=8)
    p.add_argument("--records-per-store", type=int, default=1024)
    p.add_argument("--train-records-per-domain", type=int, default=1024)
    p.add_argument("--key-dim", type=int, default=32)
    p.add_argument("--value-dim", type=int, default=32)
    p.add_argument("--value-factor-dim", type=int, default=8)
    p.add_argument("--address-dim", type=int, default=8)
    p.add_argument("--payload-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--domain-seed-base", type=int, default=45001)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("runs/E45_phase3B"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.train_domains, args.test_domains = 2, 2
        args.records_per_store = 64
        args.train_records_per_domain, args.epochs, args.batch_size = 128, 5, 64
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    common = [
        "--seed", str(args.seed), "--train-domains", str(args.train_domains),
        "--test-domains", str(args.test_domains),
        "--records-per-store", str(args.records_per_store),
        "--train-records-per-domain", str(args.train_records_per_domain),
        "--key-dim", str(args.key_dim), "--value-dim", str(args.value_dim),
        "--value-factor-dim", str(args.value_factor_dim),
        "--address-dim", str(args.address_dim), "--payload-dim", str(args.payload_dim),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--domain-seed-base", str(args.domain_seed_base),
        "--device", str(device), "--output", str(args.output),
    ]
    if args.phase == "train":
        write_phase(args, device)
    elif args.phase == "eval":
        result = evaluate(args, device)
        print(json.dumps(result, indent=2))
        (args.output / "e45_phase3B_result.json").write_text(json.dumps(result, indent=2))
    else:
        # Writer exits before a fresh evaluator process loads the frozen codec.
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--phase", "train", *common], check=True)
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               "--phase", "eval", *common],
                              check=True, capture_output=True, text=True)
        result = json.loads(proc.stdout)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "e45_phase3B_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
