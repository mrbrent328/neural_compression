#!/usr/bin/env python3
"""E45 Phase 3C: controlled value-path generalization ablation.

Factorial sweep over training-domain count and payload dimension. The total
training-record budget and evaluation-store count are fixed across cells.
Each cell uses a fresh codec and a genuine writer/evaluator process restart.

Examples:
  python E45/e45_latent_memory_phase3c.py --device cpu --smoke
  python E45/e45_latent_memory_phase3c.py --device cpu
  python E45/e45_latent_memory_phase3c.py --device cpu --seeds 45 46 47
"""
from __future__ import annotations
import argparse
import itertools
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
    def __init__(self, key_dim: int, value_dim: int, address_dim: int, payload_dim: int):
        super().__init__()
        self.key_writer = nn.Sequential(nn.Linear(key_dim, 128), nn.GELU(), nn.Linear(128, address_dim))
        self.query_encoder = nn.Sequential(nn.Linear(key_dim, 128), nn.GELU(), nn.Linear(128, address_dim))
        self.value_writer = nn.Sequential(nn.Linear(value_dim, 128), nn.GELU(), nn.Linear(128, payload_dim), nn.Tanh())
        self.reader = nn.Sequential(nn.Linear(payload_dim, 128), nn.GELU(), nn.Linear(128, value_dim))

    def write(self, keys, values):
        address = F.normalize(self.key_writer(keys), dim=-1)
        payload = self.value_writer(values)
        return torch.cat((address, payload), dim=-1)

    def query(self, keys):
        return F.normalize(self.query_encoder(keys), dim=-1)

    def decode(self, payload):
        return self.reader(payload)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_projection(factor_dim: int, value_dim: int, domain_seed: int):
    g = torch.Generator(device="cpu").manual_seed(domain_seed)
    return torch.randn(factor_dim, value_dim, generator=g) / factor_dim**0.5


def make_episode(n: int, key_dim: int, value_dim: int, factor_dim: int, seed: int, projection):
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(torch.randn(n, key_dim, generator=g), dim=-1)
    factors = torch.randn(n, factor_dim, generator=g)
    values = F.normalize(torch.tanh(factors @ projection), dim=-1)
    return keys, values


def train_codec(codec, episodes, epochs: int, batch_size: int, lr: float, device):
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
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
        if (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"epoch {epoch+1:4d}/{epochs} loss={total/max(1,batches):.5f}", flush=True)


def run_config(args):
    return {
        "seed": args.seed, "train_domains": args.train_domains,
        "train_record_budget": args.train_record_budget,
        "train_records_per_domain": args.train_record_budget // args.train_domains,
        "seen_eval_stores": args.seen_eval_stores, "unseen_eval_stores": args.unseen_eval_stores,
        "records_per_store": args.records_per_store, "epochs": args.epochs,
        "key_dim": args.key_dim, "value_dim": args.value_dim,
        "value_factor_dim": args.value_factor_dim, "address_dim": args.address_dim,
        "payload_dim": args.payload_dim, "batch_size": args.batch_size,
        "domain_seed_base": args.domain_seed_base,
    }


def write_phase(args, device):
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    per_domain = args.train_record_budget // args.train_domains
    train_episodes = []
    for domain_id in range(args.train_domains):
        projection = make_projection(args.value_factor_dim, args.value_dim,
                                     args.domain_seed_base + domain_id)
        train_episodes.append(make_episode(
            per_domain, args.key_dim, args.value_dim, args.value_factor_dim,
            args.seed + 1000 + domain_id * 7919, projection))

    codec = SharedMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    train_codec(codec, train_episodes, args.epochs, args.batch_size, args.lr, device)
    torch.save(codec.state_dict(), args.output / "shared_codec.pt")
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)

    # Keep evaluation fixed at two seen and two unseen stores in every cell.
    # For larger training-domain counts, choose two seen domains evenly.
    if args.seen_eval_stores > args.train_domains:
        raise ValueError("seen_eval_stores cannot exceed train_domains")
    if args.seen_eval_stores == 1:
        seen_ids = [0]
    else:
        seen_ids = np.linspace(0, args.train_domains - 1,
                               args.seen_eval_stores, dtype=int).tolist()
    domain_specs = [("seen", d) for d in seen_ids]
    domain_specs += [("unseen", args.train_domains + i)
                     for i in range(args.unseen_eval_stores)]

    for store_index, (group, domain_id) in enumerate(domain_specs):
        projection = make_projection(args.value_factor_dim, args.value_dim,
                                     args.domain_seed_base + domain_id)
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
        metadata = {"store_index": store_index, "group": group,
                    "domain_id": domain_id, "records": args.records_per_store,
                    "address_dim": args.address_dim, "payload_dim": args.payload_dim,
                    "store_seed": store_seed}
        (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2))


@torch.no_grad()
def evaluate(args, device):
    codec = SharedMemoryCodec(args.key_dim, args.value_dim, args.address_dim,
                              args.payload_dim).to(device)
    codec.load_state_dict(torch.load(args.output / "shared_codec.pt",
                                     map_location=device, weights_only=True))
    codec.eval()
    total_stores = args.seen_eval_stores + args.unseen_eval_stores
    memories, keys_all, values_all, groups = [], [], [], []
    memory_bytes = metadata_bytes = 0
    for store_index in range(total_stores):
        name = f"store_{store_index:03d}"
        mp, jp = args.output / f"{name}.pt", args.output / f"{name}.json"
        records = torch.load(mp, map_location="cpu", weights_only=True)["records"]
        truth = torch.load(args.output / f"{name}_eval_truth.pt",
                           map_location="cpu", weights_only=True)
        memories.append(records.to(device=device, dtype=torch.float32))
        keys_all.append(truth["keys"].to(device))
        values_all.append(truth["values"].to(device))
        groups.append("seen" if store_index < args.seen_eval_stores else "unseen")
        memory_bytes += mp.stat().st_size
        metadata_bytes += jp.stat().st_size

    memory, keys, values = torch.cat(memories), torch.cat(keys_all), torch.cat(values_all)
    addresses = F.normalize(memory[:, :args.address_dim], dim=-1)
    payloads = memory[:, args.address_dim:]
    predictions, cosine, mse = [], [], []
    for start in range(0, len(keys), 256):
        stop = min(start + 256, len(keys))
        idx = (codec.query(keys[start:stop]) @ addresses.T).argmax(dim=-1)
        predictions.append(idx)
        decoded = codec.decode(payloads[idx])
        normalized = F.normalize(decoded, dim=-1)
        cosine.extend(F.cosine_similarity(normalized, values[start:stop], dim=-1).cpu().tolist())
        mse.extend(F.mse_loss(decoded, values[start:stop], reduction="none").mean(-1).cpu().tolist())
    predictions = torch.cat(predictions)
    expected = torch.arange(len(keys), device=device)
    correct = predictions == expected
    predicted_store = predictions // args.records_per_store
    expected_store = torch.arange(total_stores, device=device).repeat_interleave(args.records_per_store)
    store_correct = predicted_store == expected_store

    per_store = []
    for i in range(total_stores):
        sl = slice(i * args.records_per_store, (i + 1) * args.records_per_store)
        per_store.append({
            "store_index": i, "domain_group": groups[i],
            "top1_record_accuracy": float(correct[sl].float().mean().item()),
            "top1_store_accuracy": float(store_correct[sl].float().mean().item()),
            "mean_cosine_reconstruction": float(np.mean(cosine[sl])),
            "mean_squared_error": float(np.mean(mse[sl])),
        })

    def group_metrics(group):
        ix = [i for i, g in enumerate(groups) if g == group]
        mask = torch.zeros(len(keys), dtype=torch.bool, device=device)
        for i in ix:
            mask[i * args.records_per_store:(i + 1) * args.records_per_store] = True
        return {
            "stores": len(ix), "query_count": int(mask.sum().item()),
            "top1_record_accuracy": float(correct[mask].float().mean().item()),
            "top1_store_accuracy": float(store_correct[mask].float().mean().item()),
            "mean_cosine_reconstruction": float(np.mean([per_store[i]["mean_cosine_reconstruction"] for i in ix])),
            "mean_squared_error": float(np.mean([per_store[i]["mean_squared_error"] for i in ix])),
        }

    codec_bytes = (args.output / "shared_codec.pt").stat().st_size
    total_records = len(keys)
    latent_total = codec_bytes + memory_bytes + metadata_bytes
    raw_total = total_records * (args.key_dim + args.value_dim) * 4
    seen, unseen = group_metrics("seen"), group_metrics("unseen")
    return {
        "experiment": "E45_persistent_latent_memory_phase3C",
        "status": "controlled train-domain-count by payload-dimension ablation; frozen codec; fresh-process evaluation",
        "configuration": run_config(args),
        "pooled_retrieval": {
            "top1_record_accuracy": float(correct.float().mean().item()),
            "top1_store_accuracy": float(store_correct.float().mean().item()),
            "mean_cosine_reconstruction": float(np.mean(cosine)),
            "mean_squared_error": float(np.mean(mse)),
            "query_count": total_records, "pooled_memory_records": len(memory),
        },
        "domain_generalization": {
            "seen_domains": seen, "unseen_domains": unseen,
            "unseen_minus_seen_cosine": unseen["mean_cosine_reconstruction"] - seen["mean_cosine_reconstruction"],
            "unseen_minus_seen_mse": unseen["mean_squared_error"] - seen["mean_squared_error"],
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
            "Synthetic vectors only; held-out domains are random projections from the same generator family.",
            "The 32-dimensional learned payload removes the dimensionality bottleneck but remains a learned float16 autoencoder, not a bit-identical raw-value oracle.",
            "Evaluation uses a fixed number of seen and unseen stores; for larger training-domain counts, two seen domains are sampled evenly across the training-domain range.",
            "Storage excludes runtime, filesystem allocation, and indexing overhead beyond listed metadata.",
            "Continual writes, deletion, conflicts, interference under updates, forgetting, natural language, and pretrained models are not tested.",
        ],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("all", "train", "eval"), default="all")
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--train-domains", type=int, default=8)
    p.add_argument("--train-record-budget", type=int, default=8192)
    p.add_argument("--seen-eval-stores", type=int, default=2)
    p.add_argument("--unseen-eval-stores", type=int, default=2)
    p.add_argument("--records-per-store", type=int, default=1024)
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
    p.add_argument("--output", type=Path, default=Path("runs/E45_phase3C"))
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--domain-counts", type=int, nargs="*", default=None)
    p.add_argument("--payload-dims", type=int, nargs="*", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.train_domains, args.train_record_budget = 2, 128
        args.seen_eval_stores, args.unseen_eval_stores = 2, 2
        args.records_per_store, args.epochs, args.batch_size = 64, 3, 64
    return args


def cell_command(script, args, phase, output):
    return [sys.executable, str(script), "--phase", phase,
            "--seed", str(args.seed), "--train-domains", str(args.train_domains),
            "--train-record-budget", str(args.train_record_budget),
            "--seen-eval-stores", str(args.seen_eval_stores),
            "--unseen-eval-stores", str(args.unseen_eval_stores),
            "--records-per-store", str(args.records_per_store),
            "--key-dim", str(args.key_dim), "--value-dim", str(args.value_dim),
            "--value-factor-dim", str(args.value_factor_dim),
            "--address-dim", str(args.address_dim), "--payload-dim", str(args.payload_dim),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--lr", str(args.lr), "--domain-seed-base", str(args.domain_seed_base),
            "--device", str(args.device), "--output", str(output)]


def run_cell(args):
    script = Path(__file__).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(cell_command(script, args, "train", args.output), check=True)
    proc = subprocess.run(cell_command(script, args, "eval", args.output),
                          check=True, capture_output=True, text=True)
    result = json.loads(proc.stdout)
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    return result


def aggregate(results):
    groups = {}
    for r in results:
        c = r["configuration"]
        key = (c["train_domains"], c["payload_dim"])
        groups.setdefault(key, []).append(r)
    cells = []
    metrics = [
        ("seen_cosine", lambda x: x["domain_generalization"]["seen_domains"]["mean_cosine_reconstruction"]),
        ("unseen_cosine", lambda x: x["domain_generalization"]["unseen_domains"]["mean_cosine_reconstruction"]),
        ("unseen_minus_seen_cosine", lambda x: x["domain_generalization"]["unseen_minus_seen_cosine"]),
        ("seen_mse", lambda x: x["domain_generalization"]["seen_domains"]["mean_squared_error"]),
        ("unseen_mse", lambda x: x["domain_generalization"]["unseen_domains"]["mean_squared_error"]),
        ("unseen_minus_seen_mse", lambda x: x["domain_generalization"]["unseen_minus_seen_mse"]),
        ("record_accuracy", lambda x: x["pooled_retrieval"]["top1_record_accuracy"]),
        ("store_accuracy", lambda x: x["pooled_retrieval"]["top1_store_accuracy"]),
        ("raw_to_latent_ratio", lambda x: x["storage"]["raw_to_shared_latent_ratio"]),
        ("bytes_per_record", lambda x: x["storage"]["shared_bytes_per_record"]),
    ]
    for (domains, payload), runs in sorted(groups.items()):
        row = {"train_domains": domains, "payload_dim": payload, "seed_count": len(runs)}
        for label, getter in metrics:
            vals = np.asarray([getter(x) for x in runs], dtype=float)
            row[label + "_mean"] = float(vals.mean())
            row[label + "_sample_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[label + "_min"] = float(vals.min())
            row[label + "_max"] = float(vals.max())
        cells.append(row)
    return {
        "experiment": "E45_persistent_latent_memory_phase3C_sweep",
        "run_count": len(results),
        "factorial_design": {
            "train_domain_counts": sorted({r["configuration"]["train_domains"] for r in results}),
            "payload_dimensions": sorted({r["configuration"]["payload_dim"] for r in results}),
            "seeds": sorted({r["configuration"]["seed"] for r in results}),
            "fixed_training_record_budget": results[0]["configuration"]["train_record_budget"] if results else None,
            "fixed_seen_and_unseen_eval_store_counts": [results[0]["configuration"]["seen_eval_stores"], results[0]["configuration"]["unseen_eval_stores"]] if results else None,
        },
        "cells": cells,
        "interpretation_guardrails": [
            "This is a synthetic projection-domain ablation, not natural-language or arbitrary semantic generalization.",
            "The 32-dimensional learned payload removes the dimensionality bottleneck but is not an identity/raw-value oracle.",
            "Compare reconstruction quality and complete storage jointly; payload dimension changes storage cost.",
            "Three seeds provide an initial replication estimate, not a definitive statistical conclusion.",
        ],
    }


def run_sweep(args):
    seeds = args.seeds or [45, 46, 47]
    domains = args.domain_counts or [2, 4, 8, 16, 32]
    payloads = args.payload_dims or [8, 16, 24, 32]
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(seeds) * len(domains) * len(payloads)
    cell = 0
    for seed, domain_count, payload_dim in itertools.product(seeds, domains, payloads):
        cell += 1
        cell_args = argparse.Namespace(**vars(args))
        cell_args.seed, cell_args.train_domains = seed, domain_count
        # Give each seed an independent family of projection domains while
        # keeping domain IDs nested and comparable within that seed.
        cell_args.domain_seed_base = args.domain_seed_base + seed * 100000
        cell_args.payload_dim = payload_dim
        cell_args.output = root / f"seed_{seed}_domains_{domain_count}_payload_{payload_dim}"
        if args.train_record_budget % domain_count:
            raise ValueError(f"train-record-budget {args.train_record_budget} not divisible by {domain_count}")
        print(f"\n=== Cell {cell}/{total}: seed={seed}, domains={domain_count}, payload={payload_dim} ===", flush=True)
        results.append(run_cell(cell_args))
        (root / "partial_results.json").write_text(json.dumps(results, indent=2))
    summary = aggregate(results)
    (root / "e45_phase3C_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.phase in ("train", "eval"):
        device = torch.device(args.device)
        if args.phase == "train":
            write_phase(args, device)
        else:
            print(json.dumps(evaluate(args, device), indent=2))
        return
    if args.smoke:
        args.seeds, args.domain_counts, args.payload_dims = [args.seed], [2], [8, 32]
    run_sweep(args)


if __name__ == "__main__":
    main()
