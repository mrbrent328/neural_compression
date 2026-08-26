#!/usr/bin/env python3
"""
WHOLE-MODEL TEST 1 — GLOBAL LATENT WHOLE-MODEL RECONSTRUCTION

Question
--------
Can one tiny global latent represent an entire learned model well enough
for a learned constructor to reconstruct its weight tensors?

This is intentionally a baseline:
    full model -> structured tensor encoder -> GLOBAL latent -> tensor decoder
    -> reconstructed full model

No held-out component task and no behavioral optimization are included yet.

The model is represented as a collection of tensor chunks so we do not build
one enormous fully-connected layer over all model parameters.

Default:
    model = ./Qwen2-0.5B
    rank  = 48
    latent = 2
    seed = 9

Important:
- The checkpoint is loaded locally.
- Original weights are never modified.
- The global latent is the only bottleneck shared by all tensors.
- Tensor identity and chunk position are explicitly provided to the decoder.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


# Keep the default scope conservative for the first run.
# It includes learned floating-point parameters and excludes buffers such as
# rotary caches / masks. LayerNorm and embedding weights are included.
DEFAULT_EXCLUDE = (
    "inv_freq",
    "rotary_emb",
    "position_ids",
    "attention_mask",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("./Qwen2-0.5B"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--rank", type=int, default=48)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--output-dir", type=Path, default=Path("./whole_model_results"))
    p.add_argument("--max-tensors", type=int, default=0,
                   help="0 = all tensors; useful for a smoke test.")
    return p.parse_args()


def load_state(model_dir: Path):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


def tensor_manifest(model, max_tensors=0):
    items = []
    for name, tensor in model.named_parameters():
        if not tensor.is_floating_point():
            continue
        if any(x in name for x in DEFAULT_EXCLUDE):
            continue
        if max_tensors and len(items) >= max_tensors:
            break
        items.append((name, tuple(tensor.shape), tensor.numel()))
    return items


def make_dataset(model, chunk_size, max_tensors=0):
    """
    Convert every selected tensor into normalized chunks.

    Each sample consists of:
        normalized parameter values
        tensor identity embedding index
        normalized chunk position
    """
    tensors = []
    for name, tensor in model.named_parameters():
        if not tensor.is_floating_point():
            continue
        if any(x in name for x in DEFAULT_EXCLUDE):
            continue
        if max_tensors and len(tensors) >= max_tensors:
            break

        x = tensor.detach().float().cpu().reshape(-1)
        mean = x.mean()
        std = x.std().clamp_min(1e-8)
        xn = (x - mean) / std

        n_chunks = math.ceil(xn.numel() / chunk_size)
        for c in range(n_chunks):
            lo = c * chunk_size
            hi = min((c + 1) * chunk_size, xn.numel())
            chunk = xn[lo:hi]

            padded = torch.zeros(chunk_size)
            padded[: chunk.numel()] = chunk

            position = 0.0 if n_chunks == 1 else c / (n_chunks - 1)

            tensors.append(
                {
                    "name": name,
                    "tensor_id": len(tensors),  # replaced below with tensor index
                    "chunk": padded,
                    "length": chunk.numel(),
                    "position": position,
                    "mean": mean,
                    "std": std,
                    "flat_start": lo,
                    "flat_end": hi,
                    "shape": tuple(tensor.shape),
                }
            )

    # Reassign compact tensor IDs.
    name_to_id = {}
    for item in tensors:
        if item["name"] not in name_to_id:
            name_to_id[item["name"]] = len(name_to_id)
        item["tensor_id"] = name_to_id[item["name"]]

    return tensors, name_to_id


class WholeModelConstructor(nn.Module):
    """
    Global latent -> tensor/chunk reconstruction.

    The decoder receives:
      global latent
      tensor identity
      chunk position

    This means the latent is global, while the decoder knows where each
    reconstructed piece belongs.
    """

    def __init__(self, latent_dim, num_tensors, chunk_size, hidden):
        super().__init__()
        self.latent_dim = latent_dim
        self.tensor_embedding = nn.Embedding(num_tensors, hidden // 2)

        self.position_net = nn.Sequential(
            nn.Linear(1, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, hidden // 4),
        )

        self.latent_net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden + hidden // 2 + hidden // 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, chunk_size),
        )

    def forward(self, z, tensor_ids, positions):
        zf = self.latent_net(z).expand(tensor_ids.shape[0], -1)
        tf = self.tensor_embedding(tensor_ids)
        pf = self.position_net(positions[:, None])
        h = torch.cat([zf, tf, pf], dim=-1)
        return self.decoder(h)


def reconstruct_loss(model, samples, device):
    chunks = torch.stack([x["chunk"] for x in samples]).to(device)
    ids = torch.tensor([x["tensor_id"] for x in samples], device=device)
    pos = torch.tensor([x["position"] for x in samples], device=device)

    z = model.global_latent.expand(chunks.shape[0], -1)
    pred = model.constructor(z, ids, pos)
    return F.mse_loss(pred, chunks)


class WholeModelSystem(nn.Module):
    def __init__(self, latent_dim, num_tensors, chunk_size, hidden):
        super().__init__()
        self.global_latent = nn.Parameter(torch.zeros(1, latent_dim))
        self.constructor = WholeModelConstructor(
            latent_dim, num_tensors, chunk_size, hidden
        )


@torch.no_grad()
def evaluate(system, samples, device):
    system.eval()

    total_sq = 0.0
    total_n = 0

    per_tensor = {}

    for item in samples:
        chunk = item["chunk"].to(device)[None, :]
        tid = torch.tensor([item["tensor_id"]], device=device)
        pos = torch.tensor([[item["position"]]], device=device)

        z = system.global_latent
        pred = system.constructor(z, tid, pos)

        n = item["length"]
        diff = pred[0, :n] - chunk[0, :n]

        total_sq += float((diff * diff).sum())
        total_n += n

        name = item["name"]
        entry = per_tensor.setdefault(name, [0.0, 0])
        entry[0] += float((diff * diff).sum())
        entry[1] += n

    result = {
        "normalized_global_rmse": math.sqrt(total_sq / max(total_n, 1)),
        "global_latent": system.global_latent.detach().cpu().flatten().tolist(),
        "per_tensor": {
            k: math.sqrt(v[0] / max(v[1], 1))
            for k, v in per_tensor.items()
        },
    }
    return result


def main():
    a = parse_args()
    torch.manual_seed(a.seed)

    device = torch.device(a.device)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("WHOLE-MODEL TEST 1 — GLOBAL LATENT RECONSTRUCTION")
    print("=" * 88)
    print(f"Model        : {a.model_dir}")
    print(f"Device       : {device}")
    print(f"Rank         : {a.rank} (reserved for follow-up structured factor test)")
    print(f"Latent       : {a.latent_dim}")
    print(f"Chunk size   : {a.chunk_size}")
    print(f"Hidden       : {a.hidden}")
    print(f"Steps        : {a.steps}")
    print(f"Seed         : {a.seed}")

    print("\n[1] Loading complete model")
    reference = load_state(a.model_dir)

    manifest = tensor_manifest(reference, a.max_tensors)
    print(f"  tensors selected : {len(manifest):,}")
    print(f"  parameters       : {sum(x[2] for x in manifest):,}")

    print("\n[2] Building whole-model chunk dataset")
    samples, name_to_id = make_dataset(
        reference,
        a.chunk_size,
        a.max_tensors,
    )
    print(f"  chunks           : {len(samples):,}")
    print(f"  tensor identities: {len(name_to_id):,}")

    del reference

    system = WholeModelSystem(
        a.latent_dim,
        len(name_to_id),
        a.chunk_size,
        a.hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=a.lr,
        weight_decay=1e-5,
    )

    print("\n[3] Training global constructor")
    for step in range(1, a.steps + 1):
        system.train()

        optimizer.zero_grad(set_to_none=True)

        # Full-batch training is intentional for the first baseline.
        chunks = torch.stack([x["chunk"] for x in samples]).to(device)
        ids = torch.tensor([x["tensor_id"] for x in samples], device=device)
        pos = torch.tensor([x["position"] for x in samples], device=device)

        z = system.global_latent.expand(chunks.shape[0], -1)
        pred = system.constructor(z, ids, pos)

        loss = F.mse_loss(pred, chunks)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == a.steps:
            print(f"  step {step:4d}/{a.steps} | normalized MSE {loss.item():.8g}")

    print("\n[4] Reconstruction")
    result = evaluate(system, samples, device)

    print(f"  global latent = {result['global_latent']}")
    print(f"  normalized global RMSE = {result['normalized_global_rmse']:.8f}")

    print("\n  Per-tensor RMSE:")
    for name, value in result["per_tensor"].items():
        print(f"    {name:70s} {value:.8f}")

    output = {
        "experiment": "whole_model_test_1_global_latent",
        "model_dir": str(a.model_dir),
        "latent_dim": a.latent_dim,
        "rank": a.rank,
        "chunk_size": a.chunk_size,
        "hidden": a.hidden,
        "steps": a.steps,
        "lr": a.lr,
        "seed": a.seed,
        "num_tensors": len(name_to_id),
        "num_chunks": len(samples),
        "result": result,
    }

    path = a.output_dir / f"whole_model_latent{a.latent_dim}_seed{a.seed}.json"
    path.write_text(json.dumps(output, indent=2))
    print(f"\nReport written to: {path}")


if __name__ == "__main__":
    main()
