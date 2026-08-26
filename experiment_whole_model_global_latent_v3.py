#!/usr/bin/env python3
"""
WHOLE-MODEL TEST 1 v3 — BATCHED GLOBAL LATENT RECONSTRUCTION

Same scientific experiment as v1/v2, but implemented for practical GPU use.

Question
--------
Can one tiny GLOBAL latent represent an entire learned model well enough
for a learned constructor to reconstruct all of its weight tensors?

Architecture
------------
    FULL MODEL
        |
        v
    tensor/chunk dataset
        |
        v
    ONE GLOBAL LATENT
        |
        + tensor identity
        + chunk position
        |
        v
    shared constructor
        |
        v
    reconstructed tensor chunks

Important:
- The latent is global. There is no per-tensor latent.
- Tensor identity and chunk position are metadata supplied to the decoder;
  they are NOT information stored in separate local latents.
- The model checkpoint is only used as the source of the target weights.
- No behavioral optimization is performed.
- Training uses mini-batches on the selected device, while accumulating the
  gradient across all chunks before one optimizer update per epoch. Thus an
  epoch still sees ALL model chunks, but GPU memory only holds one batch.

This preserves the spirit of the original full-batch experiment while avoiding
the enormous single-batch CUDA/CPU allocation.

Default:
    model      = ./Qwen2-0.5B
    latent     = 2
    chunk      = 1024
    batch      = 512
    epochs     = 2000
    device     = cuda if available
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
    p.add_argument("--rank", type=int, default=48,
                   help="Reserved for the follow-up structured-factor experiment.")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--output-dir", type=Path, default=Path("./whole_model_results"))
    p.add_argument("--max-tensors", type=int, default=0,
                   help="0 = all tensors; useful for a smoke test.")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def load_reference(model_dir: Path):
    # CPU loading keeps the checkpoint itself off the 8 GB GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    return model


def build_chunk_dataset(model, chunk_size, max_tensors=0):
    """
    Materialize the normalized whole-model chunks into compact CPU tensors.

    Returns:
      chunks:      [N, chunk_size] float32
      tensor_ids:  [N] int64
      positions:   [N, 1] float32
      metadata:    JSON-serializable tensor/chunk manifest
    """
    chunks = []
    tensor_ids = []
    positions = []
    metadata = []

    name_to_id = {}

    selected = []
    for name, tensor in model.named_parameters():
        if not tensor.is_floating_point():
            continue
        if any(x in name for x in DEFAULT_EXCLUDE):
            continue
        if max_tensors and len(selected) >= max_tensors:
            break
        selected.append((name, tensor.detach().float().cpu().reshape(-1),
                         tuple(tensor.shape)))

    for name, flat, shape in selected:
        tid = name_to_id.setdefault(name, len(name_to_id))

        mean = float(flat.mean())
        std = float(flat.std().clamp_min(1e-8))
        normalized = (flat - mean) / std

        n_chunks = math.ceil(normalized.numel() / chunk_size)

        for c in range(n_chunks):
            lo = c * chunk_size
            hi = min((c + 1) * chunk_size, normalized.numel())
            n = hi - lo

            padded = torch.zeros(chunk_size, dtype=torch.float32)
            padded[:n] = normalized[lo:hi]

            pos = 0.0 if n_chunks == 1 else c / (n_chunks - 1)

            chunks.append(padded)
            tensor_ids.append(tid)
            positions.append(pos)

            metadata.append({
                "name": name,
                "shape": list(shape),
                "tensor_id": tid,
                "chunk_index": c,
                "num_chunks": n_chunks,
                "length": n,
                "flat_start": lo,
                "flat_end": hi,
                "mean": mean,
                "std": std,
            })

    chunks = torch.stack(chunks)
    tensor_ids = torch.tensor(tensor_ids, dtype=torch.long)
    positions = torch.tensor(positions, dtype=torch.float32).unsqueeze(1)

    return chunks, tensor_ids, positions, metadata, name_to_id


class WholeModelConstructor(nn.Module):
    def __init__(self, latent_dim, num_tensors, chunk_size, hidden):
        super().__init__()

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
        pf = self.position_net(positions)

        h = torch.cat([zf, tf, pf], dim=-1)
        return self.decoder(h)


class WholeModelSystem(nn.Module):
    def __init__(self, latent_dim, num_tensors, chunk_size, hidden):
        super().__init__()
        self.global_latent = nn.Parameter(torch.zeros(1, latent_dim))
        self.constructor = WholeModelConstructor(
            latent_dim=latent_dim,
            num_tensors=num_tensors,
            chunk_size=chunk_size,
            hidden=hidden,
        )


def run_epoch(system, chunks, tensor_ids, positions, device, batch_size, optimizer):
    """
    One full pass over ALL chunks.

    We accumulate the exact mean loss gradient across mini-batches and perform
    ONE optimizer update at the end of the epoch. This avoids changing the
    experiment into a much larger number of optimizer steps.
    """
    system.train()
    optimizer.zero_grad(set_to_none=True)

    n = chunks.shape[0]
    order = torch.randperm(n)

    total_loss_value = 0.0

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]

        x = chunks[idx].to(device, non_blocking=True)
        ids = tensor_ids[idx].to(device, non_blocking=True)
        pos = positions[idx].to(device, non_blocking=True)

        z = system.global_latent.expand(x.shape[0], -1)
        pred = system.constructor(z, ids, pos)

        # Weight by batch fraction so the accumulated gradient is the
        # gradient of the full-dataset mean loss.
        batch_loss = F.mse_loss(pred, x)
        weight = x.shape[0] / n
        (batch_loss * weight).backward()

        total_loss_value += float(batch_loss.detach()) * weight

    optimizer.step()
    return total_loss_value


@torch.no_grad()
def evaluate(system, chunks, tensor_ids, positions, metadata, device, batch_size):
    system.eval()

    n = chunks.shape[0]
    total_sq = 0.0
    total_n = 0

    per_tensor_sq = {}
    per_tensor_n = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        x = chunks[start:end].to(device, non_blocking=True)
        ids = tensor_ids[start:end].to(device, non_blocking=True)
        pos = positions[start:end].to(device, non_blocking=True)

        z = system.global_latent.expand(x.shape[0], -1)
        pred = system.constructor(z, ids, pos)

        for j in range(end - start):
            meta = metadata[start + j]
            length = meta["length"]

            diff = pred[j, :length] - x[j, :length]
            sq = float((diff * diff).sum())

            total_sq += sq
            total_n += length

            name = meta["name"]
            per_tensor_sq[name] = per_tensor_sq.get(name, 0.0) + sq
            per_tensor_n[name] = per_tensor_n.get(name, 0) + length

    return {
        "normalized_global_rmse": math.sqrt(total_sq / max(total_n, 1)),
        "global_latent": system.global_latent.detach().cpu().flatten().tolist(),
        "per_tensor_rmse": {
            name: math.sqrt(per_tensor_sq[name] / max(per_tensor_n[name], 1))
            for name in per_tensor_sq
        },
    }


def main():
    a = parse_args()
    torch.manual_seed(a.seed)

    device = torch.device(a.device)
    a.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("WHOLE-MODEL TEST 1 v3 — BATCHED GLOBAL LATENT RECONSTRUCTION")
    print("=" * 88)
    print(f"Model        : {a.model_dir}")
    print(f"Device       : {device}")
    print(f"Rank         : {a.rank} (reserved for follow-up structured factor test)")
    print(f"Latent       : {a.latent_dim}")
    print(f"Chunk size   : {a.chunk_size}")
    print(f"Batch size   : {a.batch_size}")
    print(f"Hidden       : {a.hidden}")
    print(f"Epochs       : {a.epochs}")
    print(f"Seed         : {a.seed}")

    print("\n[1] Loading complete model")
    reference = load_reference(a.model_dir)

    selected = [
        (name, tuple(t.shape), t.numel())
        for name, t in reference.named_parameters()
        if t.is_floating_point()
        and not any(x in name for x in DEFAULT_EXCLUDE)
    ]
    if a.max_tensors:
        selected = selected[:a.max_tensors]

    print(f"  tensors selected : {len(selected):,}")
    print(f"  parameters       : {sum(x[2] for x in selected):,}")

    print("\n[2] Building whole-model chunk dataset")
    chunks, tensor_ids, positions, metadata, name_to_id = build_chunk_dataset(
        reference,
        a.chunk_size,
        a.max_tensors,
    )

    print(f"  chunks           : {chunks.shape[0]:,}")
    print(f"  tensor identities: {len(name_to_id):,}")
    print(f"  CPU chunk memory : {chunks.numel() * 4 / (1024**3):.2f} GiB")

    # The checkpoint is no longer needed.
    del reference

    system = WholeModelSystem(
        latent_dim=a.latent_dim,
        num_tensors=len(name_to_id),
        chunk_size=a.chunk_size,
        hidden=a.hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=a.lr,
        weight_decay=1e-5,
    )

    print("\n[3] Training global constructor")
    print("  Every epoch sees ALL chunks; gradients accumulate over mini-batches.")
    print("  One optimizer update is performed per epoch.")

    for epoch in range(1, a.epochs + 1):
        loss = run_epoch(
            system,
            chunks,
            tensor_ids,
            positions,
            device,
            a.batch_size,
            optimizer,
        )

        if epoch == 1 or epoch % a.log_every == 0 or epoch == a.epochs:
            print(f"  epoch {epoch:4d}/{a.epochs} | normalized MSE {loss:.8g}")

    print("\n[4] Reconstruction")
    result = evaluate(
        system,
        chunks,
        tensor_ids,
        positions,
        metadata,
        device,
        a.batch_size,
    )

    print(f"  global latent = {result['global_latent']}")
    print(f"  normalized global RMSE = {result['normalized_global_rmse']:.8f}")

    print("\n  Per-tensor RMSE:")
    for name, value in result["per_tensor_rmse"].items():
        print(f"    {name:70s} {value:.8f}")

    report = {
        "experiment": "whole_model_test_1_global_latent_v3",
        "model_dir": str(a.model_dir),
        "latent_dim": a.latent_dim,
        "rank": a.rank,
        "chunk_size": a.chunk_size,
        "batch_size": a.batch_size,
        "hidden": a.hidden,
        "epochs": a.epochs,
        "lr": a.lr,
        "seed": a.seed,
        "num_tensors": len(name_to_id),
        "num_chunks": int(chunks.shape[0]),
        "result": result,
    }

    out = a.output_dir / f"whole_model_latent{a.latent_dim}_seed{a.seed}_v3.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
