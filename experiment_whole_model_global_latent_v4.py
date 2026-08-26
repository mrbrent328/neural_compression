#!/usr/bin/env python3
"""
WHOLE-MODEL TEST 2 — CLEAN GLOBAL KEY

Purpose
-------
Test whether a tiny GLOBAL latent can act as the key selecting a complete
model from a shared constructor.

Important correction from v3
----------------------------
v3 used a learned tensor-ID embedding. That gave the constructor a separate
learned representation for every tensor and therefore allowed it to bypass
the global latent.

v4 removes that shortcut.

The constructor receives ONLY:
    - global latent
    - structural metadata:
        * normalized layer position
        * tensor/module category
        * weight/bias flag
        * normalized tensor dimensions
        * chunk position

There is NO learned embedding for individual tensor identity.

The global latent is therefore the only learned low-dimensional representation
that can identify the particular model instance.

Architecture
------------
                         GLOBAL LATENT [x,y]
                                  |
                                  v
                         shared constructor
                                  |
             +--------------------+--------------------+
             |                    |                    |
          structure            position             shape
             |                    |                    |
             +--------------------+--------------------+
                                  |
                                  v
                         reconstructed chunk

Training
--------
Every epoch sees ALL chunks through minibatches.
Gradients are accumulated across minibatches.
One optimizer update occurs per epoch.

This keeps the whole-model experiment while avoiding the giant full-batch
memory requirement.

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
    p.add_argument("--save-system", action="store_true",
                   help="Save constructor + latent checkpoint for later reconstruction.")
    return p.parse_args()


# Tensor categories are STRUCTURAL, not individual identities.
# The category embedding is shared by all tensors of that type.
CATEGORY_NAMES = [
    "embedding",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "attention_bias",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "layernorm",
    "output_norm",
    "other",
]
CATEGORY_TO_ID = {name: i for i, name in enumerate(CATEGORY_NAMES)}


def classify_tensor(name: str):
    if "embed_tokens" in name:
        return "embedding", False
    if ".self_attn.q_proj.weight" in name:
        return "attention_q", False
    if ".self_attn.k_proj.weight" in name:
        return "attention_k", False
    if ".self_attn.v_proj.weight" in name:
        return "attention_v", False
    if ".self_attn.o_proj.weight" in name:
        return "attention_o", False
    if ".self_attn." in name and name.endswith(".bias"):
        return "attention_bias", True
    if ".mlp.gate_proj.weight" in name:
        return "mlp_gate", False
    if ".mlp.up_proj.weight" in name:
        return "mlp_up", False
    if ".mlp.down_proj.weight" in name:
        return "mlp_down", False
    if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
        return "layernorm", False
    if name == "model.norm.weight":
        return "output_norm", False
    if name.endswith(".bias"):
        return "attention_bias", True
    return "other", name.endswith(".bias")


def infer_layer_position(name: str, num_layers: int):
    marker = "model.layers."
    if marker in name:
        rest = name.split(marker, 1)[1]
        try:
            layer = int(rest.split(".", 1)[0])
            return 0.0 if num_layers <= 1 else layer / (num_layers - 1)
        except ValueError:
            pass

    # Embedding before the stack; output norm after it.
    if "embed_tokens" in name:
        return 0.0
    if name == "model.norm.weight":
        return 1.0
    return 0.5


def load_reference(model_dir: Path):
    # Keep source checkpoint on CPU. Only batches enter the accelerator.
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
    )


def build_chunk_dataset(model, chunk_size, max_tensors=0):
    """
    Build normalized whole-model chunks plus structural metadata.

    No feature here uniquely identifies an individual tensor.

    Returns:
        chunks        [N, chunk_size]
        categories    [N]
        positions     [N, 1]
        layer_pos     [N, 1]
        weight_bias   [N, 1]
        shape_features[N, 3]
        metadata      list[dict]
    """
    # Infer transformer depth from parameter names.
    layer_ids = []
    for name, tensor in model.named_parameters():
        if "model.layers." in name:
            try:
                layer_ids.append(int(name.split("model.layers.", 1)[1].split(".", 1)[0]))
            except ValueError:
                pass
    num_layers = max(layer_ids) + 1 if layer_ids else 1

    chunks = []
    categories = []
    positions = []
    layer_positions = []
    weight_bias = []
    shape_features = []
    metadata = []

    selected = []
    for name, tensor in model.named_parameters():
        if not tensor.is_floating_point():
            continue
        selected.append((name, tensor.detach().float().cpu().reshape(-1), tuple(tensor.shape)))
        if max_tensors and len(selected) >= max_tensors:
            break

    for name, flat, shape in selected:
        category, is_bias = classify_tensor(name)
        category_id = CATEGORY_TO_ID[category]
        layer_pos = infer_layer_position(name, num_layers)

        mean = float(flat.mean())
        std = float(flat.std().clamp_min(1e-8))
        normalized = (flat - mean) / std

        n_chunks = math.ceil(normalized.numel() / chunk_size)

        # Shape metadata describes the tensor form, not its identity.
        # Use log-scaled dimensions normalized into a modest range.
        dims = list(shape[:3])
        dims += [1] * (3 - len(dims))
        shape_feat = [
            math.log2(max(int(dims[0]), 1)) / 16.0,
            math.log2(max(int(dims[1]), 1)) / 16.0,
            math.log2(max(int(dims[2]), 1)) / 16.0,
        ]

        for c in range(n_chunks):
            lo = c * chunk_size
            hi = min((c + 1) * chunk_size, normalized.numel())
            n = hi - lo

            padded = torch.zeros(chunk_size, dtype=torch.float32)
            padded[:n] = normalized[lo:hi]

            chunk_pos = 0.0 if n_chunks == 1 else c / (n_chunks - 1)

            chunks.append(padded)
            categories.append(category_id)
            positions.append(chunk_pos)
            layer_positions.append(layer_pos)
            weight_bias.append(1.0 if is_bias else 0.0)
            shape_features.append(shape_feat)

            metadata.append({
                "name": name,
                "shape": list(shape),
                "category": category,
                "category_id": category_id,
                "layer_position": layer_pos,
                "chunk_index": c,
                "num_chunks": n_chunks,
                "length": n,
                "flat_start": lo,
                "flat_end": hi,
                "mean": mean,
                "std": std,
            })

    return (
        torch.stack(chunks),
        torch.tensor(categories, dtype=torch.long),
        torch.tensor(positions, dtype=torch.float32).unsqueeze(1),
        torch.tensor(layer_positions, dtype=torch.float32).unsqueeze(1),
        torch.tensor(weight_bias, dtype=torch.float32).unsqueeze(1),
        torch.tensor(shape_features, dtype=torch.float32),
        metadata,
    )


class WholeModelConstructor(nn.Module):
    """
    Shared constructor with NO tensor-ID embedding.

    Structural category embedding is shared by category, not tensor identity.
    """

    def __init__(self, latent_dim, num_categories, chunk_size, hidden):
        super().__init__()

        self.category_embedding = nn.Embedding(num_categories, hidden // 4)

        self.position_net = nn.Sequential(
            nn.Linear(1, hidden // 8),
            nn.GELU(),
            nn.Linear(hidden // 8, hidden // 8),
        )

        self.layer_net = nn.Sequential(
            nn.Linear(1, hidden // 8),
            nn.GELU(),
            nn.Linear(hidden // 8, hidden // 8),
        )

        self.structural_net = nn.Sequential(
            nn.Linear(4, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, hidden // 4),
        )

        self.latent_net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        in_dim = hidden + hidden // 4 + hidden // 8 + hidden // 8 + hidden // 4

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, chunk_size),
        )

    def forward(
        self,
        z,
        categories,
        positions,
        layer_positions,
        weight_bias,
        shape_features,
    ):
        zf = self.latent_net(z).expand(categories.shape[0], -1)
        cf = self.category_embedding(categories)
        pf = self.position_net(positions)
        lf = self.layer_net(layer_positions)

        structural = torch.cat(
            [weight_bias, shape_features],
            dim=-1,
        )
        sf = self.structural_net(structural)

        h = torch.cat([zf, cf, pf, lf, sf], dim=-1)
        return self.decoder(h)


class WholeModelSystem(nn.Module):
    def __init__(self, latent_dim, chunk_size, hidden):
        super().__init__()

        # Start at zero exactly as v3 did.
        self.global_latent = nn.Parameter(torch.zeros(1, latent_dim))

        self.constructor = WholeModelConstructor(
            latent_dim=latent_dim,
            num_categories=len(CATEGORY_NAMES),
            chunk_size=chunk_size,
            hidden=hidden,
        )


def run_epoch(
    system,
    chunks,
    categories,
    positions,
    layer_positions,
    weight_bias,
    shape_features,
    device,
    batch_size,
    optimizer,
):
    system.train()
    optimizer.zero_grad(set_to_none=True)

    n = chunks.shape[0]
    order = torch.randperm(n)
    total_loss = 0.0

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]

        x = chunks[idx].to(device, non_blocking=True)
        cat = categories[idx].to(device, non_blocking=True)
        pos = positions[idx].to(device, non_blocking=True)
        layer = layer_positions[idx].to(device, non_blocking=True)
        wb = weight_bias[idx].to(device, non_blocking=True)
        shape = shape_features[idx].to(device, non_blocking=True)

        z = system.global_latent.expand(x.shape[0], -1)

        pred = system.constructor(
            z, cat, pos, layer, wb, shape
        )

        batch_loss = F.mse_loss(pred, x)

        # Exact mean-dataset gradient across the minibatches.
        weight = x.shape[0] / n
        (batch_loss * weight).backward()

        total_loss += float(batch_loss.detach()) * weight

    optimizer.step()
    return total_loss


@torch.no_grad()
def evaluate(
    system,
    chunks,
    categories,
    positions,
    layer_positions,
    weight_bias,
    shape_features,
    metadata,
    device,
    batch_size,
):
    system.eval()

    n = chunks.shape[0]
    total_sq = 0.0
    total_n = 0

    per_tensor_sq = {}
    per_tensor_n = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        x = chunks[start:end].to(device, non_blocking=True)
        cat = categories[start:end].to(device, non_blocking=True)
        pos = positions[start:end].to(device, non_blocking=True)
        layer = layer_positions[start:end].to(device, non_blocking=True)
        wb = weight_bias[start:end].to(device, non_blocking=True)
        shape = shape_features[start:end].to(device, non_blocking=True)

        z = system.global_latent.expand(x.shape[0], -1)

        pred = system.constructor(
            z, cat, pos, layer, wb, shape
        )

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


def save_checkpoint(system, args, output_dir):
    path = output_dir / f"whole_model_global_key_seed{args.seed}_v4.pt"

    torch.save(
        {
            "state_dict": system.state_dict(),
            "latent_dim": args.latent_dim,
            "chunk_size": args.chunk_size,
            "hidden": args.hidden,
            "category_names": CATEGORY_NAMES,
            "seed": args.seed,
        },
        path,
    )
    return path


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("WHOLE-MODEL TEST 2 — CLEAN GLOBAL KEY")
    print("=" * 88)
    print(f"Model        : {args.model_dir}")
    print(f"Device       : {device}")
    print(f"Rank         : {args.rank} (reserved for follow-up structured-factor test)")
    print(f"Latent       : {args.latent_dim}")
    print(f"Chunk size   : {args.chunk_size}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Hidden       : {args.hidden}")
    print(f"Epochs       : {args.epochs}")
    print(f"Seed         : {args.seed}")
    print()
    print("IMPORTANT: NO learned per-tensor ID embedding.")
    print("Only shared structural metadata + the global latent are provided.")

    print("\n[1] Loading complete model")
    reference = load_reference(args.model_dir)

    selected = [
        (name, t.numel())
        for name, t in reference.named_parameters()
        if t.is_floating_point()
    ]
    if args.max_tensors:
        selected = selected[:args.max_tensors]

    print(f"  tensors selected : {len(selected):,}")
    print(f"  parameters       : {sum(n for _, n in selected):,}")

    print("\n[2] Building whole-model chunk dataset")
    (
        chunks,
        categories,
        positions,
        layer_positions,
        weight_bias,
        shape_features,
        metadata,
    ) = build_chunk_dataset(
        reference,
        args.chunk_size,
        args.max_tensors,
    )

    print(f"  chunks           : {chunks.shape[0]:,}")
    print(f"  structural cats  : {len(CATEGORY_NAMES)}")
    print(f"  CPU chunk memory : {chunks.numel() * 4 / (1024**3):.2f} GiB")

    del reference

    print("\n[3] Building clean global-key constructor")
    system = WholeModelSystem(
        latent_dim=args.latent_dim,
        chunk_size=args.chunk_size,
        hidden=args.hidden,
    ).to(device)

    print(f"  trainable params : {sum(p.numel() for p in system.parameters()):,}")
    print(f"  initial latent   : {system.global_latent.detach().cpu().flatten().tolist()}")

    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    print("\n[4] Training")
    print("  Every epoch sees ALL chunks.")
    print("  Gradients accumulate over minibatches.")
    print("  One optimizer update occurs per epoch.")

    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(
            system,
            chunks,
            categories,
            positions,
            layer_positions,
            weight_bias,
            shape_features,
            device,
            args.batch_size,
            optimizer,
        )

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            latent = system.global_latent.detach().cpu().flatten().tolist()
            print(
                f"  epoch {epoch:4d}/{args.epochs} | "
                f"normalized MSE {loss:.8f} | "
                f"latent={latent}"
            )

    print("\n[5] Final evaluation")
    result = evaluate(
        system,
        chunks,
        categories,
        positions,
        layer_positions,
        weight_bias,
        shape_features,
        metadata,
        device,
        args.batch_size,
    )

    print(f"  normalized global RMSE = {result['normalized_global_rmse']:.8f}")
    print(f"  global latent          = {result['global_latent']}")

    print("\n  Per-tensor RMSE:")
    for name, value in result["per_tensor_rmse"].items():
        print(f"    {name:70s} {value:.8f}")

    report = {
        "experiment": "whole_model_test_2_clean_global_key_v4",
        "model_dir": str(args.model_dir),
        "latent_dim": args.latent_dim,
        "rank": args.rank,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "num_tensors": len(selected),
        "num_chunks": int(chunks.shape[0]),
        "category_names": CATEGORY_NAMES,
        "no_tensor_id_embedding": True,
        "result": result,
    }

    report_path = args.output_dir / f"whole_model_clean_global_key_seed{args.seed}_v4.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nReport written to: {report_path}")

    if args.save_system:
        checkpoint = save_checkpoint(system, args, args.output_dir)
        print(f"System checkpoint written to: {checkpoint}")


if __name__ == "__main__":
    main()
