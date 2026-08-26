#!/usr/bin/env python3
"""
WHOLE-MODEL TYPE SPECIALISTS — V1

Question
--------
Can we train the whole model as several independent *tensor-family*
representations, then assemble all of the reconstructed families into one
reconstructed Qwen model?

This is the generalized version of the successful LayerNorm experiment.

Instead of:
    ONE global latent -> ONE constructor -> ALL tensor types

we do:
    ONE 2D latent + ONE constructor PER TENSOR TYPE

and then reassemble every reconstructed tensor into one model.

Important control
-----------------
Every specialist uses:
    latent_dim = 2
    hidden     = 512
    rank       = 48
    chunk      = 1024
    lr         = 2e-4
    seed       = 9

There is NO per-tensor learned ID embedding.

Within a type, the constructor receives only structural metadata:
    - normalized layer position
    - chunk position
    - weight/bias flag
    - tensor shape features

Thus tensors of the same family share one learned 2D representation.

Families
--------
    embedding
    attention_q
    attention_k
    attention_v
    attention_o
    attention_bias
    mlp_gate
    mlp_up
    mlp_down
    layernorm
    output_norm
    other

The LayerNorm family intentionally combines BOTH:
    input_layernorm.weight
    post_attention_layernorm.weight

This is the experiment requested: train all LayerNorms together as ONE family,
while every other structural kind gets its own specialist too.

Outputs
-------
    whole_model_type_specialists_v1.pt
    whole_model_type_specialists_v1.json

The checkpoint contains every specialist constructor and latent, plus the
manifest needed to reconstruct the complete model later.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


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

CATEGORY_TO_ID = {
    name: i for i, name in enumerate(CATEGORY_NAMES)
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./Qwen2-0.5B"),
    )

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--rank", type=int, default=48)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=9)

    p.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="0 = all tensors. Useful for a smoke test.",
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./whole_model_type_specialists_results"),
    )

    return p.parse_args()


def classify_tensor(name: str):
    """
    Structural classification only.

    No individual tensor identity is encoded here.
    """

    lower = name.lower()

    if lower == "model.embed_tokens.weight":
        return "embedding", False

    if lower.endswith("self_attn.q_proj.weight"):
        return "attention_q", False

    if lower.endswith("self_attn.k_proj.weight"):
        return "attention_k", False

    if lower.endswith("self_attn.v_proj.weight"):
        return "attention_v", False

    if lower.endswith("self_attn.o_proj.weight"):
        return "attention_o", False

    if ".self_attn." in lower and lower.endswith(".bias"):
        return "attention_bias", True

    if lower.endswith("mlp.gate_proj.weight"):
        return "mlp_gate", False

    if lower.endswith("mlp.up_proj.weight"):
        return "mlp_up", False

    if lower.endswith("mlp.down_proj.weight"):
        return "mlp_down", False

    if "input_layernorm." in lower or "post_attention_layernorm." in lower:
        return "layernorm", lower.endswith(".bias")

    if lower == "model.norm.weight" or lower == "model.norm.bias":
        return "output_norm", lower.endswith(".bias")

    if lower.endswith(".bias"):
        return "other", True

    return "other", False


def infer_layer_position(name: str, num_layers: int):
    match = re.search(r"model\.layers\.(\d+)\.", name)

    if match is None:
        return 0.0

    layer = int(match.group(1))

    if num_layers <= 1:
        return 0.0

    return layer / float(num_layers - 1)


def shape_features(shape):
    dims = list(shape[:3])

    while len(dims) < 3:
        dims.append(1)

    return [
        math.log2(max(int(dims[0]), 1)) / 16.0,
        math.log2(max(int(dims[1]), 1)) / 16.0,
        math.log2(max(int(dims[2]), 1)) / 16.0,
    ]


def load_reference(model_dir):
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
    )


def build_dataset(
    model,
    chunk_size,
    max_tensors=0,
):
    """
    Build normalized chunks and structural metadata.

    Returns CPU tensors so the complete model can stay off the GPU.
    """

    layer_ids = []

    for name, tensor in model.named_parameters():
        if "model.layers." in name:
            match = re.search(
                r"model\.layers\.(\d+)\.",
                name,
            )
            if match:
                layer_ids.append(
                    int(match.group(1))
                )

    num_layers = (
        max(layer_ids) + 1
        if layer_ids
        else 1
    )

    selected = []

    for name, tensor in model.named_parameters():

        if not tensor.is_floating_point():
            continue

        if max_tensors and len(selected) >= max_tensors:
            break

        selected.append(
            (
                name,
                tensor.detach()
                .float()
                .cpu()
                .reshape(-1),
                tuple(tensor.shape),
            )
        )

    chunks = []
    categories = []
    positions = []
    layer_positions = []
    weight_bias = []
    shapes = []
    metadata = []

    # Per-tensor normalization is exactly the normalization used in V4.
    for name, flat, shape in selected:

        category, is_bias = classify_tensor(
            name
        )

        category_id = CATEGORY_TO_ID[
            category
        ]

        layer_pos = infer_layer_position(
            name,
            num_layers,
        )

        mean = float(flat.mean())

        std = float(
            flat.std(
                unbiased=False
            ).clamp_min(1e-8)
        )

        normalized = (
            flat - mean
        ) / std

        n_chunks = math.ceil(
            normalized.numel()
            / chunk_size
        )

        shape_feat = shape_features(
            shape
        )

        for c in range(n_chunks):

            lo = c * chunk_size
            hi = min(
                (c + 1) * chunk_size,
                normalized.numel(),
            )

            n = hi - lo

            padded = torch.zeros(
                chunk_size,
                dtype=torch.float32,
            )

            padded[:n] = normalized[
                lo:hi
            ]

            chunk_pos = (
                0.0
                if n_chunks <= 1
                else c / float(n_chunks - 1)
            )

            chunks.append(padded)

            categories.append(
                category_id
            )

            positions.append(
                chunk_pos
            )

            layer_positions.append(
                layer_pos
            )

            weight_bias.append(
                1.0 if is_bias else 0.0
            )

            shapes.append(
                shape_feat
            )

            metadata.append(
                {
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
                }
            )

    return {
        "chunks": torch.stack(
            chunks
        ).float(),

        "categories": torch.tensor(
            categories,
            dtype=torch.long,
        ),

        "positions": torch.tensor(
            positions,
            dtype=torch.float32,
        ).unsqueeze(1),

        "layer_positions": torch.tensor(
            layer_positions,
            dtype=torch.float32,
        ).unsqueeze(1),

        "weight_bias": torch.tensor(
            weight_bias,
            dtype=torch.float32,
        ).unsqueeze(1),

        "shapes": torch.tensor(
            shapes,
            dtype=torch.float32,
        ),

        "metadata": metadata,
    }


class TypeConstructor(nn.Module):
    """
    One shared constructor for one tensor family.

    The family itself has a single global 2D latent.
    """

    def __init__(
        self,
        latent_dim,
        hidden,
        chunk_size,
    ):
        super().__init__()

        metadata_dim = 1 + 1 + 1 + 3

        self.category = nn.Identity()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim
                + metadata_dim,
                hidden,
            ),
            nn.GELU(),

            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),

            nn.Linear(
                hidden,
                chunk_size,
            ),
        )

        self.global_latent = nn.Parameter(
            torch.tensor(
                [-0.0002, 0.0002],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        category_metadata,
        batch_size,
    ):
        latent = (
            self.global_latent
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        x = torch.cat(
            [
                latent,
                category_metadata,
            ],
            dim=1,
        )

        return self.network(x)


def normalized_mse(
    prediction,
    target,
):
    """
    Same normalized objective used by the component experiments.
    """

    mse = (
        prediction - target
    ).pow(2).mean()

    target_mse = (
        target.pow(2)
        .mean()
        .clamp_min(1e-12)
    )

    return mse / target_mse


def train_specialist(
    category_name,
    data,
    args,
):
    print()
    print("=" * 88)
    print(
        f"TRAINING TYPE SPECIALIST: "
        f"{category_name}"
    )
    print("=" * 88)

    mask = (
        data["categories"]
        == CATEGORY_TO_ID[
            category_name
        ]
    )

    indices = torch.nonzero(
        mask,
        as_tuple=False,
    ).flatten()

    if indices.numel() == 0:
        print("  No tensors in this category.")
        return None

    chunks = data["chunks"][indices]
    positions = data["positions"][indices]
    layer_positions = data[
        "layer_positions"
    ][indices]
    weight_bias = data[
        "weight_bias"
    ][indices]
    shapes = data["shapes"][indices]

    metadata = torch.cat(
        [
            positions,
            layer_positions,
            weight_bias,
            shapes,
        ],
        dim=1,
    )

    chunks = chunks.to(
        args.device
    )

    metadata = metadata.to(
        args.device
    )

    system = TypeConstructor(
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        chunk_size=args.chunk_size,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    start_time = time.time()

    n = chunks.shape[0]

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        permutation = torch.randperm(
            n,
            device=args.device,
        )

        running_loss = 0.0
        batches = 0

        for start in range(
            0,
            n,
            args.batch_size,
        ):

            batch_indices = permutation[
                start:
                start + args.batch_size
            ]

            x = chunks[
                batch_indices
            ]

            m = metadata[
                batch_indices
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = system(
                m,
                x.shape[0],
            )

            loss = normalized_mse(
                prediction,
                x,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item()
            )

            batches += 1

        epoch_loss = (
            running_loss
            / max(batches, 1)
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss
            best_epoch = epoch

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()

                for k, v
                in system.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        ):

            latent = (
                system.global_latent
                .detach()
                .cpu()
                .tolist()
            )

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"epoch {epoch:4d}/"
                f"{args.epochs} | "
                f"normalized MSE "
                f"{epoch_loss:.8f} | "
                f"latent={latent} | "
                f"time={elapsed:.1f}s"
            )

    system.load_state_dict(
        best_state
    )

    print()
    print(
        f"BEST {category_name}: "
        f"epoch={best_epoch} "
        f"MSE={best_loss:.8f}"
    )

    return {
        "system": system,
        "indices": indices.cpu(),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
    }


@torch.no_grad()
def evaluate_specialist(
    category_name,
    result,
    data,
    args,
):
    if result is None:
        return {
            "category": category_name,
            "tensor_count": 0,
            "chunk_count": 0,
        }

    indices = result["indices"]

    chunks = data["chunks"][
        indices
    ].to(args.device)

    positions = data[
        "positions"
    ][indices].to(args.device)

    layer_positions = data[
        "layer_positions"
    ][indices].to(args.device)

    weight_bias = data[
        "weight_bias"
    ][indices].to(args.device)

    shapes = data[
        "shapes"
    ][indices].to(args.device)

    metadata = torch.cat(
        [
            positions,
            layer_positions,
            weight_bias,
            shapes,
        ],
        dim=1,
    )

    system = result["system"]
    system.eval()

    prediction = system(
        metadata,
        chunks.shape[0],
    )

    errors = (
        prediction - chunks
    ).pow(2).mean(dim=1).sqrt()

    target_scale = (
        chunks.pow(2)
        .mean(dim=1)
        .sqrt()
        .clamp_min(1e-8)
    )

    normalized = (
        errors / target_scale
    )

    per_tensor = {}

    for local_i, original_i in enumerate(
        indices.tolist()
    ):
        meta = data["metadata"][
            original_i
        ]

        name = meta["name"]

        per_tensor.setdefault(
            name,
            [],
        ).append(
            float(
                normalized[
                    local_i
                ].item()
            )
        )

    per_tensor = {
        name: sum(values)
        / len(values)
        for name, values
        in per_tensor.items()
    }

    return {
        "category": category_name,
        "tensor_count": len(
            per_tensor
        ),
        "chunk_count": len(indices),
        "best_epoch":
            result["best_epoch"],
        "best_normalized_mse":
            result["best_loss"],
        "latent":
            result["system"]
            .global_latent
            .detach()
            .cpu()
            .tolist(),
        "mean_normalized_rmse":
            float(
                normalized.mean().item()
            ),
        "min_normalized_rmse":
            float(
                normalized.min().item()
            ),
        "max_normalized_rmse":
            float(
                normalized.max().item()
            ),
        "per_tensor_rmse":
            per_tensor,
    }


def main():
    args = parse_args()

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "WHOLE-MODEL TYPE SPECIALISTS V1"
    )
    print("=" * 88)

    print(
        f"Model        : {args.model_dir}"
    )
    print(
        f"Device       : {args.device}"
    )
    print(
        f"Latent/type  : {args.latent_dim}"
    )
    print(
        f"Chunk size   : {args.chunk_size}"
    )
    print(
        f"Batch size   : {args.batch_size}"
    )
    print(
        f"Hidden       : {args.hidden}"
    )
    print(
        f"Epochs       : {args.epochs}"
    )
    print(
        f"LR           : {args.lr}"
    )
    print(
        f"Seed         : {args.seed}"
    )

    print()
    print(
        "Each tensor family gets its OWN "
        "2D global latent + constructor."
    )
    print(
        "LayerNorm is ONE family containing "
        "both LayerNorm types."
    )
    print(
        "No learned per-tensor ID embedding."
    )

    print()
    print("[1] Loading reference model")

    reference = load_reference(
        args.model_dir
    )

    print()
    print(
        "[2] Building whole-model "
        "chunk dataset"
    )

    data = build_dataset(
        reference,
        args.chunk_size,
        args.max_tensors,
    )

    print(
        f"  tensors: "
        f"{len(set("
        "m['name'] for m in data['metadata']"
        ")):,}"
    )

    print(
        f"  chunks : "
        f"{data['chunks'].shape[0]:,}"
    )

    counts = {}

    for meta in data["metadata"]:
        category = meta["category"]
        counts[category] = (
            counts.get(category, 0)
            + 1
        )

    print()
    print("  chunks by type:")

    for category in CATEGORY_NAMES:
        print(
            f"    {category:24s} "
            f"{counts.get(category, 0):,}"
        )

    # Source model is no longer needed.
    del reference

    print()
    print(
        "[3] Training specialists"
    )

    specialist_results = {}
    evaluation = {}

    for category in CATEGORY_NAMES:

        result = train_specialist(
            category,
            data,
            args,
        )

        specialist_results[
            category
        ] = result

        evaluation[
            category
        ] = evaluate_specialist(
            category,
            result,
            data,
            args,
        )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint = {
        "experiment":
            "whole_model_type_specialists_v1",

        "model_dir":
            str(args.model_dir),

        "latent_dim":
            args.latent_dim,

        "rank":
            args.rank,

        "chunk_size":
            args.chunk_size,

        "batch_size":
            args.batch_size,

        "hidden":
            args.hidden,

        "epochs":
            args.epochs,

        "lr":
            args.lr,

        "seed":
            args.seed,

        "category_names":
            CATEGORY_NAMES,

        "metadata":
            data["metadata"],

        "specialists": {},
    }

    for category in CATEGORY_NAMES:

        result = specialist_results[
            category
        ]

        if result is None:
            continue

        system = result[
            "system"
        ]

        checkpoint[
            "specialists"
        ][category] = {
            "state_dict":
                system.state_dict(),

            "best_epoch":
                result[
                    "best_epoch"
                ],

            "best_loss":
                result[
                    "best_loss"
                ],

            "latent":
                system.global_latent
                .detach()
                .cpu()
                .tolist(),
        }

    checkpoint_path = (
        args.output_dir
        / "whole_model_type_specialists_v1.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_result = {
        "experiment":
            "whole_model_type_specialists_v1",

        "model_dir":
            str(args.model_dir),

        "config": {
            "latent_dim":
                args.latent_dim,

            "rank":
                args.rank,

            "chunk_size":
                args.chunk_size,

            "batch_size":
                args.batch_size,

            "hidden":
                args.hidden,

            "epochs":
                args.epochs,

            "lr":
                args.lr,

            "seed":
                args.seed,
        },

        "categories":
            evaluation,
    }

    result_path = (
        args.output_dir
        / "whole_model_type_specialists_v1.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_result,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 88)
    print("TYPE SPECIALIST SUMMARY")
    print("=" * 88)

    for category in CATEGORY_NAMES:

        info = evaluation[
            category
        ]

        if info.get(
            "tensor_count",
            0,
        ) == 0:
            continue

        print(
            f"{category:24s} "
            f"tensors={info['tensor_count']:3d} "
            f"chunks={info['chunk_count']:7d} "
            f"RMSE={info['mean_normalized_rmse']:.6f} "
            f"best_epoch={info['best_epoch']:4d}"
        )

    print()
    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Results:    "
        f"{result_path}"
    )

    print()
    print(
        "Next step: use the checkpoint to "
        "reconstruct ALL tensor families "
        "and load them into a fresh Qwen."
    )


if __name__ == "__main__":
    main()
