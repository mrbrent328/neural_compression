#!/usr/bin/env python3
"""
LayerNorm-by-type experiment.

Hypothesis:
    One global latent is being asked to represent two different LayerNorm
    families. Train them separately, then put both reconstructed families
    back into Qwen simultaneously.

Families:
    1. input_layernorm.weight
    2. post_attention_layernorm.weight

Important control:
    Each family gets the SAME 2D latent size and the SAME constructor
    capacity as the LayerNorm-only experiment.

We therefore test:
    one 2D representation for both families
vs.
    one 2D representation per family

This script:
    - loads Qwen2-0.5B
    - extracts the 24 input LayerNorms and 24 post-attention LayerNorms
    - trains a separate global 2D latent + constructor for each family
    - saves both constructors and both latents
    - reconstructs all 48 tensors
    - writes a JSON report with per-tensor normalized RMSE

The functional Qwen test should be run afterward using the saved checkpoint.
"""

import json
import math
import os
import random
import time

import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "Qwen2-0.5B"

SEED = 9

LATENT_DIM = 2
RANK = 48
CHUNK_SIZE = 1024
HIDDEN = 512

BATCH_SIZE = 512

EPOCHS = 500
LR = 0.0002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_FILE = "layernorm_by_type_v1.pt"
RESULT_FILE = "layernorm_by_type_v1.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LAYERNORM-BY-TYPE TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_DIR}")
print()

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float32,
)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad_(False)


# ============================================================
# COLLECT THE TWO FAMILIES
# ============================================================

families = {
    "input_layernorm": [],
    "post_attention_layernorm": [],
}

for name, tensor in model.state_dict().items():

    if name.endswith("input_layernorm.weight"):
        families["input_layernorm"].append(
            {
                "name": name,
                "tensor": tensor.detach().float().cpu(),
            }
        )

    elif name.endswith("post_attention_layernorm.weight"):
        families["post_attention_layernorm"].append(
            {
                "name": name,
                "tensor": tensor.detach().float().cpu(),
            }
        )


print("Tensor counts:")

for family, items in families.items():
    print(
        f"  {family:28s}: {len(items)}"
    )

print()


# ============================================================
# CONSTRUCTOR
# ============================================================

class Constructor(nn.Module):

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        rank=RANK,
        hidden=HIDDEN,
        chunk_size=CHUNK_SIZE,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim,
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
                rank * chunk_size,
            ),
        )


# ============================================================
# FAMILY TRAINER
# ============================================================

class FamilyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.constructor = Constructor()

        # Each family gets its own independent 2D global key.
        self.latent = nn.Parameter(
            torch.tensor(
                [-0.0002, 0.0002],
                dtype=torch.float32,
            )
        )

        self.projection = nn.Parameter(
            torch.randn(
                RANK
            ) * 0.01
        )

    def forward(self, batch_size):

        latent = self.latent.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        factors = self.constructor.network(
            latent
        )

        factors = factors.view(
            batch_size,
            RANK,
            CHUNK_SIZE,
        )

        reconstruction = torch.einsum(
            "brc,r->bc",
            factors,
            self.projection,
        )

        return reconstruction


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(items):

    targets = []

    metadata = []

    for tensor_id, item in enumerate(items):

        original = item["tensor"].flatten()

        mean = original.mean()
        std = original.std(
            unbiased=False
        )

        if std < 1e-12:
            std = torch.tensor(1.0)

        normalized = (
            original - mean
        ) / std

        num_chunks = math.ceil(
            len(normalized) / CHUNK_SIZE
        )

        for chunk_id in range(num_chunks):

            start = (
                chunk_id * CHUNK_SIZE
            )

            end = min(
                start + CHUNK_SIZE,
                len(normalized),
            )

            chunk = normalized[
                start:end
            ]

            if chunk.numel() < CHUNK_SIZE:

                padded = torch.zeros(
                    CHUNK_SIZE
                )

                padded[
                    : chunk.numel()
                ] = chunk

                chunk = padded

            mask = torch.zeros(
                CHUNK_SIZE
            )

            mask[
                : end - start
            ] = 1.0

            targets.append(chunk)

            metadata.append(
                {
                    "tensor_id": tensor_id,
                    "chunk_id": chunk_id,
                    "valid_length":
                        end - start,
                    "mean":
                        mean.item(),
                    "std":
                        std.item(),
                }
            )

    return (
        torch.stack(targets).float(),
        metadata,
    )


# ============================================================
# LOSS
# ============================================================

def normalized_mse(
    prediction,
    target,
    mask,
):

    diff = (
        prediction - target
    ) * mask

    mse = (
        diff.pow(2).sum()
        /
        mask.sum().clamp_min(1.0)
    )

    target_mse = (
        (target * mask).pow(2).sum()
        /
        mask.sum().clamp_min(1.0)
    )

    return (
        mse
        /
        target_mse.clamp_min(1e-12)
    )


# ============================================================
# TRAIN ONE FAMILY
# ============================================================

def train_family(
    family_name,
    items,
):

    print("=" * 70)
    print(f"TRAINING: {family_name}")
    print("=" * 70)

    targets, metadata = build_dataset(
        items
    )

    num_samples = targets.shape[0]

    print(
        f"Tensors: {len(items)}"
    )

    print(
        f"Chunks:  {num_samples}"
    )

    print()

    masks = torch.stack(
        [
            torch.ones(CHUNK_SIZE)
            for _ in range(num_samples)
        ]
    )

    targets = targets.to(DEVICE)
    masks = masks.to(DEVICE)

    network = FamilyModel().to(DEVICE)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=LR,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    start_time = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        permutation = torch.randperm(
            num_samples,
            device=DEVICE,
        )

        epoch_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            num_samples,
            BATCH_SIZE,
        ):

            indices = permutation[
                start :
                start + BATCH_SIZE
            ]

            batch_targets = targets[
                indices
            ]

            batch_masks = masks[
                indices
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = network(
                batch_targets.shape[0]
            )

            loss = normalized_mse(
                prediction,
                batch_targets,
                batch_masks,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            batch_count += 1

        epoch_loss /= batch_count

        if epoch_loss < best_loss:

            best_loss = epoch_loss
            best_epoch = epoch

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in network.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            elapsed = (
                time.time()
                - start_time
            )

            latent = (
                network.latent
                .detach()
                .cpu()
                .tolist()
            )

            print(
                f"epoch {epoch:4d}/{EPOCHS} | "
                f"normalized MSE "
                f"{epoch_loss:.8f} | "
                f"latent={latent} | "
                f"time={elapsed:.1f}s"
            )

    network.load_state_dict(
        best_state
    )

    print()
    print(
        f"BEST {family_name}: "
        f"epoch={best_epoch}, "
        f"MSE={best_loss:.8f}"
    )

    return {
        "network": network,
        "targets": targets,
        "metadata": metadata,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "latent":
            network.latent.detach()
            .cpu()
            .tolist(),
    }


# ============================================================
# RECONSTRUCT FAMILY
# ============================================================

@torch.no_grad()
def reconstruct_family(
    family_name,
    items,
    result,
):

    network = result["network"]

    latent = network.latent.unsqueeze(0)

    factors = network.constructor.network(
        latent
    )

    factors = factors.view(
        1,
        RANK,
        CHUNK_SIZE,
    )

    reconstructed_chunk = torch.einsum(
        "brc,r->bc",
        factors,
        network.projection,
    )[0].cpu()

    diagnostics = []

    for tensor_id, item in enumerate(items):

        original = (
            item["tensor"]
            .detach()
            .float()
            .flatten()
        )

        mean = original.mean()

        std = original.std(
            unbiased=False
        )

        if std < 1e-12:
            std = torch.tensor(1.0)

        normalized = (
            original - mean
        ) / std

        # Qwen2-0.5B LayerNorm vectors
        # fit in one 1024-value chunk.
        reconstructed_normalized = (
            reconstructed_chunk[
                : normalized.numel()
            ]
        )

        reconstructed = (
            reconstructed_normalized
            * std
            + mean
        )

        mse = (
            (
                reconstructed
                - original
            )
            .pow(2)
            .mean()
        )

        normalized_mse_value = (
            mse
            /
            original.pow(2)
            .mean().clamp_min(
                1e-12
            )
        )

        diagnostics.append(
            {
                "name":
                    item["name"],

                "family":
                    family_name,

                "shape":
                    list(item["tensor"].shape),

                "normalized_rmse":
                    math.sqrt(
                        normalized_mse_value.item()
                    ),

                "original_mean":
                    mean.item(),

                "original_std":
                    std.item(),
            }
        )

    return diagnostics


# ============================================================
# TRAIN BOTH FAMILIES
# ============================================================

results = {}

for family_name, items in families.items():

    results[family_name] = (
        train_family(
            family_name,
            items,
        )
    )


# ============================================================
# RECONSTRUCT + DIAGNOSTICS
# ============================================================

all_diagnostics = []

for family_name, items in families.items():

    diagnostics = reconstruct_family(
        family_name,
        items,
        results[family_name],
    )

    all_diagnostics.extend(
        diagnostics
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {}

for family_name in families:

    values = [
        item["normalized_rmse"]
        for item in all_diagnostics
        if item["family"] == family_name
    ]

    summary[family_name] = {
        "tensor_count":
            len(values),

        "mean_normalized_rmse":
            sum(values) / len(values),

        "min_normalized_rmse":
            min(values),

        "max_normalized_rmse":
            max(values),

        "best_epoch":
            results[family_name][
                "best_epoch"
            ],

        "best_normalized_mse":
            results[family_name][
                "best_loss"
            ],

        "latent":
            results[family_name][
                "latent"
            ],
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

checkpoint = {
    "experiment":
        "layernorm_by_type_v1",

    "model_dir":
        MODEL_DIR,

    "seed":
        SEED,

    "latent_dim":
        LATENT_DIM,

    "rank":
        RANK,

    "chunk_size":
        CHUNK_SIZE,

    "hidden":
        HIDDEN,

    "epochs":
        EPOCHS,

    "lr":
        LR,

    "families": {},
}

for family_name in families:

    network = results[
        family_name
    ]["network"]

    checkpoint["families"][
        family_name
    ] = {
        "tensor_names":
            [
                item["name"]
                for item
                in families[family_name]
            ],

        "best_epoch":
            results[family_name][
                "best_epoch"
            ],

        "best_loss":
            results[family_name][
                "best_loss"
            ],

        "latent":
            results[family_name][
                "latent"
            ],

        "state_dict":
            network.state_dict(),
    }


torch.save(
    checkpoint,
    CHECKPOINT_FILE,
)


# ============================================================
# SAVE JSON
# ============================================================

json_result = {
    "experiment":
        "layernorm_by_type_v1",

    "model_dir":
        MODEL_DIR,

    "device":
        DEVICE,

    "config": {
        "latent_dim":
            LATENT_DIM,

        "rank":
            RANK,

        "chunk_size":
            CHUNK_SIZE,

        "hidden":
            HIDDEN,

        "epochs":
            EPOCHS,

        "lr":
            LR,

        "seed":
            SEED,
    },

    "families":
        summary,

    "per_tensor":
        all_diagnostics,
}

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_result,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

for family_name in families:

    info = summary[
        family_name
    ]

    print()
    print(
        f"{family_name}:"
    )

    print(
        f"  Best epoch: "
        f"{info['best_epoch']}"
    )

    print(
        f"  Best normalized MSE: "
        f"{info['best_normalized_mse']:.8f}"
    )

    print(
        f"  Mean normalized RMSE: "
        f"{info['mean_normalized_rmse']:.8f}"
    )

    print(
        f"  Latent: "
        f"{info['latent']}"
    )

print()
print(
    f"Checkpoint: {CHECKPOINT_FILE}"
)

print(
    f"Results:    {RESULT_FILE}"
)
