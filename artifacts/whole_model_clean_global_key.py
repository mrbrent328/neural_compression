# whole_model_layernorm_only.py
#
# V4 follow-up:
# Train the same global-key constructor ONLY on LayerNorm tensors.
#
# Purpose:
#   Test whether the strong LayerNorm reconstruction from the whole-model
#   experiment becomes dramatically better when the constructor doesn't have
#   to spend capacity on the other tensor families.
#
# Key properties preserved from V4:
#   - global latent: 2 dimensions
#   - rank: 48
#   - chunk size: 1024
#   - batch size: 512
#   - hidden: 512
#   - lr: 0.0002
#   - no tensor-ID embedding
#   - same reconstruction objective
#
# Follow-up:
#   The saved checkpoint can be used to reconstruct the LayerNorm tensors
#   and insert them back into Qwen for behavioral testing.

import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "Qwen2-0.5B"

SEED = 9

LATENT_DIM = 2
RANK = 48
CHUNK_SIZE = 1024
BATCH_SIZE = 512
HIDDEN = 512

EPOCHS = 500
LR = 0.0002

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_FILE = "layernorm_only_v1.json"
CHECKPOINT_FILE = "layernorm_only_v1.pt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading model: {MODEL_DIR}")
print(f"Device: {DEVICE}")

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)


# ============================================================
# COLLECT ONLY LAYERNORM TENSORS
# ============================================================

layernorm_tensors = []

for name, tensor in model.state_dict().items():

    # Match the same semantic family used in the V4 result.
    if (
        name.endswith("input_layernorm.weight")
        or name.endswith("post_attention_layernorm.weight")
    ):
        layernorm_tensors.append(
            {
                "name": name,
                "tensor": tensor.detach().float().cpu(),
            }
        )


# Include final model norm if desired?
#
# V4 classified model.norm.weight separately as "output_norm",
# so DO NOT include it here. This keeps this experiment specifically
# about the transformer LayerNorm family.
#
# Therefore this should produce 48 tensors for Qwen2-0.5B.


print()
print("=" * 70)
print("LAYERNORM DATASET")
print("=" * 70)

print(f"LayerNorm tensors: {len(layernorm_tensors)}")

for item in layernorm_tensors:
    print(
        f"{item['name']:65s} "
        f"shape={tuple(item['tensor'].shape)}"
    )


# ============================================================
# BUILD CHUNKS
# ============================================================

samples = []

for tensor_id, item in enumerate(layernorm_tensors):

    name = item["name"]
    tensor = item["tensor"].flatten()

    # Normalize each tensor independently exactly as a reconstruction
    # target. Keep the statistics so reconstruction can be reversed.
    mean = tensor.mean()
    std = tensor.std(unbiased=False)

    # Protect against degenerate tensors.
    if std < 1e-12:
        std = torch.tensor(1.0)

    normalized = (tensor - mean) / std

    num_chunks = math.ceil(len(normalized) / CHUNK_SIZE)

    for chunk_id in range(num_chunks):

        start = chunk_id * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(normalized))

        chunk = normalized[start:end]

        # Pad final chunk.
        if len(chunk) < CHUNK_SIZE:
            padded = torch.zeros(CHUNK_SIZE)
            padded[: len(chunk)] = chunk
            chunk = padded

        mask = torch.zeros(CHUNK_SIZE)
        mask[: end - start] = 1.0

        samples.append(
            {
                "tensor_id": tensor_id,
                "chunk_id": chunk_id,
                "target": chunk,
                "mask": mask,
            }
        )


print()
print(f"Total chunks: {len(samples)}")


# ============================================================
# DATASET TENSORS
# ============================================================

targets = torch.stack(
    [x["target"] for x in samples]
).float()

masks = torch.stack(
    [x["mask"] for x in samples]
).float()

tensor_ids = torch.tensor(
    [x["tensor_id"] for x in samples],
    dtype=torch.long,
)

num_samples = len(samples)


# ============================================================
# CONSTRUCTOR
# ============================================================
#
# IMPORTANT:
#
# There is deliberately NO tensor-ID embedding here.
#
# The constructor receives:
#
#       latent
#
# and produces a representation capable of reconstructing the target
# chunk.
#
# This keeps the experiment comparable to the V4 "clean global key"
# experiment.
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

        self.latent_dim = latent_dim
        self.rank = rank
        self.chunk_size = chunk_size

        # Shared constructor.
        #
        # latent -> hidden -> rank factors -> chunk
        #
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),

            nn.Linear(hidden, hidden),
            nn.GELU(),

            nn.Linear(hidden, rank * chunk_size),
        )

    def forward(self, latent):

        x = self.network(latent)

        return x.view(
            latent.shape[0],
            self.rank,
            self.chunk_size,
        )


# ============================================================
# LOW-RANK RECONSTRUCTION HEAD
# ============================================================

class ReconstructionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.constructor = Constructor()

        # One global key.
        self.latent = nn.Parameter(
            torch.tensor(
                [-0.0002, 0.0002],
                dtype=torch.float32,
            )
        )

        # Shared projection that converts rank representation
        # into the final chunk.
        self.projection = nn.Parameter(
            torch.randn(
                RANK,
                device=DEVICE,
            )
            * 0.01
        )

    def forward(self, batch_size):

        latent = self.latent.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        factors = self.constructor(latent)

        # rank x chunk -> chunk
        reconstruction = torch.einsum(
            "brc,r->bc",
            factors,
            self.projection,
        )

        return reconstruction


# ============================================================
# MODEL
# ============================================================

reconstructor = ReconstructionModel().to(DEVICE)

optimizer = torch.optim.Adam(
    reconstructor.parameters(),
    lr=LR,
)


# ============================================================
# NORMALIZED MSE
# ============================================================

def normalized_mse(pred, target, mask):

    diff = (pred - target) * mask

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

    return mse / target_mse.clamp_min(1e-12)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING LAYERNORM-ONLY GLOBAL KEY")
print("=" * 70)

print(f"Epochs:       {EPOCHS}")
print(f"Latent dim:   {LATENT_DIM}")
print(f"Rank:         {RANK}")
print(f"Chunk size:   {CHUNK_SIZE}")
print(f"Batch size:   {BATCH_SIZE}")
print(f"Learning rate:{LR}")
print(f"Seed:         {SEED}")
print()

start_time = time.time()

best_loss = float("inf")
best_state = None
best_epoch = 0

for epoch in range(1, EPOCHS + 1):

    # Shuffle chunks.
    permutation = torch.randperm(num_samples)

    epoch_loss = 0.0
    num_batches = 0

    for start in range(0, num_samples, BATCH_SIZE):

        indices = permutation[
            start : start + BATCH_SIZE
        ]

        batch_targets = targets[
            indices
        ].to(DEVICE)

        batch_masks = masks[
            indices
        ].to(DEVICE)

        batch_size = batch_targets.shape[0]

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = reconstructor(
            batch_size
        )

        loss = normalized_mse(
            prediction,
            batch_targets,
            batch_masks,
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    epoch_loss /= num_batches

    if epoch_loss < best_loss:

        best_loss = epoch_loss
        best_epoch = epoch

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in reconstructor.state_dict().items()
        }

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        latent = (
            reconstructor.latent
            .detach()
            .cpu()
            .tolist()
        )

        elapsed = time.time() - start_time

        print(
            f"epoch {epoch:4d}/{EPOCHS} | "
            f"normalized MSE {epoch_loss:.8f} | "
            f"latent={latent} | "
            f"time={elapsed:.1f}s"
        )


# ============================================================
# RESTORE BEST STATE
# ============================================================

if best_state is not None:

    reconstructor.load_state_dict(
        best_state
    )


final_latent = (
    reconstructor.latent
    .detach()
    .cpu()
    .tolist()
)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

torch.save(
    {
        "experiment":
            "layernorm_only_global_key_v1",

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

        "batch_size":
            BATCH_SIZE,

        "hidden":
            HIDDEN,

        "epochs":
            EPOCHS,

        "lr":
            LR,

        "best_epoch":
            best_epoch,

        "best_loss":
            best_loss,

        "global_latent":
            final_latent,

        "constructor_state":
            reconstructor.state_dict(),

        "layernorm_names":
            [
                x["name"]
                for x in layernorm_tensors
            ],
    },
    CHECKPOINT_FILE,
)


# ============================================================
# SAVE HUMAN-READABLE RESULT
# ============================================================

result = {
    "experiment":
        "layernorm_only_global_key_v1",

    "model_dir":
        MODEL_DIR,

    "latent_dim":
        LATENT_DIM,

    "rank":
        RANK,

    "chunk_size":
        CHUNK_SIZE,

    "batch_size":
        BATCH_SIZE,

    "hidden":
        HIDDEN,

    "epochs":
        EPOCHS,

    "lr":
        LR,

    "seed":
        SEED,

    "num_layernorm_tensors":
        len(layernorm_tensors),

    "num_chunks":
        num_samples,

    "best_epoch":
        best_epoch,

    "normalized_global_mse":
        best_loss,

    "global_latent":
        final_latent,

    "layernorm_names":
        [
            x["name"]
            for x in layernorm_tensors
        ],
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
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

print(
    f"Best epoch:          {best_epoch}"
)

print(
    f"Best normalized MSE: {best_loss:.8f}"
)

print(
    f"Final latent:        {final_latent}"
)

print(
    f"Saved checkpoint:    {CHECKPOINT_FILE}"
)

print(
    f"Saved result:        {OUTPUT_FILE}"
)
