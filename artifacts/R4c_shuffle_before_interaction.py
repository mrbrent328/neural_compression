#!/usr/bin/env python3
"""
R4c — Shuffle BEFORE interaction test

Tests whether the shared interaction layer depends on the
structured arrangement of local tokens.

Modes:
normal:
    encoder -> interaction -> decoder

shuffle:
    encoder -> shuffle tokens -> interaction -> decoder

reverse:
    encoder -> reverse tokens -> interaction -> decoder
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from R3_local_interaction_cifar10 import (
    CIFARParquetDataset,
    LocalInteractionAutoencoder,
    find_parquet,
    fixed_split_indices,
    TEST_SIZE,
    SEED,
)


@torch.no_grad()
def evaluate(model, dataset, device, mode):
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    errors = []

    for x, _ in loader:
        x = x.to(device)

        # Get patch embeddings before interaction.
        patches = model.local_encoder(
            __import__("R3_local_interaction_cifar10").patchify(x)
        )

        local = patches + model.position

        if mode == "shuffle":
            idx = torch.randperm(local.shape[1], device=device)
            local = local[:, idx, :]

        elif mode == "reverse":
            idx = torch.arange(
                local.shape[1] - 1,
                -1,
                -1,
                device=device,
            )
            local = local[:, idx, :]

        # Now let the interaction system process the altered structure.
        for block in model.interaction:
            local = block(local)

        recon = model.decode_local(local)

        mse = ((recon - x) ** 2).mean(dim=(1, 2, 3))
        errors.append(mse.cpu().numpy())

    return float(np.concatenate(errors).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    arch = checkpoint["architecture"]

    model = LocalInteractionAutoencoder(
        local_dim=arch["local_dim"],
        ff_dim=arch["ff_dim"],
        heads=arch["heads"],
        interaction_layers=arch["interaction_layers"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    dataset = CIFARParquetDataset(
        find_parquet(args.data_root)
    )

    _, test_indices = fixed_split_indices(
        len(dataset),
        TEST_SIZE,
        SEED,
    )

    test_set = Subset(
        dataset,
        test_indices.tolist(),
    )

    print("R4c shuffle-before-interaction test")
    print()

    for mode in ["normal", "shuffle", "reverse"]:
        score = evaluate(
            model,
            test_set,
            device,
            mode,
        )
        print(f"{mode:10s} MSE = {score:.8f}")


if __name__ == "__main__":
    main()
