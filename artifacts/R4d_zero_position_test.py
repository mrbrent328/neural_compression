#!/usr/bin/env python3
"""
R4d — Zero position embedding test

Question:
Is the learned position embedding required at inference?

Compare:
normal:
    encoder + position + interaction + decoder

zero_position:
    encoder + zero position + interaction + decoder

This does NOT retrain. It only removes the learned GPS-like signal.
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
    patchify,
)


@torch.no_grad()
def evaluate(model, dataset, device, mode):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    errors = []

    model.eval()

    for x, _ in loader:
        x = x.to(device)

        local = model.local_encoder(
            patchify(x)
        )

        if mode == "normal":
            local = local + model.position

        elif mode == "zero_position":
            local = local + torch.zeros_like(model.position)

        for block in model.interaction:
            local = block(local)

        recon = model.decode_local(local)

        mse = ((recon - x) ** 2).mean(dim=(1,2,3))
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    arch = ckpt["architecture"]

    model = LocalInteractionAutoencoder(
        local_dim=arch["local_dim"],
        ff_dim=arch["ff_dim"],
        heads=arch["heads"],
        interaction_layers=arch["interaction_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    dataset = CIFARParquetDataset(
        find_parquet(args.data_root)
    )

    _, test_indices = fixed_split_indices(
        len(dataset),
        TEST_SIZE,
        SEED,
    )

    test_set = Subset(dataset, test_indices.tolist())

    print("R4d zero-position test")
    print()

    for mode in ["normal", "zero_position"]:
        score = evaluate(model, test_set, device, mode)
        print(f"{mode:15s} MSE = {score:.8f}")


if __name__ == "__main__":
    main()
