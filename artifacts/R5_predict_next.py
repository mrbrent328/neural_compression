#!/usr/bin/env python3
"""
R5 — Next Character Prediction Test

Converts the reconstruction model into a simple next-character predictor.

This is a first test only:
- same learned embedding/interactions
- causal mask added so future characters are hidden
- predicts the next character from previous context

This is NOT a full LLM yet.
"""

import argparse
from pathlib import Path

import torch

from R5_text_reconstruction import LocalTextAutoencoder


def causal_mask(size, device):
    return torch.triu(
        torch.ones(size, size, device=device) * float("-inf"),
        diagonal=1,
    )


@torch.no_grad()
def generate(model, seed, vocab, inverse, length, device):
    ids = [vocab[c] for c in seed if c in vocab]

    for _ in range(length):
        x = torch.tensor(
            ids[-model.seq_len:],
            dtype=torch.long,
        ).unsqueeze(0).to(device)

        local = model.embedding(x) + model.position[:, :x.shape[1], :]

        mask = causal_mask(
            x.shape[1],
            device,
        )

        for block in model.interaction.layers:
            local = block(
                local,
                src_mask=mask,
            )

        logits = model.decoder(local[:, -1])

        next_id = torch.argmax(
            logits,
            dim=-1,
        ).item()

        ids.append(next_id)

    return "".join(
        inverse[i]
        for i in ids
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
    )

    vocab = ckpt["vocab"]
    inverse = {v:k for k,v in vocab.items()}
    config = ckpt["config"]

    model = LocalTextAutoencoder(
        len(vocab),
        config["seq_len"],
        config["dim"],
        config["heads"],
        config["layers"],
    ).to(device)

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    print(generate(
        model,
        args.prompt,
        vocab,
        inverse,
        args.length,
        device,
    ))


if __name__ == "__main__":
    main()
