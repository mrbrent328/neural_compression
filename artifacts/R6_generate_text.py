#!/usr/bin/env python3
"""
R6 — Causal Text Generation

Loads a causal text model and generates from a prompt.
"""

import argparse
from pathlib import Path
import torch

from R6_causal_text_model import CausalLocalTextModel


def generate(model, prompt, vocab, inverse, length, device):
    ids = [vocab[c] for c in prompt if c in vocab]

    for _ in range(length):
        x = torch.tensor(
            ids[-model.seq_len:],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        logits, _ = model(x)

        next_id = torch.argmax(
            logits[:, -1, :],
            dim=-1
        ).item()

        ids.append(next_id)

    return "".join(inverse[i] for i in ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--length", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(
        args.checkpoint,
        map_location=device
    )

    vocab = ckpt["vocab"]
    inverse = {v: k for k, v in vocab.items()}
    cfg = ckpt["config"]

    model = CausalLocalTextModel(
        len(vocab),
        cfg["seq_len"],
        cfg["dim"],
        cfg["heads"],
        cfg["layers"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
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
