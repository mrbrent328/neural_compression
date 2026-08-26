#!/usr/bin/env python3
"""
R5 — View Text Reconstruction

Loads an R5 checkpoint and prints input/output examples.
"""

import argparse
from pathlib import Path

import torch

from R5_text_reconstruction import LocalTextAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
    )

    vocab = ckpt["vocab"]
    inverse = {v: k for k, v in vocab.items()}
    config = ckpt["config"]

    model = LocalTextAutoencoder(
        len(vocab),
        config["seq_len"],
        config["dim"],
        config["heads"],
        config["layers"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    text = Path(args.text_file).read_text(
        errors="ignore"
    )

    ids = [
        vocab[c]
        for c in text
        if c in vocab
    ]

    print("=" * 60)
    print("R5 RECONSTRUCTION VIEW")
    print("=" * 60)

    with torch.no_grad():
        for i in range(args.samples):
            start = i * config["seq_len"]

            x = torch.tensor(
                ids[start:start+config["seq_len"]],
                dtype=torch.long,
            ).unsqueeze(0).to(device)

            logits, _ = model(x)

            out = logits.argmax(dim=-1)[0]

            original = "".join(
                inverse[int(v)]
                for v in x[0].cpu()
            )

            reconstructed = "".join(
                inverse[int(v)]
                for v in out.cpu()
            )

            print()
            print("INPUT:")
            print(original)
            print()
            print("OUTPUT:")
            print(reconstructed)
            print("-" * 60)


if __name__ == "__main__":
    main()
