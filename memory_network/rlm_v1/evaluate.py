#!/usr/bin/env python3
"""Evaluate an RLM v1 checkpoint on a held-out suffix of a byte corpus."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import RelationalLanguageModel


def load_ids(path: Path) -> torch.Tensor:
    return torch.tensor(list(path.read_bytes()), dtype=torch.long)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("data")
    p.add_argument("--split", type=float, default=0.9, help="fraction used for training; evaluation uses the remainder")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batches", type=int, default=100)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = RelationalLanguageModel(256, cfg["dim"], cfg["heads"], cfg["rounds"], cfg["context"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = load_ids(Path(args.data))
    split = int(len(ids) * args.split)
    context = cfg["context"]
    if split <= context + 1 or len(ids) - split <= context + 1:
        raise ValueError("corpus is too short for the requested split/context")

    eval_ids = ids[split:]
    losses = []
    with torch.no_grad():
        for _ in range(args.batches):
            starts = torch.randint(0, len(eval_ids) - context - 1, (args.batch_size,))
            x = torch.stack([eval_ids[s:s + context] for s in starts])
            y = torch.stack([eval_ids[s + 1:s + context + 1] for s in starts])
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item())

    loss = sum(losses) / len(losses)
    print(f"validation_loss={loss:.6f}")
    print(f"validation_ppl={torch.exp(torch.tensor(loss)).item():.6f}")
    print(f"split={args.split:.3f} context={context} batches={args.batches}")


if __name__ == "__main__":
    main()
