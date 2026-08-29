#!/usr/bin/env python3
"""Minimal S7E1 training harness for RLM v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import RelationalLanguageModel


def load_ids(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    return torch.tensor(list(data), dtype=torch.long)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default="rlm_v1_checkpoint.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ids = load_ids(Path(args.data))
    if len(ids) <= args.context + 1:
        raise ValueError("training data is too short for the requested context")

    model = RelationalLanguageModel(256, args.dim, args.heads, args.rounds, args.context)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(ids) - args.context - 1, (args.batch_size,))
        x = torch.stack([ids[s:s + args.context] for s in starts])
        y = torch.stack([ids[s + 1:s + args.context + 1] for s in starts])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            print(f"step={step} loss={loss.item():.6f} ppl={loss.exp().item():.3f}")

    torch.save({"model": model.state_dict(), "config": vars(args)}, args.checkpoint)
    print(json.dumps({"checkpoint": args.checkpoint, "vocab_size": 256, "parameters": sum(p.numel() for p in model.parameters())}, indent=2))


if __name__ == "__main__":
    main()
