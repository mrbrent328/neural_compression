#!/usr/bin/env python3
"""Generate byte-level text from an RLM v1 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import RelationalLanguageModel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = RelationalLanguageModel(256, cfg["dim"], cfg["heads"], cfg["rounds"], cfg["context"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    generated = bytearray(args.prompt.encode("utf-8"))
    context = cfg["context"]
    temperature = max(args.temperature, 1e-6)

    with torch.no_grad():
        for _ in range(args.tokens):
            window = generated[-context:]
            x = torch.tensor([list(window)], dtype=torch.long)
            logits = model(x)[0, -1] / temperature
            if args.top_k > 0 and args.top_k < logits.numel():
                values, indices = torch.topk(logits, args.top_k)
                filtered = torch.full_like(logits, float("-inf"))
                filtered[indices] = values
                logits = filtered
            probs = torch.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()
            generated.append(next_byte)

    print(generated.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
