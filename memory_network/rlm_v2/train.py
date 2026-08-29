#!/usr/bin/env python3
"""Train RLM v2 with a real train/validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import RelationalLanguageModel
from tokenizer import Vocabulary, tokenize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--checkpoint", default="rlm_v2_checkpoint.pt")
    p.add_argument("--vocab", default="rlm_v2_vocab.txt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    text = Path(args.data).read_text(encoding="utf-8")
    tokens = tokenize(text)
    split = int(len(tokens) * (1.0 - args.val_fraction))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    vocab = Vocabulary(train_tokens)
    train_ids = torch.tensor(vocab.encode(train_tokens), dtype=torch.long)
    val_ids = torch.tensor(vocab.encode(val_tokens), dtype=torch.long)
    if len(train_ids) <= args.context + 1 or len(val_ids) <= args.context + 1:
        raise ValueError("training and validation data must both exceed context length")

    model = RelationalLanguageModel(len(vocab.itos), args.dim, args.heads, args.rounds, args.context)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        starts = torch.randint(0, len(train_ids) - args.context - 1, (args.batch_size,))
        x = torch.stack([train_ids[s:s + args.context] for s in starts])
        y = torch.stack([train_ids[s + 1:s + args.context + 1] for s in starts])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab.itos)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            print(f"step={step} train_loss={loss.item():.6f} train_ppl={loss.exp().item():.3f}")

    model.eval()
    with torch.no_grad():
        starts = torch.arange(0, len(val_ids) - args.context - 1, args.context)
        losses = []
        for s in starts:
            x = val_ids[s:s + args.context].unsqueeze(0)
            y = val_ids[s + 1:s + args.context + 1].unsqueeze(0)
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, len(vocab.itos)), y.reshape(-1)).item())
        val_loss = sum(losses) / len(losses)

    checkpoint = {
        "model": model.state_dict(),
        "config": vars(args),
        "vocab_size": len(vocab.itos),
    }
    torch.save(checkpoint, args.checkpoint)
    vocab.save(Path(args.vocab))
    print(json.dumps({"checkpoint": args.checkpoint, "vocab": args.vocab, "vocab_size": len(vocab.itos), "train_tokens": len(train_ids), "validation_tokens": len(val_ids), "validation_loss": val_loss, "validation_ppl": float(torch.exp(torch.tensor(val_loss)))}, indent=2))


if __name__ == "__main__":
    main()
