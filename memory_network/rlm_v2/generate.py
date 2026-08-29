#!/usr/bin/env python3
"""Generate text from an RLM v2 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import RelationalLanguageModel
from tokenizer import Vocabulary, tokenize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--vocab", default="rlm_v2_vocab.txt")
    p.add_argument("--prompt", default="SOCRATES:")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab = Vocabulary.load(Path(args.vocab))
    model = RelationalLanguageModel(ckpt["vocab_size"], cfg["dim"], cfg["heads"], cfg["rounds"], cfg["context"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = vocab.encode(tokenize(args.prompt))
    if not ids:
        raise ValueError("prompt produced no tokens")
    generated = list(ids)
    with torch.no_grad():
        for _ in range(args.tokens):
            context = generated[-cfg["context"]:]
            x = torch.tensor(context, dtype=torch.long).unsqueeze(0)
            logits = model(x)[0, -1] / max(args.temperature, 1e-5)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            generated.append(next_id)
    print(vocab.decode(generated))


if __name__ == "__main__":
    main()
