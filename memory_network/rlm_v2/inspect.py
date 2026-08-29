#!/usr/bin/env python3
"""Inspect persistent token identities learned by RLM v2."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import RelationalLanguageModel
from tokenizer import Vocabulary, tokenize


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = RelationalLanguageModel(
        ckpt["vocab_size"], cfg["dim"], cfg["heads"], cfg["rounds"], cfg["context"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def resolve_token(vocab: Vocabulary, text: str) -> tuple[str, int]:
    parts = tokenize(text)
    if len(parts) != 1 or parts[0] != text:
        raise ValueError(f"'{text}' is not a single tokenizer token")
    if text not in vocab.stoi:
        raise ValueError(f"token not found in vocabulary: {text!r}")
    return text, vocab.stoi[text]


def nearest(model: RelationalLanguageModel, vocab: Vocabulary, token_id: int, k: int):
    emb = F.normalize(model.identity.weight.detach(), dim=1)
    sims = emb @ emb[token_id]
    vals, ids = torch.topk(sims, min(k + 1, len(vocab.itos)))
    rows = []
    for value, idx in zip(vals.tolist(), ids.tolist()):
        if idx == token_id:
            continue
        rows.append((vocab.itos[idx], value))
        if len(rows) >= k:
            break
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect RLM v2 persistent token identity space")
    p.add_argument("checkpoint")
    p.add_argument("--vocab", default="rlm_v2_vocab.txt")
    p.add_argument("--token", help="single token to inspect")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--list", action="store_true", help="show the most frequent vocabulary entries")
    args = p.parse_args()

    model, ckpt = load_model(args.checkpoint)
    vocab = Vocabulary.load(Path(args.vocab))

    print(f"vocab_size={len(vocab.itos)}")
    print(f"embedding_dim={model.dim}")
    print(f"parameters={sum(p.numel() for p in model.parameters())}")

    if args.list:
        print("vocabulary_sample:")
        for i, token in enumerate(vocab.itos[:50]):
            print(f"  {i}: {token!r}")

    if args.token is None:
        return

    token, token_id = resolve_token(vocab, args.token)
    vector = model.identity.weight[token_id].detach()
    print(f"token={token!r}")
    print(f"token_id={token_id}")
    print(f"identity_norm={vector.norm().item():.6f}")
    print(f"identity_mean={vector.mean().item():.6f}")
    print(f"identity_std={vector.std().item():.6f}")
    print("nearest_identities:")
    for other, similarity in nearest(model, vocab, token_id, args.top):
        print(f"  {similarity:+.6f}  {token!r}  {other!r}")


if __name__ == "__main__":
    main()
