#!/usr/bin/env python3
"""
LocalRelationalLanguage_20260827_E1_V2

E1_V2 — S4-informed optimizer transfer.

Parent baseline:
    LocalRelationalLanguage_20260827_E1_V1

ONLY intended model/training change from V1:
    Adam(lr=0.001) -> SGD(lr=0.04)

Architecture remains:
    character IDs
      -> embedding (48)
      -> local representations
      -> one shared causal interaction block
      -> final local representation
      -> next-character prediction

Controls:
    context       = 10
    embedding     = 48
    FFN           = 300
    heads         = 4
    interaction   = 1
    positional    = none
    causal        = true
    batch         = 128
    epochs        = 20
    optimizer     = SGD
    learning rate = 0.04

This is an architecture/training experiment, NOT compression yet.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

SEED = 314
DEFAULT_CONTEXT = 10
DEFAULT_EMBED_DIM = 48
DEFAULT_FF_DIM = 300
DEFAULT_HEADS = 4
DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 20
DEFAULT_LR = 0.04


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Training text contains no characters.")
        self.itos = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(ch for ch in text if ch not in self.stoi))
        if unknown:
            raise ValueError(
                "Text contains characters absent from the training vocabulary: "
                + repr(unknown)
            )
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


class NextCharDataset(Dataset):
    def __init__(self, ids: list[int], context_size: int):
        if len(ids) <= context_size:
            raise ValueError("Input is too short for the requested context.")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.ids) - self.context_size

    def __getitem__(self, index: int):
        x = self.ids[index:index + self.context_size]
        y = self.ids[index + self.context_size]
        return x, y


class SharedCausalInteraction(nn.Module):
    """Explicit R3-style shared interaction block."""

    def __init__(self, dim: int, ff_dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("embed_dim must be divisible by number of heads")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        h = self.norm1(x)
        attended, _ = self.attn(
            h, h, h,
            attn_mask=mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ff(self.norm2(x))


class LocalRelationalLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, DEFAULT_EMBED_DIM)
        self.interaction = SharedCausalInteraction(
            dim=DEFAULT_EMBED_DIM,
            ff_dim=DEFAULT_FF_DIM,
            heads=DEFAULT_HEADS,
        )
        self.predictor = nn.Linear(DEFAULT_EMBED_DIM, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        local = self.embedding(tokens)
        local = self.interaction(local)
        return self.predictor(local[:, -1, :])


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)

    loss = total_loss / total
    return {
        "loss": loss,
        "accuracy": total_correct / total,
        "perplexity": math.exp(min(loss, 20.0)),
    }


def generate(model, tokenizer, prompt, length, context_size, device, temperature=0.8):
    model.eval()
    ids = tokenizer.encode(prompt)

    with torch.no_grad():
        for _ in range(length):
            context = ids[-context_size:]
            if len(context) < context_size:
                context = [0] * (context_size - len(context)) + context
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x).squeeze(0) / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

    return tokenizer.decode(ids)


def run(args):
    seed_everything(args.seed)
    device = torch.device(args.device)

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)

    dataset = NextCharDataset(ids, args.context)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LocalRelationalLanguageModel(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        epoch_t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_accuracy = total_correct / total
        val = evaluate(model, val_loader, device)
        seconds = time.time() - epoch_t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_perplexity": val["perplexity"],
            "seconds": seconds,
        }
        history.append(record)
        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} "
            f"ppl {val['perplexity']:.3f} | {seconds:.2f}s"
        )

    elapsed = time.time() - t0
    sample = generate(
        model,
        tokenizer,
        args.sample,
        args.sample_length,
        args.context,
        device,
        args.temperature,
    )

    result = {
        "experiment": "LocalRelationalLanguage_20260827_E1_V2",
        "date": "2026-08-27",
        "parent_experiment": "LocalRelationalLanguage_20260827_E1_V1",
        "purpose": "Isolate the S4-informed optimizer transfer from Adam to SGD.",
        "change_from_parent": {
            "optimizer": "Adam -> SGD",
            "learning_rate": "0.001 -> 0.04",
        },
        "architecture": {
            "tokenization": "character-level",
            "context_size": args.context,
            "embedding_dim": DEFAULT_EMBED_DIM,
            "ff_dim": DEFAULT_FF_DIM,
            "heads": DEFAULT_HEADS,
            "interaction_layers": 1,
            "positional_embedding": False,
            "causal": True,
            "prediction": "final local representation -> next character",
        },
        "input": {
            "path": str(input_path),
            "characters": len(text),
            "vocabulary": tokenizer.vocab_size,
            "samples": len(dataset),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "SGD",
            "seed": args.seed,
            "seconds": elapsed,
        },
        "parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "generated_sample": sample,
    }

    output = Path(args.results) if args.results else input_path.with_name(
        input_path.name + ".LocalRelationalLanguage_20260827_E1_V2.json"
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"results: {output}")
    print(f"parameters: {result['parameters']}")
    print(f"total time: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", default="The ")
    parser.add_argument("--sample-length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--results", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
