#!/usr/bin/env python3
"""
LocalRelationalLanguage_20260827_E1

E1 — R3-style Local Representations + Shared Causal Interaction
for character-level language prediction.

Purpose
-------
Test whether the architecture used by the image experiments can perform
causal language prediction without replacing it with a conventional
TransformerEncoder.

Architecture
------------
    character IDs
        -> embedding
        -> local representations
        -> shared causal interaction
        -> local representations
        -> final-position prediction head

The interaction block intentionally mirrors the R3 design philosophy:
LayerNorm -> multi-head self-attention -> residual -> LayerNorm -> FFN ->
residual. Attention is causally masked so a position cannot see future input.

Important experimental controls
-------------------------------
* Character-level vocabulary. No BPE/SentencePiece/pretrained tokenizer.
* No learned positional embedding in E1.
* One interaction block.
* 32-dimensional local representations.
* 4 attention heads.
* 16-character context.
* The model predicts only the next character from the final local state.

This is an architecture experiment, not a compression experiment yet.
Compression/range coding can be added after prediction is validated.

Usage
-----
    python3 LocalRelationalLanguage_20260827_E1.py TEXTFILE

Optional:
    --epochs 20
    --context 16
    --embed-dim 32
    --ff-dim 64
    --heads 4
    --batch-size 128
    --lr 0.001
    --device cpu
    --sample "The "

Outputs a JSON results file beside the input unless --results is supplied.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


SEED = 314
DEFAULT_CONTEXT = 10
DEFAULT_EMBED_DIM = 48
DEFAULT_FF_DIM = 300
DEFAULT_HEADS = 4
DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Character tokenizer
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Minimal reversible character tokenizer.

    The vocabulary is constructed from the training text. Each unique
    character receives one integer ID. This intentionally keeps tokenization
    simple so E1 tests architecture rather than tokenizer quality.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Training text contains no characters.")
        self.itos = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            raise ValueError(
                "Text contains characters absent from the training vocabulary: "
                + repr(sorted(set(unknown)))
            )
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NextCharDataset(Dataset):
    def __init__(self, ids: list[int], context_size: int):
        if len(ids) <= context_size:
            raise ValueError(
                f"Text has {len(ids)} characters but context requires "
                f"more than {context_size}."
            )
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.ids) - self.context_size

    def __getitem__(self, index: int):
        x = self.ids[index:index + self.context_size]
        y = self.ids[index + self.context_size]
        return x, y


# ---------------------------------------------------------------------------
# R3-style shared interaction
# ---------------------------------------------------------------------------


class SharedCausalInteraction(nn.Module):
    """One shared relational block over local representations.

    This is deliberately written explicitly instead of using
    nn.TransformerEncoderLayer so the experiment preserves the architecture
    under investigation rather than silently substituting a standard
    Transformer implementation.
    """

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
        # x: [batch, sequence, representation]
        sequence_length = x.size(1)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        h = self.norm1(x)
        attended, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attended

        x = x + self.ff(self.norm2(x))
        return x


class LocalRelationalLanguageModel(nn.Module):
    """E1 language predictor.

    Each input character becomes a local representation. The shared causal
    interaction layer lets those local states communicate. Only the final
    local state is used to predict the next character.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = DEFAULT_EMBED_DIM,
        ff_dim: int = DEFAULT_FF_DIM,
        heads: int = DEFAULT_HEADS,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.interaction = SharedCausalInteraction(
            dim=embed_dim,
            ff_dim=ff_dim,
            heads=heads,
        )
        self.predictor = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        local = self.embedding(tokens)
        local = self.interaction(local)
        final_local = local[:, -1, :]
        return self.predictor(final_local)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            batch = y.numel()
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += batch

    return {
        "loss": total_loss / max(1, total),
        "accuracy": total_correct / max(1, total),
        "perplexity": math.exp(min(20.0, total_loss / max(1, total))),
    }


def train(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            batch = y.numel()
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += batch

        train_loss = total_loss / max(1, total)
        train_accuracy = total_correct / max(1, total)
        validation = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_perplexity": validation["perplexity"],
            "seconds": elapsed,
        }
        history.append(record)

        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {validation['loss']:.4f} | "
            f"val acc {validation['accuracy']:.4f} | "
            f"ppl {validation['perplexity']:.2f} | "
            f"{elapsed:.2f}s"
        )

    return history


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    model,
    tokenizer,
    prompt: str,
    context_size: int,
    length: int,
    temperature: float,
    device,
):
    model.eval()
    ids = tokenizer.encode(prompt)

    with torch.no_grad():
        for _ in range(length):
            context = ids[-context_size:]
            if len(context) < context_size:
                # E1 has no padding token. Repeat the first known character
                # only for short prompts; normal evaluation uses full context.
                context = [context[0]] * (context_size - len(context)) + context

            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0]
            logits = logits / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--ff-dim", type=int, default=DEFAULT_FF_DIM)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--sample-length", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)

    text = args.text.read_text(encoding="utf-8")
    if len(text) < args.context + 2:
        raise ValueError("Input text is too short for the selected context.")

    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)

    dataset = NextCharDataset(ids, args.context)
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Not enough samples after validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available.")

    model = LocalRelationalLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        heads=args.heads,
    ).to(device)

    parameters = sum(p.numel() for p in model.parameters())

    print(f"Experiment: LocalRelationalLanguage_20260827_E1")
    print(f"Input: {args.text}")
    print(f"Characters: {len(text)}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Parameters: {parameters:,}")
    print(f"Context: {args.context}")
    print(f"Embedding dimension: {args.embed_dim}")
    print(f"FFN dimension: {args.ff_dim}")
    print(f"Heads: {args.heads}")
    print(f"Device: {device}")
    print("Positional embedding: NONE (E1 control)")
    print()

    start = time.perf_counter()
    history = train(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        device,
    )
    total_seconds = time.perf_counter() - start

    if args.sample is None:
        prompt = text[:args.context]
    else:
        prompt = args.sample

    sample = generate(
        model,
        tokenizer,
        prompt,
        args.context,
        args.sample_length,
        args.temperature,
        device,
    )

    print()
    print("--- generated sample ---")
    print(sample)
    print("------------------------")

    results_path = args.results or args.text.with_suffix(
        args.text.suffix + ".LocalRelationalLanguage_20260827_E1.json"
    )

    results = {
        "experiment": "LocalRelationalLanguage_20260827_E1",
        "date": "2026-08-27",
        "purpose": "Test R3-style local representations plus shared causal interaction for character-level next-character prediction.",
        "architecture": {
            "tokenization": "character-level",
            "context_size": args.context,
            "embedding_dim": args.embed_dim,
            "ff_dim": args.ff_dim,
            "heads": args.heads,
            "interaction_layers": 1,
            "positional_embedding": False,
            "causal": True,
            "prediction": "final local representation -> next character",
        },
        "input": {
            "path": str(args.text),
            "characters": len(text),
            "vocabulary": tokenizer.vocab_size,
            "samples": len(dataset),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "Adam",
            "seed": args.seed,
            "seconds": total_seconds,
        },
        "parameters": parameters,
        "history": history,
        "generated_sample": sample,
    }

    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
