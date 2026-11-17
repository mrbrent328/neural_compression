#!/usr/bin/env python3
"""R7: causal local-token language prediction.

Purpose
-------
Carry the local-state / shared-interaction idea from the CIFAR experiments
into causal language prediction without turning the experiment into a
multi-layer conventional Transformer.

This first implementation intentionally keeps the experiment small and
explicit. The tokenizer is a deterministic word/punctuation tokenizer so
that tokens are meaningful language units rather than individual bytes or
characters. The interaction block preserves one state per context position,
uses a causal self-attention interaction, and projects the final local state
to the next-token vocabulary.

The script also saves the learned local states for validation examples and
runs the R4-style latent-state measurements on them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 314
DEFAULT_CONTEXT = 4
DEFAULT_EMBED_DIM = 48
DEFAULT_FF_DIM = 512
DEFAULT_HEADS = 1
DEFAULT_BATCH = 16
DEFAULT_EPOCHS = 100
DEFAULT_LR = 0.04
DEFAULT_INTERACTIONS = 1


@dataclass
class Config:
    context: int = DEFAULT_CONTEXT
    embed_dim: int = DEFAULT_EMBED_DIM
    ff_dim: int = DEFAULT_FF_DIM
    heads: int = DEFAULT_HEADS
    batch: int = DEFAULT_BATCH
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    interactions: int = DEFAULT_INTERACTIONS
    seed: int = SEED
    val_fraction: float = 0.10


class SimpleTokenizer:
    """Deterministic word/punctuation tokenizer.

    Words, numbers, and individual punctuation marks are separate tokens.
    This is deliberately simple for the first architecture-transfer test.
    """

    PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)

    def __init__(self, text: str):
        tokens = self.tokenize(text)
        vocab = sorted(set(tokens))
        self.itos = ["<UNK>"] + vocab
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls.PATTERN.findall(text)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, 0) for tok in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] if 0 <= i < len(self.itos) else "<UNK>" for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


class NextTokenDataset(Dataset):
    def __init__(self, ids: list[int], context: int):
        if len(ids) <= context:
            raise ValueError("Corpus is shorter than context + 1 tokens")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context = context

    def __len__(self) -> int:
        return self.ids.numel() - self.context

    def __getitem__(self, index: int):
        x = self.ids[index : index + self.context]
        y = self.ids[index + self.context]
        return x, y


class LocalInteractionBlock(nn.Module):
    """One shared causal interaction over local states.

    The same block is applied to the complete set of local states. Residual
    connections keep each position represented after interaction.
    """

    def __init__(self, dim: int, ff_dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"embed_dim ({dim}) must be divisible by heads ({heads})")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal mask: position i may only interact with positions <= i.
        n = x.size(1)
        causal = torch.triu(
            torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x


class LocalTokenModel(nn.Module):
    def __init__(self, vocab_size: int, context: int, embed_dim: int, ff_dim: int, heads: int, interactions: int):
        super().__init__()
        self.context = context
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Parameter(torch.zeros(1, context, embed_dim))
        self.interactions = nn.ModuleList(
            [LocalInteractionBlock(embed_dim, ff_dim, heads) for _ in range(interactions)]
        )
        self.output = nn.Linear(embed_dim, vocab_size)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, return_states: bool = False):
        if tokens.size(1) != self.context:
            raise ValueError(f"Expected context {self.context}, got {tokens.size(1)}")
        x = self.embedding(tokens) + self.position
        for block in self.interactions:
            x = block(x)
        # Predict the token immediately following the observed context.
        logits = self.output(x[:, -1, :])
        if return_states:
            return logits, x
        return logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_ids(ids: list[int], val_fraction: float) -> tuple[list[int], list[int]]:
    cut = max(1, int(len(ids) * (1.0 - val_fraction)))
    # Keep the split contiguous so validation remains a genuine held-out
    # continuation rather than randomly leaking neighboring text.
    return ids[:cut], ids[cut:]


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.numel()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
    loss = total_loss / total
    acc = total_correct / total
    return loss, acc, math.exp(min(loss, 20.0))


def latent_analysis(states: np.ndarray) -> dict:
    # states: [examples, context, embed_dim]
    token_variance = states.var(axis=0).mean(axis=1)
    token_means = states.mean(axis=0)
    token_similarity = np.corrcoef(token_means)
    token_activity = states.mean(axis=2).var(axis=0)
    return {
        "examples": int(states.shape[0]),
        "tokens": int(states.shape[1]),
        "local_dim": int(states.shape[2]),
        "token_variance": token_variance.tolist(),
        "token_activity": token_activity.tolist(),
        "token_similarity": np.nan_to_num(token_similarity).tolist(),
        "overall_variance": float(states.var()),
    }


def collect_states(model, loader, device, max_batches: int = 32):
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch_index, (x, _) in enumerate(loader):
            if batch_index >= max_batches:
                break
            _, states = model(x.to(device), return_states=True)
            chunks.append(states.cpu().numpy())
    if not chunks:
        return np.empty((0, model.context, model.embed_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("r7_results"))
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--ff-dim", type=int, default=DEFAULT_FF_DIM)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--interactions", type=int, default=DEFAULT_INTERACTIONS)
    args = parser.parse_args()

    cfg = Config(
        context=args.context,
        embed_dim=args.embed_dim,
        ff_dim=args.ff_dim,
        heads=args.heads,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        interactions=args.interactions,
    )

    seed_everything(cfg.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    text = args.text.read_text(encoding="utf-8")
    tokenizer = SimpleTokenizer(text)
    ids = tokenizer.encode(text)
    train_ids, val_ids = split_ids(ids, cfg.val_fraction)

    train_ds = NextTokenDataset(train_ids, cfg.context)
    val_ds = NextTokenDataset(val_ids, cfg.context)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalTokenModel(
        tokenizer.vocab_size,
        cfg.context,
        cfg.embed_dim,
        cfg.ff_dim,
        cfg.heads,
        cfg.interactions,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    config_record = asdict(cfg) | {
        "text": str(args.text),
        "vocab_size": tokenizer.vocab_size,
        "token_count": len(ids),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "optimizer": "SGD",
    }
    (args.out / "config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    (args.out / "vocab.json").write_text(json.dumps(tokenizer.itos, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(config_record, indent=2))

    history = []
    best_val_acc = -1.0
    best_val_ppl = float("inf")
    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.perf_counter()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y.numel()
            train_correct += (logits.argmax(dim=-1) == y).sum().item()
            train_total += y.numel()

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss, val_acc, val_ppl = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_ppl": val_ppl,
            "seconds": elapsed,
        }
        history.append(record)
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} ppl {val_ppl:.3f} | {elapsed:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out / "best_accuracy.pt")
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), args.out / "best_perplexity.pt")

    total_time = time.perf_counter() - start
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    states = collect_states(model, val_loader, device)
    np.save(args.out / "local_states.npy", states)
    (args.out / "latent_analysis.json").write_text(
        json.dumps(latent_analysis(states), indent=2), encoding="utf-8"
    )

    summary = {
        "best_val_accuracy": max(r["val_acc"] for r in history),
        "best_val_accuracy_epoch": max(history, key=lambda r: r["val_acc"])["epoch"],
        "best_val_perplexity": min(r["val_ppl"] for r in history),
        "best_val_perplexity_epoch": min(history, key=lambda r: r["val_ppl"])["epoch"],
        "final_val_accuracy": history[-1]["val_acc"],
        "final_val_perplexity": history[-1]["val_ppl"],
        "training_seconds": total_time,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
