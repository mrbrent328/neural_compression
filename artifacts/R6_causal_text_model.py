#!/usr/bin/env python3
"""
R6 — Causal Local Representation Text Model

First predictive experiment.

Unlike R5:
- future characters are hidden during training
- model predicts the next character

Goal:
text context -> latent representation -> next character

This is still a small experiment, not a full LLM.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CausalTextDataset(Dataset):
    def __init__(self, path, seq_len):
        text = Path(path).read_text(errors="ignore")

        chars = sorted(set(text))
        self.vocab = {c: i for i, c in enumerate(chars)}

        ids = [self.vocab[c] for c in text]

        self.samples = []
        for i in range(len(ids) - seq_len - 1):
            x = ids[i:i + seq_len]
            y = ids[i + 1:i + seq_len + 1]

            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CausalLocalTextModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        dim=64,
        heads=4,
        layers=2,
    ):
        super().__init__()

        self.seq_len = seq_len

        self.embedding = nn.Embedding(
            vocab_size,
            dim,
        )

        self.position = nn.Parameter(
            torch.zeros(1, seq_len, dim)
        )

        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
        )

        self.interaction = nn.TransformerEncoder(
            block,
            num_layers=layers,
        )

        self.decoder = nn.Linear(
            dim,
            vocab_size,
        )

    def forward(self, x):
        n = x.shape[1]

        state = self.embedding(x)
        state = state + self.position[:, :n]

        mask = torch.triu(
            torch.ones(
                n,
                n,
                device=x.device,
            ) * float("-inf"),
            diagonal=1,
        )

        state = self.interaction(
            state,
            mask=mask,
        )

        return self.decoder(state), state


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CausalTextDataset(
        args.text_file,
        args.seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = CausalLocalTextModel(
        len(dataset.vocab),
        args.seq_len,
        args.dim,
        args.heads,
        args.layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    loss_fn = nn.CrossEntropyLoss()

    print("=" * 60)
    print("R6 — CAUSAL TEXT MODEL")
    print("=" * 60)
    print("Device:", device)
    print("Samples:", len(dataset))
    print("Vocabulary:", len(dataset.vocab))
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    for epoch in range(args.epochs):
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch+1:4d}/{args.epochs} "
                f"loss={total/len(loader):.6f}"
            )

    torch.save(
        {
            "model": model.state_dict(),
            "vocab": dataset.vocab,
            "config": vars(args),
        },
        args.output,
    )

    print("Saved:", args.output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text-file", required=True)
    parser.add_argument(
        "--output",
        default="r6_causal_text_model.pt",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
