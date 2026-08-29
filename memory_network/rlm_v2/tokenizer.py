"""Small deterministic word/punctuation tokenizer for RLM v2.

This intentionally favors inspectability over subword sophistication.
Tokens are words, whitespace runs, punctuation, and other non-word runs.
"""
from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"\s+|[A-Za-z0-9_]+(?:['’-][A-Za-z0-9_]+)*|[^\w\s]", re.UNICODE)


class Vocabulary:
    def __init__(self, tokens: list[str], min_count: int = 1):
        counts = Counter(tokens)
        ordered = sorted((tok for tok, n in counts.items() if n >= min_count), key=lambda t: (-counts[t], t))
        self.itos = ["<unk>"] + ordered
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(tok, unk) for tok in tokens]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        path.write_text("\n".join(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path):
        vocab = cls.__new__(cls)
        vocab.itos = path.read_text(encoding="utf-8").splitlines()
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        return vocab


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)
