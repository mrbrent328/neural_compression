"""Small, inspectable relational language model.

S7E1 design:
- persistent token identity embeddings
- per-occurrence contextual states
- shared relational interaction field
- causal next-token prediction
- no supplied graph or geometry
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelationalField(nn.Module):
    def __init__(self, dim: int, heads: int = 4, rounds: int = 2):
        super().__init__()
        self.rounds = rounds
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, causal: bool = True, return_attention: bool = False):
        weights = None
        mask = None
        if causal:
            n = x.size(1)
            mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        for _ in range(self.rounds):
            y, weights = self.attn(x, x, x, attn_mask=mask, need_weights=return_attention)
            x = self.norm1(x + y)
            x = self.norm2(x + self.ff(x))
        return (x, weights) if return_attention else x


class RelationalLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, heads: int = 4, rounds: int = 2, max_context: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_context = max_context
        self.identity = nn.Embedding(vocab_size, dim)
        self.position = nn.Embedding(max_context, dim)
        self.field = RelationalField(dim, heads, rounds)
        self.readout = nn.Linear(dim, vocab_size, bias=False)
        self.readout.weight = self.identity.weight

    def forward(self, input_ids: torch.Tensor, return_diagnostics: bool = False):
        if input_ids.size(1) > self.max_context:
            raise ValueError("sequence exceeds max_context")
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.identity(input_ids) + self.position(positions)
        x, attention = self.field(x, causal=True, return_attention=return_diagnostics)
        logits = self.readout(x)
        if return_diagnostics:
            return logits, {"contextual_states": x.detach(), "attention": None if attention is None else attention.detach()}
        return logits
