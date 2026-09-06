#!/usr/bin/env python3
"""Compatibility runner for latent_transformer_v0.

This keeps the model/training code unchanged but replaces PyTorch's fused
scaled_dot_product_attention call with a plain causal attention implementation.
Useful on GPU stacks where the fused SDPA backend crashes at the native level
(segmentation fault) instead of raising a Python exception.

Usage:
    python latent_transformer_v0_compat.py --smoke-test --mode latent --steps 50
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

import latent_transformer_v0 as base


def manual_attention_forward(self, x):
    b, t, c = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    mask = torch.ones((t, t), device=x.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
    if self.training and self.dropout:
        probs = F.dropout(probs, p=self.dropout)
    y = torch.matmul(probs, v)
    y = y.transpose(1, 2).contiguous().view(b, t, c)
    return self.out(y)


base.CausalSelfAttention.forward = manual_attention_forward

if __name__ == "__main__":
    base.main()
