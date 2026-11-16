#!/usr/bin/env python3
"""
Fast Latent Transformer v1.0
=======================

Optimized version of latent_transformer_v0 for faster inference and training.
Key improvements based on research insights:
1. Latent/shared-constructor architecture for parameter efficiency
2. Fast factor generation with batched coordinates
3. Factor caching for inference acceleration
4. Minimal memory footprint with parameter generation

Based on research insights from Neural Compression project.
"""

from __future__ import annotation

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
@dataclass
class ModelConfig:
    vocab_size: int = 256
    context: int = 256
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0
    mode: str = "latent"  # dense, lowrank, latent
    rank: int = 64
    latent_dim: int = 32
    constructor_hidden: int = 128
    constructor_type_dim: int = 16
    constructor_fourier_bands: int = 8
    # Fast-LLM specific
    constructor_groups: int = 4  # Number of factor sets to generate per constructor call
    cache_factors: bool = True  # Cache generated factors for inference
    activation_aware: bool = True  # Use activation-aware rank selection

    def to_dict(self) -> Dict[str, any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}

    def __post_init__(self):
        # Ensure dependencies are met
        self.rank = min(self.rank, self.d_model // 2)
        self.latent_dim = min(self.latent_dim, self.rank)

    def get_constructor_groups(self, d_model: int) -> int:
        # Determine number of groups based on dimension
        return max(1, min(self.constructor_groups, d_model // (self.rank * 4)))
class MatrixGroupAwareConstructor(nn.Module):
    """Efficient constructor that generates multiple factor sets with batched processing."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.hidden = config.constructor_hidden
        self.type_dim = config.constructor_type_dim
        self.fourier_bands = config.constructor_fourier_bands
        self.groups = config.constructor_groups

        # Fourier features for coordinates - batch optimized
        self.fourier_weights = nn.Parameter(
            torch.randn(config.fourier_bands * 2, config.constructor_hidden)
        )

        # Type embeddings for different factor types (L vs R)
        self.type_embedding = nn.Embedding(2, config.constructor_hidden)

        # Generator MLP for batch processing
        self.generator = nn.Sequential(
            nn.Linear(config.constructor_hidden * 3, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.rank * config.groups)
        )

        # Coordinate projection
        self.coord_proj = nn.Linear(config.fourier_bands * 2, config.constructor_hidden)

    def forward(self, latent: Tensor, matrix_type: int, coords: Tensor) -> Tensor:
        """
        Generate factors for multiple MLP matrices efficiently.

        Args:
            latent: (batch, latent_dim) - latent vectors for each matrix
            matrix_type: int - 0 for L, 1 for R
            coords: (batch, coord_dim) - Fourier coordinate features

        Returns:
            (batch, groups, rank) - generated factors
        """
        batch_size = latent.shape[0]

        # Type embedding
        type_emb = self.type_embedding(torch.tensor([matrix_type], device=latent.device)).expand(batch_size, -1)

        # Fourier encoding
        freq = 2 * math.pi
        fourier_emb = torch.cat([torch.sin(coords * freq), torch.cos(coords * freq)], dim=-1)

        # Encode coordinates
        coord_encoded = self.coord_proj(fourier_emb)

        # Combine inputs
        combined = torch.cat([latent, type_emb, coord_encoded], dim=-1)

        # Generate all groups at once
        generated = self.generator(combined)
        factors = generated.view(batch_size, self.groups, self.rank)

        return factors
class FastLatentMLP(nn.Module):
    """MLP with latent factors and caching for faster inference."""

    def __init__(self, config: ModelConfig, constructor: MatrixGroupAwareConstructor):
        super().__init__()
        self.config = config
        self.constructor = constructor

        # Latents for each layer
        self.layer_latents = nn.Parameter(torch.randn(config.n_layers, config.latent_dim))

        # Layer gains
        self.layer_gains = nn.Parameter(torch.ones(config.n_layers))

        # Factor cache for inference
        self.factor_cache: Dict[int, Tuple[Tensor, Tensor]] = {}

    def get_factors_for_layer(self, layer_idx: int, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Get or generate factors for a specific layer."""
        layer = layer_idx % self.config.n_layers

        if layer not எ                     # Pre-compute coordinates once for this layer and sequence length
        coords = self._get_coords(layer, x.shape[1]

        l_factors = self.constructor.forward(latent, matrix_type=2, coords=coords)
        r_factors = self.constructor.forward(latent, matrix_type=1, coords=coords)

        self.factor_cache[layer] = (l_factors, r_factors)

        return self.factor_cache[layer]

    def _get_coords(self, layer_idx: int, seq_len: int) -> Tensor:
        """Generate coordinate features."""
        device = self.layer_latents.device
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Layer-specific frequency modulation
        layer_freq = 2.0 ** (layer_idx % 4)

        # Simple positional encoding with layer modulation
        coords = torch.cat([
            torch.sin(pos * layer_freq / self.config.d_model),
            torch.cos(pos * layer_freq / self.config.d_model),
            torch.tensor([layer_idx % self.config.n_heads], device=device).float().repeat(seq_len, 1) / self.config.d_model,
        ], dim=-1)

        return coords

    def forward(self, x: Tensor, layer_idx: int = 0) -> Tensor:
        """Forward pass with factor generation."""
        # Get cached or generate factors
        layer = layer_idx % self.config.n_layers
        if layer not in self.factor_cache:
            latent = self.layer_latents[layer].unsqueeze(0).expand(x.shape[0], -1)
            coords = self._get_coords(layer, x.shape[1])

            l_factors = self.constructor.forward(latent, matrix_type=0, coords=coords)
            r_factors = self.constructor.forward(latent, matrix_type=1, coords=coords)
            self.factor_cache[layer] = (l_factors, r_factors)
        else:
            l_factors, r_factors = self.factor_cache[layer]

        # Apply gains
        gain = self.layer_gains[layer]

        # Compute: y = (x @ Rᵀ) @ Lᵀ summed over groups
        batch_size, seq_len, d_model = x.shape

        if self.config.constructor_groups == 1:
            # Single group computation
            y = torch.matmul(x, r_factors.squeeze(1).transpose(-1, -2))
            y = torch.matmul(y, l_factors.squeeze(1).transpose(-1, -2))
            return y.squeeze(-1) * gain
        else:
            # Multiple groups computation
            y = torch.zeros(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)
            for g in range(self.config.constructor_groups):
                r_t = r_factors[:, g].transpose(-1, -2)
                l_t = l_factors[:, g].transpose(-1, -2)
                y += torch.matmul(torch.matmul(x, r_t), l_t)
            return y.squeeze(-1) * gain
class FastTransformerBlock(nn.Module):
    """Transformer block using fast latent MLP."""

    def __init__(self, config: ModelConfig, constructor: MatrixGroupAwareConstructor):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.mlp = FastLatentMLP(config, constructor)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, layer_idx: int = 0) -> Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.dropout(x)

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x, layer_idx)
        x = residual + self.dropout(x)

        return x
def build_fast_model(config: ModelConfig) -> nn.Module:
    """Build the fast latent transformer model."""
    constructor = MatrixGroupAwareConstructor(config)

    blocks = nn.ModuleList([
        FastTransformerBlock(config, constructor) for _ in range(config.n_layers)
    ])

    norm = nn.LayerNorm(config.d_model)
    embed = nn.Embedding(config.vocab_size, config.d_model)
    pos_emb = nn.Parameter(torch.randn(config.context, config.d_model))

    class FastLatentTransformer(nn.Module):
        def __init__(self, config, constructor, blocks, norm, embed, pos_emb):
            super().__init__()
            self.config = config
            self.constructor = constructor
            self.blocks = blocks
            self.norm = norm
            self.embed = embed
            self.pos_emb = pos_emb

        def forward(self, x: Tensor, causal_mask: Optional[Tensor] = None) -> Tensor:
            # Embed tokens
            x = self.embed(x) + self.pos_emb[:x.shape[1]]

            # Pass through blocks
            for i, block in enumerate(self.blocks):
                x = block(x, i)

            # Final normalization
            x = self.norm(x)

            # Output projection
            logits = torch.matmul(x, self.embed.weight.t())
            return logits

        def generate_sample(self, prompt: Tensor, max_tokens: int = 100) -> Tensor:
            """Generate text with factor caching."""
            self.eval()
            with torch.no_grad():
                generated = prompt
                for _ in range(max_tokens):
                    out = self.forward(generated[:, -1:], None)
                    next_token = out.argmax(dim=-1)
                    generated = torch.cat([generated, next_token], dim=-1)
                    if _ % 20 == 0:
                        self.clear_factor_cache()
                return generated

        def clear_factor_cache(self):
            for block in self.blocks:
                block.mlp.factor_cache.clear()

    return FastLatentTransformer(config, constructor, blocks, norm, embed, pos_emb)
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
def train_fast_model(config: ModelConfig, args):
    """Train the fast latent transformer."""
    model = build_fast_model(config)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history = []
    start_time = time.time()

    for step in range(args.steps):
        optimizer.zero_grad()

        # Generate synthetic data
        input_ids = torch.randint(0, config.vocab_size, (args.batch_size, config.context), device=args.device)
        targets = input_ids.clone()

        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            rate = (step * args.batch_size * config.context) / max(elapsed, 1e-9)
            param_count = count_parameters(model)

            print(f"Step {step:6d} | Loss {loss.item():.4f} | {rate:,.0f} tok/s")
            print(f"  Params: {param_count['total']:,} | Trainable: {param_count['trainable']:,}")

            history.append({
                "step": step,
                "train_loss": loss.item(),
                "elapsed_seconds": elapsed,
                "tokens_per_second": rate,
                "parameters": param_count,
            })

        # Evaluation
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for _ in range(5):
                    eval_ids = torch.randint(0, config.vocab_size, (args.batch_size, config.context), device=args.device)
                    eval_targets = eval_ids.clone()
                    eval_logits = model(eval_ids)
                    eval_loss += criterion(eval_logits.view(-1, config.vocab_size), eval_targets.view(-1))

            eval_loss /= 5
            print(f"Eval {step:6d} | Val Loss {eval_loss:.4f}")
            history.append({"step": step, "val_loss": eval_loss.item()})

            model.train()

    # Save model
    run_dir = Path(f"runs/fast_latent_L{config.n_layers}_D{config.d_model}_R{config.rank}_Z{config.latent_dim}")
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_config": config.to_dict(),
        "model_state": model.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, run_dir / "model.pt")

    # Generate sample
    sample_input = torch.randint(0, config.vocab_size, (1, 10), device=args.device)
    sample_output = model.generate_sample(sample_input, max_tokens=50)
    sample_text = sample_output[0].cpu().tolist()

    (run_dir / "sample.txt").write_text(str(sample_text))

    print(f"\nModel saved to {run_dir}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training time: {time.time() - start_time:.1f}s")

    return model, history
def parse_args():
    parser = argparse.ArgumentParser(description="Train fast latent transformer")
    parser.add_argument("--data", type=Path, help="Training data path")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    print("=== Fast Latent Transformer Training ===")
    print(f"Steps: {args.steps}")
    print(f"Rank: {args.rank}")
    print(f"Latent Dim: {args.latent_dim}")
    print(f"Constructor Groups: 4")
    print()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Create config
    config = ModelConfig(
        vocab_size=256,
        context=args.context,
        n_layers=args.layers,
        d_model=args.d_model,
        n_heads=args.heads,
        d_ff=args.d_ff,
        mode="latent",
        rank=args.rank,
        latent_dim=args.latent_dim,
    )

    # Train model
    model, history = train_fast_model(config, args)

    print("\n=== Training Complete ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Final training loss: {history[-1]['train_loss']:.4f}")

    return model, history
if __name__ == "__main__":
    main()
