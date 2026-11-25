#!/usr/bin/env python3
"""
Latent Transformer v0
=====================
A tiny decoder-only language model for comparing three MLP parameterizations:

  dense    : ordinary dense MLP matrices
  lowrank  : each MLP matrix owns trainable L/R factors
  latent   : each MLP matrix owns only a small latent code; ONE shared
             coordinate constructor generates its L/R factors on demand

The latent model is the main experiment. Its MLP matrices are never learned or
serialized as independent dense tensors.

This script intentionally uses byte-level tokens (vocab=256) so the first test
has no tokenizer dependency. Give it any UTF-8 text file.

Example:
  python latent_transformer_v0.py --data corpus.txt --mode latent --device auto

For a quick plumbing test:
  python latent_transformer_v0.py --smoke-test --mode latent --steps 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    mode: str = "latent"
    rank: int = 64
    latent_dim: int = 32
    constructor_hidden: int = 128
    constructor_type_dim: int = 16
    constructor_fourier_bands: int = 8


MATRIX_TYPES = {"gate": 0, "up": 1, "down": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dense/low-rank/latent tiny LMs")
    p.add_argument("--data", type=Path, default=None, help="UTF-8 training text")
    p.add_argument("--val-data", type=Path, default=None,
                   help="Optional separate UTF-8 validation text; prevents sequential split leakage")
    p.add_argument("--smoke-test", action="store_true",
                   help="Use a tiny built-in corpus to verify the code path")
    p.add_argument("--mode", choices=("dense", "lowrank", "latent"), default="latent")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    p.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--constructor-hidden", type=int, default=128)
    p.add_argument("--constructor-type-dim", type=int, default=16)
    p.add_argument("--constructor-fourier-bands", type=int, default=8)

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)

    p.add_argument("--output-dir", type=Path, default=Path("runs/latent_transformer_v0"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--generate-prompt", default="The ")
    p.add_argument("--generate-tokens", type=int, default=160)
    return p.parse_args()


SMOKE_CORPUS = ("""
The small model learns patterns from text. A model predicts the next byte.
Compression is useful when shared structure can replace repeated information.
The rain falls on the garden. The sun warms the soil. Plants grow toward light.
A good experiment compares a new method with a simple baseline.
Language contains repeated structures, relationships, and context.
""" * 300).encode("utf-8")


def load_bytes(path: Optional[Path], smoke: bool) -> Tensor:
    if smoke:
        raw = SMOKE_CORPUS
    else:
        if path is None:
            raise SystemExit("Pass --data FILE, or use --smoke-test.")
        raw = path.read_bytes()
    if len(raw) < 4096:
        raise SystemExit(f"Corpus is too small ({len(raw):,} bytes); need at least 4 KiB.")
    return torch.tensor(list(raw), dtype=torch.long)


def split_data(data: Tensor, fraction: float = 0.9) -> Tuple[Tensor, Tensor]:
    cut = max(1, min(data.numel() - 1, int(data.numel() * fraction)))
    return data[:cut], data[cut:]


def sample_batch(data: Tensor, batch_size: int, context: int, device: torch.device,
                 generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
    if data.numel() <= context + 1:
        raise ValueError(f"split has {data.numel()} bytes but context={context}")
    starts = torch.randint(0, data.numel() - context - 1, (batch_size,),
                           generator=generator)
    x = torch.stack([data[i:i + context] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts.tolist()])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class LowRankLinear(nn.Module):
    """y = (x @ R.T) @ L.T, with W = L @ R."""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        rank = min(rank, in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.R = nn.Parameter(torch.empty(rank, in_features))
        self.L = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.R, a=math.sqrt(5))
        nn.init.normal_(self.L, mean=0.0, std=0.02 / math.sqrt(max(1, rank)))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.R), self.L)


class CoordinateFactorConstructor(nn.Module):
    """
    ONE shared hypernetwork for all latent MLP matrices.

    It never has an output layer proportional to matrix size. Instead it maps
    (latent code, matrix-type embedding, coordinate Fourier features) -> one
    rank-dimensional row/column at a time. Thus constructor parameter count is
    independent of d_model, d_ff, and layer count.
    """
    def __init__(self, latent_dim: int, rank: int, hidden: int,
                 type_dim: int, fourier_bands: int, n_types: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        self.fourier_bands = fourier_bands
        self.type_embed = nn.Embedding(n_types, type_dim)
        coord_dim = 1 + 2 * fourier_bands
        in_dim = latent_dim + type_dim + coord_dim

        def make_net() -> nn.Sequential:
            net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, rank),
            )
            nn.init.normal_(net[-1].weight, std=0.002)
            nn.init.zeros_(net[-1].bias)
            return net

        self.row_net = make_net()
        self.col_net = make_net()

    def coord_features(self, n: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        x = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype).unsqueeze(1)
        feats = [x]
        for k in range(self.fourier_bands):
            f = math.pi * (2.0 ** k)
            feats.extend((torch.sin(f * x), torch.cos(f * x)))
        return torch.cat(feats, dim=1)

    def _inputs(self, z: Tensor, type_id: int, n: int) -> Tensor:
        dtype = z.dtype
        device = z.device
        coords = self.coord_features(n, device=device, dtype=dtype)
        z_rep = z.reshape(1, -1).expand(n, -1)
        type_tensor = torch.tensor(type_id, device=device, dtype=torch.long)
        e = self.type_embed(type_tensor).to(dtype).reshape(1, -1).expand(n, -1)
        return torch.cat((z_rep, e, coords), dim=1)

    def factors(self, z: Tensor, type_id: int,
                in_features: int, out_features: int) -> Tuple[Tensor, Tensor]:
        L = self.row_net(self._inputs(z, type_id, out_features))
        R_cols = self.col_net(self._inputs(z, type_id, in_features))
        R = R_cols.transpose(0, 1).contiguous()
        scale = 1.0 / math.sqrt(max(1, self.rank))
        return L * scale, R


class LatentLinear(nn.Module):
    """A matrix whose only matrix-specific trainable state is z (+ one gain)."""
    def __init__(self, in_features: int, out_features: int,
                 latent_dim: int, type_id: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_id = int(type_id)
        self.z = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.log_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, constructor: CoordinateFactorConstructor) -> Tensor:
        L, R = constructor.factors(self.z, self.type_id, self.in_features, self.out_features)
        return F.linear(F.linear(x, R), L) * self.log_gain.exp()


class DenseMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor, constructor=None) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LowRankMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = LowRankLinear(cfg.d_model, cfg.d_ff, cfg.rank)
        self.up = LowRankLinear(cfg.d_model, cfg.d_ff, cfg.rank)
        self.down = LowRankLinear(cfg.d_ff, cfg.d_model, cfg.rank)

    def forward(self, x: Tensor, constructor=None) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LatentMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = LatentLinear(cfg.d_model, cfg.d_ff, cfg.latent_dim, MATRIX_TYPES["gate"])
        self.up = LatentLinear(cfg.d_model, cfg.d_ff, cfg.latent_dim, MATRIX_TYPES["up"])
        self.down = LatentLinear(cfg.d_ff, cfg.d_model, cfg.latent_dim, MATRIX_TYPES["down"])

    def forward(self, x: Tensor, constructor: CoordinateFactorConstructor) -> Tensor:
        return self.down(F.silu(self.gate(x, constructor)) * self.up(x, constructor), constructor)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        if cfg.mode == "dense":
            self.mlp = DenseMLP(cfg)
        elif cfg.mode == "lowrank":
            self.mlp = LowRankMLP(cfg)
        elif cfg.mode == "latent":
            self.mlp = LatentMLP(cfg)
        else:
            raise ValueError(cfg.mode)

    def forward(self, x: Tensor, constructor: Optional[CoordinateFactorConstructor]) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x), constructor)
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token.weight

        self.constructor: Optional[CoordinateFactorConstructor]
        if cfg.mode == "latent":
            self.constructor = CoordinateFactorConstructor(
                latent_dim=cfg.latent_dim,
                rank=cfg.rank,
                hidden=cfg.constructor_hidden,
                type_dim=cfg.constructor_type_dim,
                fourier_bands=cfg.constructor_fourier_bands,
            )
        else:
            self.constructor = None

        self.apply(self._init_weights)
        if self.constructor is not None:
            for net in (self.constructor.row_net, self.constructor.col_net):
                nn.init.normal_(net[-1].weight, std=0.002)
                nn.init.zeros_(net[-1].bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        _, t = idx.shape
        if t > self.cfg.context:
            raise ValueError(f"sequence length {t} exceeds context {self.cfg.context}")
        pos = torch.arange(t, device=idx.device)
        x = self.token(idx) + self.position(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, self.constructor)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            crop = idx[:, -self.cfg.context:]
            logits, _ = self(crop)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        self.train(was_training)
        return idx


def count_parameters(model: nn.Module) -> Dict[str, int]:
    groups = {"total": 0, "constructor": 0, "latents": 0, "mlp": 0, "attention": 0, "other": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        groups["total"] += n
        if "constructor" in name:
            groups["constructor"] += n
        elif name.endswith(".z") or name.endswith(".log_gain"):
            groups["latents"] += n
        elif ".mlp." in name:
            groups["mlp"] += n
        elif ".attn." in name:
            groups["attention"] += n
        else:
            groups["other"] += n
    return groups


def state_dict_bytes(model: nn.Module) -> int:
    seen = set()
    total = 0
    for t in model.state_dict().values():
        ptr = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        if ptr not in seen:
            seen.add(ptr)
            total += t.untyped_storage().nbytes()
    return total


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} GiB"


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if total <= warmup:
        return peak
    ratio = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * ratio)))


@torch.no_grad()
def evaluate(model: TinyLM, data: Tensor, batch_size: int, context: int,
             batches: int, device: torch.device,
             generator: Optional[torch.Generator] = None) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = sample_batch(data, batch_size, context, device, generator)
        _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def peak_memory_bytes(device: torch.device) -> Optional[int]:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def train(a: argparse.Namespace) -> None:
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    # Separate seeded CPU RNG streams isolate data sampling from model initialization.
    train_generator = torch.Generator(device="cpu").manual_seed(a.seed + 1)
    eval_generator = torch.Generator(device="cpu").manual_seed(a.seed + 2)

    device = choose_device(a.device)
    dtype = dtype_from_name(a.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        print("WARNING: float16 on CPU is usually slow/unsupported; using float32 instead.")
        dtype = torch.float32

    cfg = ModelConfig(
        context=a.context,
        n_layers=a.layers,
        d_model=a.d_model,
        n_heads=a.heads,
        d_ff=a.d_ff,
        dropout=a.dropout,
        mode=a.mode,
        rank=a.rank,
        latent_dim=a.latent_dim,
        constructor_hidden=a.constructor_hidden,
        constructor_type_dim=a.constructor_type_dim,
        constructor_fourier_bands=a.constructor_fourier_bands,
    )
    if cfg.rank > min(cfg.d_model, cfg.d_ff):
        raise SystemExit("--rank must be <= min(d_model, d_ff)")

    data = load_bytes(a.data, a.smoke_test)
    if a.val_data is not None:
        if a.smoke_test:
            raise SystemExit("--val-data cannot be combined with --smoke-test")
        train_data = data
        val_data = load_bytes(a.val_data, smoke=False)
    else:
        train_data, val_data = split_data(data)
    if min(train_data.numel(), val_data.numel()) <= cfg.context + 1:
        raise SystemExit("Corpus/split is too small for requested --context")

    model = TinyLM(cfg).to(device=device, dtype=dtype)
    counts = count_parameters(model)
    model_bytes = state_dict_bytes(model)

    run_name = a.run_name or f"{a.mode}_L{a.layers}_D{a.d_model}_R{a.rank}_Z{a.latent_dim}_seed{a.seed}"
    run_dir = a.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("LATENT TRANSFORMER v0")
    print("=" * 88)
    print(f"mode:          {a.mode}")
    print(f"device/dtype:  {device} / {dtype}")
    print(f"corpus bytes:  {data.numel():,} (train {train_data.numel():,}, val {val_data.numel():,})")
    print(f"parameters:    {counts['total']:,}")
    print(f"state bytes:   {model_bytes:,} ({human_bytes(model_bytes)})")
    print(f"breakdown:     {counts}")
    print(f"run dir:       {run_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    reset_peak_memory(device)
    history = []
    tokens_seen = 0
    start = time.perf_counter()
    last_log = start
    last_log_tokens = 0

    for step in range(a.steps):
        lr = lr_at(step, a.steps, a.warmup_steps, a.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(a.grad_accum):
            x, y = sample_batch(train_data, a.batch_size, cfg.context, device,
                                train_generator)
            _, loss = model(x, y)
            (loss / a.grad_accum).backward()
            total_loss += float(loss.detach())
            tokens_seen += x.numel()

        if a.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        optimizer.step()

        one_based = step + 1
        if one_based == 1 or one_based % a.log_every == 0:
            now = time.perf_counter()
            dt = max(1e-9, now - last_log)
            rate = (tokens_seen - last_log_tokens) / dt
            print(f"step {one_based:6d}/{a.steps} | train {total_loss/a.grad_accum:.4f} | "
                  f"lr {lr:.3g} | {rate:,.0f} tok/s")
            last_log, last_log_tokens = now, tokens_seen

        if one_based == 1 or one_based % a.eval_every == 0 or one_based == a.steps:
            val_loss = evaluate(model, val_data, a.batch_size, cfg.context,
                                a.eval_batches, device, eval_generator)
            elapsed = time.perf_counter() - start
            row = {
                "step": one_based,
                "tokens_seen": tokens_seen,
                "train_loss": total_loss / a.grad_accum,
                "val_loss": val_loss,
                "perplexity": math.exp(min(20.0, val_loss)),
                "elapsed_seconds": elapsed,
                "tokens_per_second_overall": tokens_seen / max(elapsed, 1e-9),
                "peak_device_memory_bytes": peak_memory_bytes(device),
            }
            history.append(row)
            print(f"  EVAL val {val_loss:.4f} | ppl {row['perplexity']:.2f} | "
                  f"overall {row['tokens_per_second_overall']:,.0f} tok/s")

    prompt = a.generate_prompt.encode("utf-8", errors="replace")
    prompt_ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    generated = model.generate(prompt_ids, a.generate_tokens)[0].detach().cpu().tolist()
    sample_text = bytes(generated).decode("utf-8", errors="replace")

    checkpoint = {
        "model_config": asdict(cfg),
        "model_state": model.state_dict(),
        "training_args": vars(a) | {"data": str(a.data) if a.data else None,
                                    "val_data": str(a.val_data) if a.val_data else None,
                                    "output_dir": str(a.output_dir)},
        "history": history,
    }
    checkpoint_path = run_dir / "model.pt"
    torch.save(checkpoint, checkpoint_path)

    report = {
        "experiment": "latent_transformer_v0",
        "model_config": asdict(cfg),
        "mode": a.mode,
        "seed": a.seed,
        "device": str(device),
        "dtype": str(dtype),
        "corpus_bytes": int(data.numel()),
        "parameter_counts": counts,
        "state_dict_bytes_before_save": model_bytes,
        "checkpoint_file_bytes": checkpoint_path.stat().st_size,
        "optimizer_parameter_elements": sum(p.numel() for g in optimizer.param_groups for p in g["params"]),
        "history": history,
        "sample": sample_text,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (run_dir / "sample.txt").write_text(sample_text, errors="replace")

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"checkpoint: {checkpoint_path} ({human_bytes(checkpoint_path.stat().st_size)})")
    print(f"report:     {run_dir / 'report.json'}")
    print("\nSAMPLE:\n" + sample_text)


def main() -> None:
    a = parse_args()
    if a.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if a.batch_size < 1 or a.grad_accum < 1:
        raise SystemExit("batch size and grad accumulation must be >= 1")
    train(a)


if __name__ == "__main__":
    main()
