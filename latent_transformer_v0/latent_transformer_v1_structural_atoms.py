#!/usr/bin/env python3
"""Latent Transformer v1: shared sparse structural-atom constructor.

Isolated experimental variant of latent_transformer_v0. The byte-level LM,
attention, optimizer, data split, and training loop are inherited unchanged.
Only the latent MLP constructor changes.

Each matrix-specific latent selects top-k entries from a shared learned atom
dictionary. A shared coordinate-conditioned generator turns selected atoms
into factor rows/columns. Dense matrices are never stored.

Run from this directory:
  python latent_transformer_v1_structural_atoms.py --data corpus.txt --mode structural
  python latent_transformer_v1_structural_atoms.py --smoke-test --mode structural \
    --layers 1 --d-model 48 --heads 4 --d-ff 96 --rank 8 --context 32 \
    --batch-size 2 --steps 2 --eval-every 1 --eval-batches 1 --device cpu

Custom options:
  --atom-count 32 --atom-dim 16 --atom-topk 4 --atom-temperature 1.0
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import latent_transformer_v0 as base

ATOM_DEFAULTS = {
    "atom_count": 32,
    "atom_dim": 16,
    "atom_topk": 4,
    "atom_temperature": 1.0,
}
ACTIVE_ATOM_CONFIG = dict(ATOM_DEFAULTS)
LAST_MODEL = None


def parse_args_with_atoms():
    """Consume v1-only flags, then delegate common flags to v0."""
    original_argv = list(sys.argv)
    argv = list(sys.argv[1:])
    values = dict(ATOM_DEFAULTS)
    flags = {
        "--atom-count": ("atom_count", int),
        "--atom-dim": ("atom_dim", int),
        "--atom-topk": ("atom_topk", int),
        "--atom-temperature": ("atom_temperature", float),
    }
    cleaned = []
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = False
        for flag, (key, cast) in flags.items():
            if token == flag:
                if i + 1 >= len(argv):
                    raise SystemExit(f"{flag} requires a value")
                values[key] = cast(argv[i + 1])
                i += 2
                matched = True
                break
            if token.startswith(flag + "="):
                values[key] = cast(token.split("=", 1)[1])
                i += 1
                matched = True
                break
        if not matched:
            cleaned.append(token)
            i += 1

    for i, token in enumerate(cleaned):
        if token == "--mode" and i + 1 < len(cleaned) and cleaned[i + 1] == "structural":
            cleaned[i + 1] = "latent"
        elif token == "--mode=structural":
            cleaned[i] = "--mode=latent"

    sys.argv = [original_argv[0], *cleaned]
    try:
        args = base.parse_args()
    finally:
        sys.argv = original_argv

    args.mode = "structural"
    for key, value in values.items():
        setattr(args, key, value)
    if args.atom_count < 2:
        raise SystemExit("--atom-count must be >= 2")
    if not 1 <= args.atom_topk <= args.atom_count:
        raise SystemExit("--atom-topk must be between 1 and --atom-count")
    if args.atom_dim < 1 or args.atom_temperature <= 0:
        raise SystemExit("--atom-dim and --atom-temperature must be positive")
    return args


class StructuralAtomConstructor(nn.Module):
    """Shared dictionary plus sparse, coordinate-conditioned atom generator."""
    def __init__(self, latent_dim, rank, hidden, type_dim, fourier_bands,
                 atom_count, atom_dim, atom_topk, atom_temperature):
        super().__init__()
        self.rank = rank
        self.fourier_bands = fourier_bands
        self.atom_count = atom_count
        self.atom_topk = atom_topk
        self.atom_temperature = atom_temperature
        self.atom_embedding = nn.Embedding(atom_count, atom_dim)
        self.type_embed = nn.Embedding(len(base.MATRIX_TYPES), type_dim)
        self.selector = nn.Linear(latent_dim, atom_count)
        coord_dim = 1 + 2 * fourier_bands
        in_dim = atom_dim + type_dim + coord_dim

        def make_net():
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
        nn.init.normal_(self.atom_embedding.weight, std=0.02)
        nn.init.normal_(self.type_embed.weight, std=0.02)

    def coord_features(self, n, *, device, dtype):
        x = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype).unsqueeze(1)
        feats = [x]
        for band in range(self.fourier_bands):
            frequency = math.pi * (2.0 ** band)
            feats.extend((torch.sin(frequency * x), torch.cos(frequency * x)))
        return torch.cat(feats, dim=1)

    def selected_atoms(self, z):
        logits = self.selector(z)
        values, indices = torch.topk(logits, k=self.atom_topk, dim=-1)
        weights = F.softmax(values / self.atom_temperature, dim=-1)
        return indices, weights

    def atom_inputs(self, atom_index, type_id, n, device, dtype):
        atom = self.atom_embedding(atom_index).to(dtype).reshape(1, -1).expand(n, -1)
        type_index = torch.tensor(type_id, device=device, dtype=torch.long)
        type_vec = self.type_embed(type_index).to(dtype).reshape(1, -1).expand(n, -1)
        coords = self.coord_features(n, device=device, dtype=dtype)
        return torch.cat((atom, type_vec, coords), dim=-1)

    def factors(self, z, type_id, in_features, out_features):
        indices, weights = self.selected_atoms(z)
        row_parts, col_parts = [], []
        for slot in range(self.atom_topk):
            atom_index = indices[slot]
            coefficient = weights[slot]
            row_input = self.atom_inputs(atom_index, type_id, out_features, z.device, z.dtype)
            col_input = self.atom_inputs(atom_index, type_id, in_features, z.device, z.dtype)
            row_parts.append(coefficient * self.row_net(row_input))
            col_parts.append(coefficient * self.col_net(col_input))
        left = torch.stack(row_parts, dim=0).sum(dim=0)
        right_columns = torch.stack(col_parts, dim=0).sum(dim=0)
        scale = 1.0 / math.sqrt(max(1, self.rank))
        return left * scale, right_columns.transpose(0, 1).contiguous()

    @torch.no_grad()
    def usage_summary(self, latents):
        counts = torch.zeros(self.atom_count, dtype=torch.long)
        matrix_count = 0
        for z in latents:
            indices, _ = self.selected_atoms(z.detach())
            counts[indices.cpu()] += 1
            matrix_count += 1
        total = int(counts.sum())
        probs = counts.float() / max(1, total)
        nonzero = probs[probs > 0]
        entropy = float((-(nonzero * nonzero.log()).sum()).item()) if nonzero.numel() else 0.0
        return {
            "atom_count": self.atom_count,
            "topk_per_matrix": self.atom_topk,
            "matrix_latent_count": matrix_count,
            "total_atom_selections": total,
            "used_atom_count": int((counts > 0).sum()),
            "selection_counts": counts.tolist(),
            "selection_entropy_nats": entropy,
            "maximum_entropy_nats": math.log(self.atom_count),
        }


class StructuralAtomLinear(nn.Module):
    """A matrix-specific latent selecting sparse components from shared atoms."""
    def __init__(self, in_features, out_features, latent_dim, type_id):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.type_id = int(type_id)
        self.z = nn.Parameter(torch.randn(latent_dim) * 0.02)
        self.log_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x, constructor):
        left, right = constructor.factors(
            self.z, self.type_id, self.in_features, self.out_features)
        return F.linear(F.linear(x, right), left) * self.log_gain.exp()


class StructuralAtomMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = StructuralAtomLinear(
            cfg.d_model, cfg.d_ff, cfg.latent_dim, base.MATRIX_TYPES["gate"])
        self.up = StructuralAtomLinear(
            cfg.d_model, cfg.d_ff, cfg.latent_dim, base.MATRIX_TYPES["up"])
        self.down = StructuralAtomLinear(
            cfg.d_ff, cfg.d_model, cfg.latent_dim, base.MATRIX_TYPES["down"])

    def forward(self, x, constructor):
        return self.down(
            F.silu(self.gate(x, constructor)) * self.up(x, constructor),
            constructor,
        )


class StructuralAtomTinyLM(base.TinyLM):
    def __init__(self, cfg):
        global LAST_MODEL
        latent_cfg = replace(cfg, mode="latent")
        super().__init__(latent_cfg)
        self.cfg = cfg
        self.constructor = StructuralAtomConstructor(
            latent_dim=cfg.latent_dim,
            rank=cfg.rank,
            hidden=cfg.constructor_hidden,
            type_dim=cfg.constructor_type_dim,
            fourier_bands=cfg.constructor_fourier_bands,
            **ACTIVE_ATOM_CONFIG,
        )
        for block in self.blocks:
            block.mlp = StructuralAtomMLP(cfg)
        LAST_MODEL = self

    def structural_usage(self):
        latents = [module.z for module in self.modules()
                   if isinstance(module, StructuralAtomLinear)]
        return self.constructor.usage_summary(latents)


def train_with_structural_report(args):
    global ACTIVE_ATOM_CONFIG
    ACTIVE_ATOM_CONFIG = {
        "atom_count": args.atom_count,
        "atom_dim": args.atom_dim,
        "atom_topk": args.atom_topk,
        "atom_temperature": args.atom_temperature,
    }
    run_name = args.run_name or (
        f"structural_L{args.layers}_D{args.d_model}_R{args.rank}_"
        f"Z{args.latent_dim}_A{args.atom_count}_K{args.atom_topk}_seed{args.seed}"
    )
    args.run_name = run_name
    original_train(args)
    report_path = args.output_dir / run_name / "report.json"
    report = json.loads(report_path.read_text())
    report["experiment"] = "latent_transformer_v1_structural_atoms"
    report["architecture_variant"] = "shared_sparse_structural_atom_dictionary"
    report["structural_atom_config"] = ACTIVE_ATOM_CONFIG
    report["structural_atom_usage"] = LAST_MODEL.structural_usage()
    report_path.write_text(json.dumps(report, indent=2) + "\n")


base.TinyLM = StructuralAtomTinyLM
base.parse_args = parse_args_with_atoms
original_train = base.train
base.train = train_with_structural_report

if __name__ == "__main__":
    base.main()
