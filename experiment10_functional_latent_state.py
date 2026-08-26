#!/usr/bin/env python3
"""
Experiment 10: functional factor -> latent state.

Takes Qwen2-0.5B layer-0 q_proj, learns an activation-aware low-rank
factorization, then compresses the factor pair into a small latent.
Measures:
  - latent coordinates and statistics
  - latent population geometry
  - reconstructed-weight error
  - held-out Qwen behavior after injecting reconstructed weights
  - storage accounting

Run from the directory containing Qwen2-0.5B and Experiment 9 files.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from experiment9_round2_validation import PROBES, assess, forced_logits, greedy
from experiment9_weight_latent_functional import (
    TARGET_WEIGHT, inject_weight, load_model, load_safetensor_weight,
    numerical_metrics, set_seed, weight_statistics,
)

CALIBRATION = (
    "Explain gravity to a curious child.",
    "Write a short haiku about rain.",
    "The purpose of a compiler is to",
    "Describe the difference between a stack and a queue.",
    "A reliable experiment should include",
    "The Pythagorean theorem states that",
    "A REST API commonly uses HTTP methods such as",
    "The moon affects ocean tides because",
    "An algorithm is efficient when it",
    "The main purpose of version control is to",
    "A dictionary in Python maps keys to",
    "The human heart pumps blood through",
    "A good error message should tell the user",
    "The area of a circle is",
    "A test set should not be used to",
)

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("./Qwen2-0.5B"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--ranks", default="64,128")
    p.add_argument("--latents", default="8,16,32,64,128")
    p.add_argument("--seeds", default="9,10,11")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--continuation-tokens", type=int, default=6)
    p.add_argument("--output-dir", type=Path,
                   default=Path("experiment10_functional_latent_results"))
    return p.parse_args()

def tokenize(tok, prompts):
    return [tok(x, add_special_tokens=False)["input_ids"] for x in prompts]

@torch.inference_mode()
def capture_q_inputs(model, ids, device):
    module = model.get_submodule(TARGET_WEIGHT.removesuffix(".weight"))
    out = []
    def hook(_m, inputs):
        out.append(inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1]))
    h = module.register_forward_pre_hook(hook)
    try:
        for ids_one in ids:
            model(input_ids=torch.tensor([ids_one], dtype=torch.long, device=device))
    finally:
        h.remove()
    return torch.cat(out)

def learn_factors(W, X, rank, steps, lr, device):
    u, s, vh = torch.linalg.svd(W.float(), full_matrices=False)
    L = nn.Parameter((u[:, :rank] * s[:rank]).to(device))
    R = nn.Parameter(vh[:rank].to(device))
    Xd, Wd = X.to(device), W.float().to(device)
    target = Xd @ Wd.T
    opt = torch.optim.AdamW((L, R), lr=lr, weight_decay=0)
    for step in range(1, steps + 1):
        n = min(512, Xd.shape[0])
        idx = torch.randint(Xd.shape[0], (n,), device=device)
        pred = (Xd[idx] @ R.T) @ L.T
        loss = F.mse_loss(pred, target[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step == steps or step % 200 == 0:
            print(f"  factor {step:4d}/{steps} | activation MSE {loss.item():.8g}")
    return L.detach().cpu(), R.detach().cpu()

class AE(nn.Module):
    def __init__(self, n, z):
        super().__init__()
        h = min(512, max(64, n // 2))
        self.enc = nn.Sequential(nn.Linear(n, h), nn.GELU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.GELU(), nn.Linear(h, n))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)

def factor_vector(L, R):
    return torch.cat((L.float().flatten(), R.float().flatten()))

def split_vector(v, Lshape, Rshape):
    n = 1
    for d in Lshape: n *= d
    return v[:n].reshape(Lshape), v[n:].reshape(Rshape)

def train_latent(v, zdim, steps, lr, seed):
    set_seed(seed)
    mean, std = v.mean(), v.std().clamp_min(1e-8)
    x = ((v - mean) / std).unsqueeze(0)
    m = AE(v.numel(), zdim)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0)
    for step in range(1, steps + 1):
        z = m.encode(x)
        y = m.decode(z)
        loss = F.mse_loss(y, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step == steps or step % 200 == 0:
            print(f"  latent {step:4d}/{steps} | factor MSE {loss.item():.8g}")
    return m, mean, std

@torch.inference_mode()
def latent_and_reconstruction(m, mean, std, v):
    x = ((v - mean) / std).unsqueeze(0)
    z = m.encode(x).squeeze(0)
    y = m.decode(z.unsqueeze(0)).squeeze(0) * std + mean
    return z.cpu(), y.cpu()

def sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

def latent_stats(Z):
    Z = torch.stack(Z).float()
    mu, sd = Z.mean(0), Z.std(0)
    C = torch.cov(Z.T) if Z.shape[0] > 1 else torch.zeros((Z.shape[1], Z.shape[1]))
    eig = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)
    total = eig.sum().item()
    ratio = (eig / total) if total else torch.zeros_like(eig)
    eff = float((eig.sum()**2 / (eig.square().sum() + 1e-12)))
    return {
        "samples": Z.shape[0], "dimension": Z.shape[1],
        "mean": mu.tolist(), "std": sd.tolist(),
        "min": Z.min(0).values.tolist(), "max": Z.max(0).values.tolist(),
        "eigenvalues": eig.tolist(), "variance_ratio": ratio.tolist(),
        "effective_rank": eff,
    }

def main():
    a = args()
    device = torch.device(a.device)
    ranks = [int(x) for x in a.ranks.split(",") if x]
    latents = [int(x) for x in a.latents.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]
    a.output_dir.mkdir(parents=True, exist_ok=True)

    W = load_safetensor_weight(a.model_dir / "model.safetensors")
    tok = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    cal_ids = tokenize(tok, CALIBRATION)
    eval_ids = tokenize(tok, PROBES)

    print("=" * 88)
    print("EXPERIMENT 10 — FUNCTIONAL FACTOR -> LATENT STATE")
    print("=" * 88)
    print(f"Target: {TARGET_WEIGHT}")
    print(f"Shape: {tuple(W.shape)}  dtype: {W.dtype}")
    print(f"SHA256: {weight_statistics(W)['sha256']}")
    print(f"Ranks: {ranks} | Latents: {latents} | Seeds: {seeds}")

    ref = load_model(a.model_dir, device, W.dtype)
    X = capture_q_inputs(ref, cal_ids, device)
    baseline_cont = greedy(ref, eval_ids, a.continuation_tokens,
                           tok.pad_token_id, device)
    baseline_logits = forced_logits(ref, eval_ids, baseline_cont,
                                    tok.pad_token_id, device)
    del ref
    print(f"Captured calibration vectors: {X.shape[0]} x {X.shape[1]}")

    results = []
    for rank in ranks:
        print(f"\n{'-'*88}\nFUNCTIONAL FACTOR RANK {rank}\n{'-'*88}")
        L, R = learn_factors(W, X, rank, a.steps, a.lr, device)
        v = factor_vector(L, R)
        print(f"Factor values: {v.numel():,}")

        for zdim in latents:
            population = []
            for seed in seeds:
                print(f"\nRANK {rank} | LATENT {zdim} | SEED {seed}")
                m, mean, std = train_latent(v, zdim, a.steps, a.lr, seed)
                z, rv = latent_and_reconstruction(m, mean, std, v)
                population.append(z)

                L2, R2 = split_vector(rv, tuple(L.shape), tuple(R.shape))
                W2 = (L2 @ R2).to(W.dtype).contiguous()
                num = numerical_metrics(W, W2)

                clone = load_model(a.model_dir, device, W.dtype)
                inject_weight(clone, W2)
                fun = assess(clone, eval_ids, baseline_cont, baseline_logits,
                             tok.pad_token_id, device)
                del clone
                if device.type == "cuda": torch.cuda.empty_cache()

                dec_params = sum(p.numel() for p in m.parameters())
                latent_bytes = zdim * 4
                package_bytes = latent_bytes + dec_params * 4

                row = {
                    "rank": rank, "latent_dim": zdim, "seed": seed,
                    "numerical": num, "functional": fun,
                    "latent": {
                        "values": z.tolist(), "mean": float(z.mean()),
                        "std": float(z.std()), "min": float(z.min()),
                        "max": float(z.max()), "sha256": sha(z),
                    },
                    "storage": {
                        "original_weight_bytes_bfloat16": W.numel() * 2,
                        "latent_bytes_float32": latent_bytes,
                        "decoder_bytes_float32": dec_params * 4,
                        "latent_plus_decoder_bytes": package_bytes,
                    },
                }
                results.append(row)

                print("LATENT:", [round(x, 6) for x in z.tolist()])
                print(f"weight rel error: {num['relative_frobenius_error']:.6f}")
                print(f"greedy match:     {fun['greedy_token_match_rate']:.4f}")
                print(f"exact continuations: {fun['greedy_exact_continuation_count']}")
                print(f"logit RMSE:       {fun['logit_rmse']:.6f}")

            pop = latent_stats(population)
            print(f"LATENT POPULATION {zdim}: effective rank={pop['effective_rank']:.4f}")
            print("variance ratio:", [round(x, 5) for x in pop["variance_ratio"]])

    report = {
        "experiment": "Experiment 10: functional factor to latent state",
        "target": TARGET_WEIGHT,
        "weight_sha256": weight_statistics(W)["sha256"],
        "calibration_prompts": list(CALIBRATION),
        "evaluation_prompts": list(PROBES),
        "configuration": {
            "ranks": ranks, "latents": latents, "seeds": seeds,
            "steps": a.steps, "learning_rate": a.lr,
        },
        "results": results,
    }
    out = a.output_dir / "summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print("rank latent seed | rel-error | greedy | exact | logit-RMSE")
    for r in results:
        f = r["functional"]
        print(f"{r['rank']:4d} {r['latent_dim']:6d} {r['seed']:4d} | "
              f"{r['numerical']['relative_frobenius_error']:.6f} | "
              f"{f['greedy_token_match_rate']:.4f} | "
              f"{f['greedy_exact_continuation_count']:5d} | "
              f"{f['logit_rmse']:.6f}")
    print(f"\nResults: {out}")

if __name__ == "__main__":
    main()
