#!/usr/bin/env python3
"""
EXPERIMENT 10 — FUNCTIONAL FACTOR -> LATENT STATE

Fully standalone. No Experiment 9 imports.

Pipeline:
    Qwen Q-projection weight W
        |
        v
    activation-aware low-rank factors L,R
        |
        v
    compact latent Z
        |
        v
    reconstructed factors L',R'
        |
        v
    reconstructed W'
        |
        v
    cloned Qwen with W' injected
        |
        v
    held-out behavior comparison

The experiment also prints the actual latent coordinates and analyzes
their geometry across independent training seeds.

The original checkpoint is never modified.

Expected layout:
    latent_model_test/
        experiment10_functional_latent_state.py
        Qwen2-0.5B/
            model.safetensors
            config.json
            tokenizer files...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file


TARGET_WEIGHT = "model.layers.0.self_attn.q_proj.weight"

CALIBRATION_PROMPTS = [
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
]

EVALUATION_PROMPTS = [
    "Explain photosynthesis in simple terms.",
    "A binary search works efficiently because",
    "Water freezes at approximately",
    "A good scientific hypothesis should be",
    "In Python, a list is different from a tuple because",
    "The capital of France is",
    "An operating system is responsible for",
    "A triangle with three equal sides is",
    "A database index is useful because",
    "The first law of thermodynamics states that",
    "An electric circuit needs",
    "A variable is useful in a program because",
    "The Earth completes one orbit around the Sun in",
    "A function takes inputs and",
    "Version control lets developers",
    "A compiler translates source code into",
    "A prime number is divisible by",
    "A balanced ecosystem contains",
    "A probability between zero and one represents",
    "A stack follows the",
    "A map scale tells us",
    "An API allows software to",
    "The opposite of compression is",
    "A good test should be",
]


def parse_args():
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


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_sha256(t):
    # NumPy does not support torch.bfloat16 directly.
    # Hash the raw values in a stable float32 representation instead.
    x = t.detach().cpu().float().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def load_weight(model_dir):
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")

    tensors = load_file(str(path), device="cpu")
    if TARGET_WEIGHT not in tensors:
        raise KeyError(f"Tensor not found: {TARGET_WEIGHT}")
    return tensors[TARGET_WEIGHT].contiguous()


def load_model(model_dir, device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model


def inject_weight(model, weight):
    module_name = TARGET_WEIGHT.rsplit(".weight", 1)[0]
    module = model.get_submodule(module_name)

    with torch.no_grad():
        module.weight.copy_(
            weight.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        )


def tokenize(tokenizer, prompts):
    return [
        tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        for prompt in prompts
    ]


@torch.inference_mode()
def capture_q_inputs(model, token_lists, device):
    module_name = TARGET_WEIGHT.rsplit(".weight", 1)[0]
    module = model.get_submodule(module_name)

    captured = []

    def pre_hook(_module, inputs):
        x = inputs[0].detach().float().cpu()
        captured.append(
            x.reshape(-1, x.shape[-1])
        )

    handle = module.register_forward_pre_hook(pre_hook)

    try:
        for ids in token_lists:
            x = torch.tensor(
                [ids],
                dtype=torch.long,
                device=device,
            )
            model(input_ids=x)
    finally:
        handle.remove()

    return torch.cat(captured, dim=0).contiguous()


@torch.inference_mode()
def generate_greedy(
    model,
    token_lists,
    tokenizer,
    continuation_tokens,
    device,
):
    outputs = []

    for ids in token_lists:
        x = torch.tensor(
            [ids],
            dtype=torch.long,
            device=device,
        )

        y = model.generate(
            x,
            max_new_tokens=continuation_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        outputs.append(
            y[0, x.shape[1]:].detach().cpu()
        )

    return outputs


@torch.inference_mode()
def continuation_logits(
    model,
    token_lists,
    continuations,
    device,
):
    outputs = []

    for ids, continuation in zip(
        token_lists,
        continuations,
    ):
        full = torch.cat(
            [
                torch.tensor(
                    ids,
                    dtype=torch.long,
                ),
                continuation,
            ]
        )

        if continuation.numel() == 0:
            outputs.append(
                torch.empty(
                    (0, 0),
                    dtype=torch.float32,
                )
            )
            continue

        x = full[:-1].unsqueeze(0).to(device)

        logits = model(input_ids=x).logits[0]

        outputs.append(
            logits[-continuation.numel():]
            .float()
            .cpu()
        )

    return outputs


def behavior_metrics(
    baseline_continuations,
    candidate_continuations,
    baseline_logits,
    candidate_logits,
):
    total_tokens = 0
    matching_tokens = 0
    exact = 0

    squared_error = 0.0
    logit_count = 0

    for base, cand, base_logits, cand_logits in zip(
        baseline_continuations,
        candidate_continuations,
        baseline_logits,
        candidate_logits,
    ):
        n = min(
            base.numel(),
            cand.numel(),
        )

        total_tokens += n

        if n:
            matching_tokens += int(
                (
                    base[:n] == cand[:n]
                ).sum()
            )

        if (
            base.numel() == cand.numel()
            and torch.equal(base, cand)
        ):
            exact += 1

        m = min(
            base_logits.shape[0],
            cand_logits.shape[0],
        )

        if m:
            diff = (
                base_logits[:m]
                - cand_logits[:m]
            )

            squared_error += float(
                (diff * diff).sum()
            )

            logit_count += diff.numel()

    return {
        "greedy_token_match_rate": (
            matching_tokens / total_tokens
            if total_tokens
            else 0.0
        ),
        "greedy_exact_continuation_count": exact,
        "greedy_total_continuations": len(
            baseline_continuations
        ),
        "logit_rmse": (
            (squared_error / logit_count) ** 0.5
            if logit_count
            else float("nan")
        ),
    }


def relative_weight_error(original, reconstructed):
    return float(
        torch.linalg.vector_norm(
            original.float()
            - reconstructed.float()
        )
        / torch.linalg.vector_norm(
            original.float()
        )
    )


def learn_activation_aware_factors(
    weight,
    activations,
    rank,
    steps,
    lr,
    device,
):
    """
    Learn W ~= L @ R by minimizing error in the actual Q-projection
    outputs seen by the model:

        X W^T ~= X R^T L^T
    """

    u, s, vh = torch.linalg.svd(
        weight.float(),
        full_matrices=False,
    )

    L = nn.Parameter(
        (
            u[:, :rank]
            * s[:rank]
        ).to(device)
    )

    R = nn.Parameter(
        vh[:rank].to(device)
    )

    X = activations.to(device)
    W = weight.float().to(device)

    target = X @ W.T

    optimizer = torch.optim.AdamW(
        [L, R],
        lr=lr,
        weight_decay=0.0,
    )

    for step in range(1, steps + 1):
        batch = min(
            512,
            X.shape[0],
        )

        indices = torch.randint(
            X.shape[0],
            (batch,),
            device=device,
        )

        prediction = (
            X[indices] @ R.T
        ) @ L.T

        loss = F.mse_loss(
            prediction,
            target[indices],
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        if (
            step == 1
            or step == steps
            or step % 200 == 0
        ):
            print(
                f"  factor step {step:4d}/{steps}"
                f" | activation MSE "
                f"{loss.item():.8g}"
            )

    return (
        L.detach().cpu().contiguous(),
        R.detach().cpu().contiguous(),
    )


class FactorAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        hidden = min(
            512,
            max(
                64,
                input_dim // 2,
            ),
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                latent_dim,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                input_dim,
            ),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def flatten_factors(L, R):
    return torch.cat(
        [
            L.float().reshape(-1),
            R.float().reshape(-1),
        ]
    )


def split_factors(
    vector,
    L_shape,
    R_shape,
):
    n = 1

    for d in L_shape:
        n *= d

    L = vector[:n].reshape(
        L_shape
    )

    R = vector[n:].reshape(
        R_shape
    )

    return L, R


def train_latent(
    factors,
    latent_dim,
    steps,
    lr,
    seed,
):
    set_seed(seed)

    mean = factors.mean()
    std = factors.std().clamp_min(
        1e-8
    )

    x = (
        (factors - mean) / std
    ).unsqueeze(0)

    model = FactorAutoencoder(
        factors.numel(),
        latent_dim,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    for step in range(1, steps + 1):
        z = model.encode(x)
        reconstruction = model.decode(z)

        loss = F.mse_loss(
            reconstruction,
            x,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        if (
            step == 1
            or step == steps
            or step % 200 == 0
        ):
            print(
                f"  latent step {step:4d}/{steps}"
                f" | factor MSE "
                f"{loss.item():.8g}"
            )

    return (
        model,
        mean,
        std,
    )


@torch.inference_mode()
def encode_decode(
    model,
    mean,
    std,
    factors,
):
    x = (
        (factors - mean) / std
    ).unsqueeze(0)

    z = model.encode(x).squeeze(0)

    reconstructed = (
        model.decode(
            z.unsqueeze(0)
        ).squeeze(0)
        * std
        + mean
    )

    return (
        z.cpu(),
        reconstructed.cpu(),
    )


def latent_population_geometry(
    vectors
):
    Z = torch.stack(
        vectors
    ).float()

    mean = Z.mean(dim=0)
    std = Z.std(dim=0)

    if Z.shape[0] > 1:
        centered = Z - mean

        covariance = (
            centered.T @ centered
            / (Z.shape[0] - 1)
        )

        eigenvalues = (
            torch.linalg.eigvalsh(
                covariance
            )
            .clamp_min(0)
            .flip(0)
        )

        total = eigenvalues.sum()

        if total > 0:
            variance_ratio = (
                eigenvalues / total
            )
        else:
            variance_ratio = torch.zeros_like(
                eigenvalues
            )

        effective_rank = float(
            total ** 2
            / (
                eigenvalues.square().sum()
                + 1e-12
            )
        )
    else:
        eigenvalues = torch.zeros(
            Z.shape[1]
        )

        variance_ratio = torch.zeros_like(
            eigenvalues
        )

        effective_rank = 0.0

    return {
        "samples": int(Z.shape[0]),
        "dimension": int(Z.shape[1]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": Z.min(
            dim=0
        ).values.tolist(),
        "max": Z.max(
            dim=0
        ).values.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "variance_ratio": variance_ratio.tolist(),
        "effective_rank": effective_rank,
    }


def main():
    a = parse_args()

    device = torch.device(
        a.device
    )

    ranks = [
        int(x)
        for x in a.ranks.split(",")
        if x
    ]

    latent_sizes = [
        int(x)
        for x in a.latents.split(",")
        if x
    ]

    seeds = [
        int(x)
        for x in a.seeds.split(",")
        if x
    ]

    model_dir = (
        a.model_dir.resolve()
    )

    weight_path = (
        model_dir / "model.safetensors"
    )

    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {weight_path}"
        )

    a.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "EXPERIMENT 10 — FUNCTIONAL FACTOR -> LATENT STATE"
    )
    print("=" * 88)

    print(
        f"Model directory : {model_dir}"
    )
    print(
        f"Target weight   : {TARGET_WEIGHT}"
    )
    print(
        f"Device          : {device}"
    )
    print(
        f"Ranks           : {ranks}"
    )
    print(
        f"Latents         : {latent_sizes}"
    )
    print(
        f"Seeds           : {seeds}"
    )
    print()

    weight = load_weight(
        model_dir
    )

    print(
        f"Weight shape    : {tuple(weight.shape)}"
    )
    print(
        f"Weight dtype    : {weight.dtype}"
    )
    print(
        f"Weight values   : {weight.numel():,}"
    )
    print(
        f"Weight SHA-256  : {tensor_sha256(weight)}"
    )
    print()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    calibration_ids = tokenize(
        tokenizer,
        CALIBRATION_PROMPTS,
    )

    evaluation_ids = tokenize(
        tokenizer,
        EVALUATION_PROMPTS,
    )

    print(
        "[1] Reference model / behavior"
    )

    reference = load_model(
        model_dir,
        device,
    )

    calibration_activations = (
        capture_q_inputs(
            reference,
            calibration_ids,
            device,
        )
    )

    baseline_continuations = (
        generate_greedy(
            reference,
            evaluation_ids,
            tokenizer,
            a.continuation_tokens,
            device,
        )
    )

    baseline_logits = (
        continuation_logits(
            reference,
            evaluation_ids,
            baseline_continuations,
            device,
        )
    )

    del reference

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        "Calibration activations:"
        f" {tuple(calibration_activations.shape)}"
    )

    results = []

    for rank in ranks:
        print()
        print("-" * 88)
        print(
            f"[2] ACTIVATION-AWARE FACTORIZATION"
            f" — RANK {rank}"
        )
        print("-" * 88)

        L, R = (
            learn_activation_aware_factors(
                weight,
                calibration_activations,
                rank,
                a.steps,
                a.lr,
                device,
            )
        )

        factors = flatten_factors(
            L,
            R,
        )

        print(
            f"Factor values: {factors.numel():,}"
        )

        population = {
            dim: []
            for dim in latent_sizes
        }

        for latent_dim in latent_sizes:
            for seed in seeds:
                print()
                print("=" * 72)
                print(
                    f"RANK {rank} | LATENT {latent_dim}"
                    f" | SEED {seed}"
                )
                print("=" * 72)

                latent_model, mean, std = (
                    train_latent(
                        factors,
                        latent_dim,
                        a.steps,
                        a.lr,
                        seed,
                    )
                )

                z, reconstructed_factors = (
                    encode_decode(
                        latent_model,
                        mean,
                        std,
                        factors,
                    )
                )

                population[
                    latent_dim
                ].append(z)

                L2, R2 = split_factors(
                    reconstructed_factors,
                    tuple(L.shape),
                    tuple(R.shape),
                )

                reconstructed_weight = (
                    L2 @ R2
                ).to(
                    weight.dtype
                ).contiguous()

                weight_error = (
                    relative_weight_error(
                        weight,
                        reconstructed_weight,
                    )
                )

                print(
                    "[3] Functional test"
                )

                candidate = load_model(
                    model_dir,
                    device,
                )

                inject_weight(
                    candidate,
                    reconstructed_weight,
                )

                candidate_continuations = (
                    generate_greedy(
                        candidate,
                        evaluation_ids,
                        tokenizer,
                        a.continuation_tokens,
                        device,
                    )
                )

                candidate_logits = (
                    continuation_logits(
                        candidate,
                        evaluation_ids,
                        baseline_continuations,
                        device,
                    )
                )

                behavior = behavior_metrics(
                    baseline_continuations,
                    candidate_continuations,
                    baseline_logits,
                    candidate_logits,
                )

                del candidate

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                decoder_parameters = sum(
                    p.numel()
                    for p in latent_model.parameters()
                )

                latent_bytes = (
                    latent_dim * 4
                )

                decoder_bytes = (
                    decoder_parameters * 4
                )

                package_bytes = (
                    latent_bytes
                    + decoder_bytes
                )

                print()
                print(
                    "LATENT"
                )
                print(
                    "  values:",
                    [
                        round(
                            float(v),
                            7,
                        )
                        for v in z
                    ],
                )
                print(
                    f"  mean = {z.mean():.8f}"
                )
                print(
                    f"  std  = {z.std():.8f}"
                )
                print(
                    f"  min  = {z.min():.8f}"
                )
                print(
                    f"  max  = {z.max():.8f}"
                )
                print(
                    f"  SHA256 = {tensor_sha256(z)}"
                )

                print()
                print(
                    "WEIGHT"
                )
                print(
                    f"  relative error = "
                    f"{weight_error:.8f}"
                )

                print()
                print(
                    "BEHAVIOR"
                )
                print(
                    f"  greedy match = "
                    f"{behavior['greedy_token_match_rate']:.4f}"
                )
                print(
                    f"  exact = "
                    f"{behavior['greedy_exact_continuation_count']}/"
                    f"{behavior['greedy_total_continuations']}"
                )
                print(
                    f"  logit RMSE = "
                    f"{behavior['logit_rmse']:.8f}"
                )

                print()
                print(
                    "STORAGE"
                )
                print(
                    f"  original weight = "
                    f"{weight.numel() * 2:,} bytes"
                )
                print(
                    f"  latent only     = "
                    f"{latent_bytes:,} bytes"
                )
                print(
                    f"  decoder         = "
                    f"{decoder_bytes:,} bytes"
                )
                print(
                    f"  package         = "
                    f"{package_bytes:,} bytes"
                )

                results.append(
                    {
                        "rank": rank,
                        "latent_dim": latent_dim,
                        "seed": seed,
                        "latent": {
                            "values": z.tolist(),
                            "mean": float(
                                z.mean()
                            ),
                            "std": float(
                                z.std()
                            ),
                            "min": float(
                                z.min()
                            ),
                            "max": float(
                                z.max()
                            ),
                            "sha256":
                                tensor_sha256(z),
                        },
                        "weight": {
                            "relative_error":
                                weight_error,
                        },
                        "behavior": behavior,
                        "storage": {
                            "original_weight_bytes_bfloat16":
                                weight.numel() * 2,
                            "latent_bytes_float32":
                                latent_bytes,
                            "decoder_bytes_float32":
                                decoder_bytes,
                            "package_bytes_float32":
                                package_bytes,
                        },
                    }
                )

        print()
        print("-" * 88)
        print(
            f"[4] LATENT GEOMETRY — RANK {rank}"
        )
        print("-" * 88)

        for latent_dim, vectors in (
            population.items()
        ):
            geometry = (
                latent_population_geometry(
                    vectors
                )
            )

            cumulative = torch.cumsum(
                torch.tensor(
                    geometry[
                        "variance_ratio"
                    ]
                ),
                dim=0,
            )

            print(
                f"latent {latent_dim:4d}"
                f" | samples={geometry['samples']}"
                f" | effective-rank="
                f"{geometry['effective_rank']:.4f}"
            )

            print(
                "  variance ratio:",
                [
                    round(
                        float(v),
                        6,
                    )
                    for v in geometry[
                        "variance_ratio"
                    ]
                ],
            )

            print(
                "  cumulative:",
                [
                    round(
                        float(v),
                        6,
                    )
                    for v in cumulative
                ],
            )

    report = {
        "experiment":
            "Experiment 10 — functional factor to latent state",
        "target_weight": TARGET_WEIGHT,
        "model_directory": str(model_dir),
        "weight": {
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "values": weight.numel(),
            "sha256": tensor_sha256(weight),
        },
        "calibration_prompts":
            CALIBRATION_PROMPTS,
        "evaluation_prompts":
            EVALUATION_PROMPTS,
        "configuration": {
            "ranks": ranks,
            "latent_sizes": latent_sizes,
            "seeds": seeds,
            "steps": a.steps,
            "learning_rate": a.lr,
        },
        "results": results,
    }

    output_path = (
        a.output_dir / "summary.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(
        "rank latent seed | weight-error | "
        "greedy | exact | logit-RMSE"
    )

    for row in results:
        behavior = row["behavior"]

        print(
            f"{row['rank']:4d}"
            f" {row['latent_dim']:6d}"
            f" {row['seed']:4d} |"
            f" {row['weight']['relative_error']:.6f}"
            f" | {behavior['greedy_token_match_rate']:.4f}"
            f" | "
            f"{behavior['greedy_exact_continuation_count']:2d}/"
            f"{behavior['greedy_total_continuations']:2d}"
            f" | {behavior['logit_rmse']:.6f}"
        )

    print()
    print(
        f"Results saved to: {output_path}"
    )
    print(
        "Original model.safetensors was never modified."
    )


if __name__ == "__main__":
    main()
