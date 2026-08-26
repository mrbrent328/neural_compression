#!/usr/bin/env python3
"""
E18_FUNCTIONAL_LATENT.py
=========================

Question:
    Can a frozen shared constructor recover a held-out Q projection
    more effectively when the held-out latent is optimized for FUNCTION
    (Q-projection activations) instead of raw factor/weight MSE?

Design:
    - Same 4 training Q projections + 1 held-out target as E14.
    - Same rank-48 factor representation.
    - Same shared constructor.
    - Same seeds 9,10,11.
    - Constructor is trained once per seed using factor MSE.
    - The held-out latent is then optimized TWO ways:
        A) WEIGHT objective: factor MSE (historical baseline)
        B) RELATIONAL ATTENTION objective: MSE between Q outputs X @ W.T
           on held-out calibration prompts.
    - Both reconstructions are evaluated on the original 24-prompt
      behavioral benchmark.

This is intentionally a paired test: the constructor is identical.
Only the held-out latent objective changes.

Outputs:
    results/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file


DEFAULT_TARGETS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.2.self_attn.q_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.layers.4.self_attn.q_proj.weight",
]

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
    p.add_argument("--rank", type=int, default=48)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--latent-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--latent-lr", type=float, default=2e-3)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--continuation-tokens", type=int, default=6)
    p.add_argument("--seeds", default="9,10,11")
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("E19_relational_attention_results"),
    )
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_weight(model_dir: Path, target_name: str):
    tensors = load_file(
        str(model_dir / "model.safetensors"),
        device="cpu",
    )
    if target_name not in tensors:
        raise KeyError(f"Tensor not found: {target_name}")
    return tensors[target_name].contiguous()


def load_model(model_dir: Path, device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model


def tokenize(tokenizer, prompts):
    return [
        tokenizer(x, add_special_tokens=False)["input_ids"]
        for x in prompts
    ]


@torch.no_grad()
def capture_attention_context(model, token_lists, device, attention_module_name):
    q_module = model.get_submodule(attention_module_name + ".q_proj")
    k_module = model.get_submodule(attention_module_name + ".k_proj")
    v_module = model.get_submodule(attention_module_name + ".v_proj")

    captured = {"hidden": [], "q": [], "k": [], "v": []}

    def pre_hook(_module, inputs):
        captured["hidden"].append(inputs[0].detach().float().cpu().clone())
    def q_hook(_module, _inputs, output):
        captured["q"].append(output.detach().float().cpu().clone())
    def k_hook(_module, _inputs, output):
        captured["k"].append(output.detach().float().cpu().clone())
    def v_hook(_module, _inputs, output):
        captured["v"].append(output.detach().float().cpu().clone())

    handles = [
        q_module.register_forward_pre_hook(pre_hook),
        q_module.register_forward_hook(q_hook),
        k_module.register_forward_hook(k_hook),
        v_module.register_forward_hook(v_hook),
    ]
    try:
        for ids in token_lists:
            model(input_ids=torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        for h in handles:
            h.remove()

    if not captured["hidden"]:
        raise RuntimeError("No attention context was captured.")
    return captured


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
                torch.tensor(ids, dtype=torch.long),
                continuation,
            ]
        )

        if continuation.numel() == 0:
            outputs.append(
                torch.empty((0, 0), dtype=torch.float32)
            )
            continue

        logits = model(
            input_ids=full[:-1].unsqueeze(0).to(device)
        ).logits[0]

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
        n = min(base.numel(), cand.numel())

        total_tokens += n

        if n:
            matching_tokens += int(
                (base[:n] == cand[:n]).sum()
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
        "exact": exact,
        "total_prompts": len(baseline_continuations),
        "logit_rmse": (
            squared_error / logit_count
        ) ** 0.5 if logit_count else float("nan"),
    }


def learn_svd_factors(weight, rank):
    u, s, vh = torch.linalg.svd(
        weight.float(),
        full_matrices=False,
    )
    r = min(rank, s.numel())
    L = u[:, :r] * s[:r]
    R = vh[:r]
    return L.contiguous(), R.contiguous()


def flatten_factors(L, R):
    return torch.cat(
        [
            L.float().reshape(-1),
            R.float().reshape(-1),
        ]
    )


def split_factors(vector, L_shape, R_shape):
    vector = vector.reshape(-1)

    L_n = 1
    for d in L_shape:
        L_n *= d

    R_n = 1
    for d in R_shape:
        R_n *= d

    expected = L_n + R_n

    if vector.numel() != expected:
        raise ValueError(
            f"Factor vector has {vector.numel():,} values, "
            f"but L ({L_shape}) + R ({R_shape}) "
            f"requires {expected:,}."
        )

    L = vector[:L_n].reshape(L_shape)
    R = vector[L_n:L_n + R_n].reshape(R_shape)

    return L, R


class SharedConstructor(nn.Module):
    def __init__(
        self,
        input_dim,
        target_count,
        latent_dim,
        hidden=512,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.constructor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
        )

        self.latents = nn.Parameter(
            torch.randn(target_count, latent_dim)
        )

    def decode(self, z):
        return self.constructor(z)


def train_shared_constructor(
    factor_vectors,
    latent_dim,
    steps,
    lr,
    seed,
    device,
):
    set_seed(seed)

    X = torch.stack(
        factor_vectors
    ).float().to(device)

    means = X.mean(
        dim=1,
        keepdim=True,
    )

    stds = X.std(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)

    normalized = (
        X - means
    ) / stds

    model = SharedConstructor(
        input_dim=X.shape[1],
        target_count=X.shape[0],
        latent_dim=latent_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    for step in range(1, steps + 1):
        reconstruction = model.decode(
            model.latents
        )

        loss = F.mse_loss(
            reconstruction,
            normalized,
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()
        optimizer.step()

        if (
            step == 1
            or step == steps
            or step % 100 == 0
        ):
            print(
                f"  constructor {step:4d}/{steps} "
                f"| factor MSE {loss.item():.8g}"
            )

    return model, means.cpu(), stds.cpu()


def decode_weight_from_latent(
    constructor,
    z,
    mean,
    std,
    L_shape,
    R_shape,
):
    factor_vector = (
        constructor.decode(z).reshape(-1)
        * std
        + mean
    )

    L, R = split_factors(
        factor_vector,
        L_shape,
        R_shape,
    )

    return L @ R


def optimize_heldout_latent_weight(
    constructor,
    target_factor,
    mean,
    std,
    latent_dim,
    steps,
    lr,
    seed,
    device,
):
    set_seed(seed)

    target = (
        (target_factor - mean) / std
    ).to(device)

    z = nn.Parameter(
        torch.randn(
            1,
            latent_dim,
            device=device,
        )
    )

    for p in constructor.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [z],
        lr=lr,
        weight_decay=0.0,
    )

    for _ in range(steps):
        prediction = constructor.decode(z)

        loss = F.mse_loss(
            prediction,
            target.reshape(1, -1),
        )

        optimizer.zero_grad(
            set_to_none=True
        )
        loss.backward()
        optimizer.step()

    return z.detach().cpu()


def _attention_relational_losses(candidate_weight, context, num_heads):
    score_losses = []
    output_losses = []

    for hidden, original_q, k, v in zip(
        context["hidden"], context["q"], context["k"], context["v"]
    ):
        hidden = hidden.to(candidate_weight.device)
        original_q = original_q.to(candidate_weight.device)
        k = k.to(candidate_weight.device)
        v = v.to(candidate_weight.device)

        candidate_q = hidden @ candidate_weight.T
        _, seq, q_dim = candidate_q.shape

        # Some transformer architectures use grouped/multi-query attention:
        # Q can have more heads than K/V. E19 must preserve that relationship
        # instead of forcing K/V into the Q-head count.
        if q_dim % num_heads:
            raise ValueError(
                f"Q dimension {q_dim} is not divisible by "
                f"num_heads={num_heads}"
            )
        head_dim = q_dim // num_heads

        def q_heads(x):
            if x.shape[-1] != q_dim:
                raise ValueError(
                    f"Expected Q width {q_dim}, got {x.shape[-1]}"
                )
            return x.reshape(1, seq, num_heads, head_dim).transpose(1, 2)

        kv_dim = k.shape[-1]
        if kv_dim % head_dim:
            raise ValueError(
                f"K/V width {kv_dim} is not divisible by Q head_dim "
                f"{head_dim}"
            )
        kv_heads = kv_dim // head_dim

        def kv_heads_fn(x):
            if x.shape[-1] != kv_dim:
                raise ValueError(
                    f"Expected K/V width {kv_dim}, got {x.shape[-1]}"
                )
            return x.reshape(1, seq, kv_heads, head_dim).transpose(1, 2)

        cq = q_heads(candidate_q)
        oq = q_heads(original_q)
        kh = kv_heads_fn(k)
        vh = kv_heads_fn(v)

        # Grouped-query attention: repeat each K/V head across its Q-head group.
        if kv_heads != num_heads:
            if num_heads % kv_heads != 0:
                raise ValueError(
                    f"Q heads ({num_heads}) must be divisible by "
                    f"K/V heads ({kv_heads}) for grouped-query attention"
                )
            repeat = num_heads // kv_heads
            kh = kh.repeat_interleave(repeat, dim=1)
            vh = vh.repeat_interleave(repeat, dim=1)

        scale = head_dim ** -0.5

        c_scores = torch.matmul(cq, kh.transpose(-1, -2)) * scale
        o_scores = torch.matmul(oq, kh.transpose(-1, -2)) * scale

        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=candidate_weight.device),
            diagonal=1,
        )
        c_scores = c_scores.masked_fill(mask, float("-inf"))
        o_scores = o_scores.masked_fill(mask, float("-inf"))

        c_probs = torch.softmax(c_scores, dim=-1)
        o_probs = torch.softmax(o_scores, dim=-1)

        c_ctx = torch.matmul(c_probs, vh)
        o_ctx = torch.matmul(o_probs, vh)

        score_losses.append(F.mse_loss(c_probs, o_probs))
        output_losses.append(F.mse_loss(c_ctx, o_ctx))

    return torch.stack(score_losses).mean(), torch.stack(output_losses).mean()


def optimize_heldout_latent_relational(
    constructor, target_factor, mean, std, L_shape, R_shape,
    context, num_heads, latent_dim, steps, lr, seed, device
):
    set_seed(seed)
    z = nn.Parameter(torch.randn(1, latent_dim, device=device))
    for p in constructor.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW([z], lr=lr, weight_decay=0.0)

    for step in range(1, steps + 1):
        factor_vector = (
            constructor.decode(z).reshape(-1)
            * std.to(device) + mean.to(device)
        )
        L, R = split_factors(factor_vector, L_shape, R_shape)
        candidate_weight = L @ R

        score_loss, output_loss = _attention_relational_losses(
            candidate_weight, context, num_heads
        )
        loss = score_loss + output_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == steps or step % 100 == 0:
            print(
                f"  relational step {step:4d}/{steps} | "
                f"attention MSE {score_loss.item():.8g} | "
                f"context MSE {output_loss.item():.8g}"
            )

    return z.detach().cpu()

def relative_weight_error(
    original,
    reconstructed,
):
    return float(
        torch.linalg.vector_norm(
            original.float()
            - reconstructed.float()
        )
        / torch.linalg.vector_norm(
            original.float()
        )
    )


def evaluate_weight(
    model_dir,
    device,
    target_name,
    reconstructed_weight,
    baseline_continuations,
    baseline_logits,
    evaluation_ids,
    tokenizer,
    continuation_tokens,
):
    candidate = load_model(
        model_dir,
        device,
    )

    module = candidate.get_submodule(
        target_name.rsplit(".weight", 1)[0]
    )

    with torch.no_grad():
        module.weight.copy_(
            reconstructed_weight.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        )

    candidate_continuations = generate_greedy(
        candidate,
        evaluation_ids,
        tokenizer,
        continuation_tokens,
        device,
    )

    candidate_logits = continuation_logits(
        candidate,
        evaluation_ids,
        baseline_continuations,
        device,
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

    return behavior


def main():
    a = parse_args()

    device = torch.device(a.device)

    targets = [
        x.strip()
        for x in a.targets.split(",")
        if x.strip()
    ]

    seeds = [
        int(x.strip())
        for x in a.seeds.split(",")
        if x.strip()
    ]

    if len(targets) < 3:
        raise ValueError(
            "Need at least 3 targets; the last is held out."
        )

    train_targets = targets[:-1]
    heldout_target = targets[-1]

    a.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "E18 — FUNCTIONAL HELD-OUT LATENT "
        "vs WEIGHT-MSE LATENT"
    )
    print("=" * 88)
    print(f"Device       : {device}")
    print(f"Rank         : {a.rank}")
    print(f"Latent dims  : 2 (fixed for E18)")
    print(f"Train targets: {len(train_targets)}")
    print(f"Held-out     : {heldout_target}")
    print(f"Seeds        : {seeds}")

    tokenizer = AutoTokenizer.from_pretrained(
        a.model_dir,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_ids = tokenize(
        tokenizer,
        CALIBRATION_PROMPTS,
    )

    evaluation_ids = tokenize(
        tokenizer,
        EVALUATION_PROMPTS,
    )

    print("\n[1] Reference model / behavior")

    reference = load_model(
        a.model_dir,
        device,
    )

    baseline_continuations = generate_greedy(
        reference,
        evaluation_ids,
        tokenizer,
        a.continuation_tokens,
        device,
    )

    baseline_logits = continuation_logits(
        reference,
        evaluation_ids,
        baseline_continuations,
        device,
    )
    num_heads = a.num_heads
    if num_heads is None:
        num_heads = int(reference.config.num_attention_heads)
    print(f"  attention heads = {num_heads}")


    print("\n[2] Capture Q/K/V attention context")

    activations = {}
    for target in targets:
        attention_name = target.rsplit(".q_proj.weight", 1)[0]
        print(f"  {attention_name}")
        activations[target] = capture_attention_context(
            reference, calibration_ids, device, attention_name
        )
        print(
            f"    captured prompts = "
            f"{len(activations[target]['hidden'])}"
        )

    del reference

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[3] Build rank-limited factors")

    factor_data = {}

    for target in targets:
        weight = load_weight(
            a.model_dir,
            target,
        )

        L, R = learn_svd_factors(
            weight,
            a.rank,
        )

        factor_data[target] = {
            "weight": weight,
            "L": L,
            "R": R,
            "factors": flatten_factors(L, R),
        }

        print(
            f"  {target}"
            f" | weight={tuple(weight.shape)}"
            f" | factor_values="
            f"{factor_data[target]['factors'].numel():,}"
        )

    results = []

    # We use one fixed latent dimension per invocation. This keeps E18
    # focused on the objective rather than combining another latent sweep
    # with it.
    latent_dim = 2

    for seed in seeds:
        print("\n" + "#" * 88)
        print(
            f"SEED {seed} | LATENT DIM {latent_dim}"
        )
        print("#" * 88)

        train_factors = [
            factor_data[t]["factors"]
            for t in train_targets
        ]

        constructor, means, stds = (
            train_shared_constructor(
                train_factors,
                latent_dim,
                a.train_steps,
                a.lr,
                seed,
                device,
            )
        )

        heldout_data = factor_data[
            heldout_target
        ]

        heldout_factor = heldout_data[
            "factors"
        ]

        mean = heldout_factor.mean()
        std = heldout_factor.std().clamp_min(
            1e-8
        )

        L_shape = tuple(
            heldout_data["L"].shape
        )

        R_shape = tuple(
            heldout_data["R"].shape
        )

        print("\n[4A] Held-out latent: WEIGHT-MSE objective")

        z_weight = optimize_heldout_latent_weight(
            constructor,
            heldout_factor,
            mean,
            std,
            latent_dim,
            a.latent_steps,
            a.latent_lr,
            seed,
            device,
        )

        W_weight = decode_weight_from_latent(
            constructor,
            z_weight.to(device),
            mean,
            std,
            L_shape,
            R_shape,
        ).cpu().contiguous()

        weight_error_weight = relative_weight_error(
            heldout_data["weight"],
            W_weight,
        )

        behavior_weight = evaluate_weight(
            a.model_dir,
            device,
            heldout_target,
            W_weight,
            baseline_continuations,
            baseline_logits,
            evaluation_ids,
            tokenizer,
            a.continuation_tokens,
        )

        print(
            f"  latent = "
            f"{z_weight.reshape(-1).tolist()}"
        )
        print(
            f"  weight error = "
            f"{weight_error_weight:.8f}"
        )
        print(
            f"  greedy       = "
            f"{behavior_weight['greedy_token_match_rate']:.4f}"
        )
        print(
            f"  exact        = "
            f"{behavior_weight['exact']}/"
            f"{behavior_weight['total_prompts']}"
        )
        print(
            f"  logit RMSE   = "
            f"{behavior_weight['logit_rmse']:.8f}"
        )

        print("\n[4B] Held-out latent: RELATIONAL ATTENTION objective")

        z_function = optimize_heldout_latent_relational(
            constructor,
            heldout_factor,
            mean,
            std,
            L_shape,
            R_shape,
            activations[heldout_target],
            num_heads,
            latent_dim,
            a.latent_steps,
            a.latent_lr,
            seed,
            device,
        )


        W_function = decode_weight_from_latent(
            constructor,
            z_function.to(device),
            mean,
            std,
            L_shape,
            R_shape,
        ).cpu().contiguous()

        weight_error_function = relative_weight_error(
            heldout_data["weight"],
            W_function,
        )

        behavior_function = evaluate_weight(
            a.model_dir,
            device,
            heldout_target,
            W_function,
            baseline_continuations,
            baseline_logits,
            evaluation_ids,
            tokenizer,
            a.continuation_tokens,
        )

        print(
            f"  latent = "
            f"{z_function.reshape(-1).tolist()}"
        )
        print(
            f"  weight error = "
            f"{weight_error_function:.8f}"
        )
        print(
            f"  greedy       = "
            f"{behavior_function['greedy_token_match_rate']:.4f}"
        )
        print(
            f"  exact        = "
            f"{behavior_function['exact']}/"
            f"{behavior_function['total_prompts']}"
        )
        print(
            f"  logit RMSE   = "
            f"{behavior_function['logit_rmse']:.8f}"
        )

        results.append(
            {
                "seed": seed,
                "latent_dim": latent_dim,
                "weight_objective": {
                    "latent": z_weight.reshape(-1).tolist(),
                    "weight_error": weight_error_weight,
                    "behavior": behavior_weight,
                },
                "function_objective": {
                    "latent": z_function.reshape(-1).tolist(),
                    "weight_error": weight_error_function,
                    "behavior": behavior_function,
                },
            }
        )

        del constructor

        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "experiment": (
            "E18 — functional held-out latent "
            "vs weight-MSE latent"
        ),
        "question": (
            "Does optimizing a frozen held-out latent "
            "for Q-projection function improve behavioral "
            "recovery compared with optimizing it for factor MSE?"
        ),
        "rank": a.rank,
        "latent_dim": latent_dim,
        "train_targets": train_targets,
        "heldout_target": heldout_target,
        "calibration_prompt_count": len(
            CALIBRATION_PROMPTS
        ),
        "evaluation_prompt_count": len(
            EVALUATION_PROMPTS
        ),
        "train_steps": a.train_steps,
        "latent_steps": a.latent_steps,
        "learning_rate": a.lr,
        "latent_learning_rate": a.latent_lr,
        "seeds": seeds,
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

    print("\n" + "=" * 88)
    print("E19 FINAL SUMMARY")
    print("=" * 88)

    for row in results:
        s = row["seed"]

        w = row["weight_objective"]
        f = row["function_objective"]

        wb = w["behavior"]
        fb = f["behavior"]

        print(
            f"seed {s:3d} | WEIGHT "
            f"| weight {w['weight_error']:.6f} "
            f"| greedy {wb['greedy_token_match_rate']:.4f} "
            f"| exact {wb['exact']:2d}/{wb['total_prompts']:2d} "
            f"| RMSE {wb['logit_rmse']:.6f}"
        )

        print(
            f"seed {s:3d} | FUNCTION "
            f"| weight {f['weight_error']:.6f} "
            f"| greedy {fb['greedy_token_match_rate']:.4f} "
            f"| exact {fb['exact']:2d}/{fb['total_prompts']:2d} "
            f"| RMSE {fb['logit_rmse']:.6f}"
        )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
