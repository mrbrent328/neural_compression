#!/usr/bin/env python3
"""
EXPERIMENT 16 — LOCAL LATENTS + TRUE PAIRWISE RELATIONAL INTERACTION

Hypothesis
----------
A target-specific local representation may become useful for held-out
generalization when the model can explicitly compare representations
pairwise, rather than merely receiving a pooled group summary.

Architecture
------------
Each target has a local latent:

    A ─┐
    B ─┤
    C ─┼──> pairwise interaction / self-attention ──> constructor ──> factors
    D ─┘

The relational block is intentionally small so the experiment does not
simply become a large memorizer.

Training
--------
- 4 Q-projection weights are observed.
- Each gets its own learned local latent.
- A small self-attention block lets every local latent interact with
  every other local latent.
- The shared relational block + constructor are trained jointly.
- The 5th Q-projection weight is held out completely.

Held-out test
-------------
- Freeze the trained relational block and constructor.
- Introduce ONLY a new local latent for the held-out target.
- The new latent is optimized against the held-out factor target.
- The held-out relational state is computed by comparing it with the
  frozen training-target representations.
- Evaluate weight error and model behavior.

Controls
--------
The script can optionally run an independent-local control with no
relational interaction:

    --run-control

This is important because it separates:
    local representation capacity
from:
    actual benefit of relational interaction.

Fixed comparison settings
-------------------------
- rank = 48
- local latent dimension = 4
- relational dimension = 32
- hidden constructor dimension = 512
- 500 shared training steps
- 500 held-out latent steps
- seeds 9, 10, 11
- same held-out layer convention as Experiments 13–15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    p.add_argument("--local-latent-dim", type=int, default=4)
    p.add_argument("--relational-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--latent-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--latent-lr", type=float, default=2e-3)
    p.add_argument("--continuation-tokens", type=int, default=6)
    p.add_argument("--seeds", default="9,10,11")
    p.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    p.add_argument(
        "--run-control",
        action="store_true",
        help="Also run independent-local control with no relational interaction.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment16_pairwise_relational_results"),
    )
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_weight(model_dir, target_name):
    tensors = load_file(
        str(Path(model_dir) / "model.safetensors"),
        device="cpu",
    )
    if target_name not in tensors:
        raise KeyError(f"Tensor not found: {target_name}")
    return tensors[target_name].contiguous()


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


def tokenize(tokenizer, prompts):
    return [
        tokenizer(x, add_special_tokens=False)["input_ids"]
        for x in prompts
    ]


@torch.inference_mode()
def generate_greedy(model, token_lists, tokenizer, continuation_tokens, device):
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

        logits = model(
            input_ids=full[:-1].unsqueeze(0).to(device)
        ).logits[0]

        outputs.append(
            logits[-continuation.numel():]
            .float()
            .cpu()
        )

    return outputs



@torch.inference_mode()
def capture_q_inputs(
    model,
    token_lists,
    device,
    module_name,
):
    """
    Capture the input activations arriving at a Q-projection module.

    This is the same activation-capture role used by the earlier
    experiments: the Q projection receives a hidden-state tensor, and
    those inputs are collected as calibration activations.
    """
    module = model.get_submodule(module_name)
    captured = []

    def pre_hook(_module, inputs):
        x = inputs[0].detach().float().cpu()
        captured.append(
            x.reshape(-1, x.shape[-1])
        )

    handle = module.register_forward_pre_hook(
        pre_hook
    )

    try:
        for ids in token_lists:
            model(
                input_ids=torch.tensor(
                    [ids],
                    dtype=torch.long,
                    device=device,
                )
            )
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            f"No activations were captured for {module_name}."
        )

    return torch.cat(
        captured,
        dim=0,
    ).contiguous()


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
            squared_error / logit_count
        ) ** 0.5
        if logit_count
        else float("nan"),
    }


def learn_activation_aware_factors(
    weight,
    activations,
    rank,
    steps,
    lr,
    device,
):
    u, s, vh = torch.linalg.svd(
        weight.float(),
        full_matrices=False,
    )

    L = nn.Parameter(
        (
            u[:, :rank] * s[:rank]
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

    for step in range(
        1,
        steps + 1,
    ):
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
                f"  factor step {step:4d}/{steps} | "
                f"activation MSE {loss.item():.8g}"
            )

    return (
        L.detach().cpu().contiguous(),
        R.detach().cpu().contiguous(),
    )


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


class PairwiseRelationalConstructor(nn.Module):
    """
    True pairwise relational model.

    Each target has a local latent. A small self-attention block allows
    each target to compare its latent with every other target latent.

    The output corresponding to each target is then passed through the
    shared constructor.
    """

    def __init__(
        self,
        input_dim,
        target_count,
        local_dim,
        relational_dim,
        hidden=512,
        heads=4,
    ):
        super().__init__()

        if relational_dim % heads != 0:
            raise ValueError(
                "relational_dim must be divisible by heads."
            )

        self.target_count = target_count
        self.local_dim = local_dim
        self.relational_dim = relational_dim

        self.local_latents = nn.Parameter(
            torch.randn(
                target_count,
                local_dim,
            )
        )

        self.local_projection = nn.Linear(
            local_dim,
            relational_dim,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=relational_dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            relational_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                relational_dim,
                relational_dim * 2,
            ),
            nn.GELU(),
            nn.Linear(
                relational_dim * 2,
                relational_dim,
            ),
        )

        self.norm2 = nn.LayerNorm(
            relational_dim
        )

        self.constructor = nn.Sequential(
            nn.Linear(
                relational_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                input_dim,
            ),
        )

    def relational_states(
        self,
        latents,
    ):
        x = self.local_projection(
            latents
        ).unsqueeze(0)

        attended, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )

        x = self.norm1(
            x + attended
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x.squeeze(0)

    def decode_all(
        self,
        latents=None,
    ):
        if latents is None:
            latents = self.local_latents

        relational = self.relational_states(
            latents
        )

        return self.constructor(
            relational
        )

    def decode_heldout(
        self,
        heldout_latent,
        training_latents,
    ):
        """
        The held-out target is compared against the frozen training
        representations. Its relational output is taken from the final
        token only.
        """
        all_latents = torch.cat(
            [
                training_latents,
                heldout_latent,
            ],
            dim=0,
        )

        relational = self.relational_states(
            all_latents
        )

        return self.constructor(
            relational[-1:].contiguous()
        )


class IndependentLocalConstructor(nn.Module):
    """
    Control:
        local latent -> shared constructor

    There is no relational interaction.
    """

    def __init__(
        self,
        input_dim,
        target_count,
        local_dim,
        hidden=512,
    ):
        super().__init__()

        self.local_latents = nn.Parameter(
            torch.randn(
                target_count,
                local_dim,
            )
        )

        self.constructor = nn.Sequential(
            nn.Linear(
                local_dim,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.GELU(),
            nn.Linear(
                hidden,
                input_dim,
            ),
        )

    def decode_all(
        self,
        latents=None,
    ):
        if latents is None:
            latents = self.local_latents

        return self.constructor(
            latents
        )

    def decode_heldout(
        self,
        heldout_latent,
        training_latents,
    ):
        return self.constructor(
            heldout_latent
        )


def train_model(
    factor_vectors,
    local_dim,
    relational_dim,
    hidden,
    heads,
    steps,
    lr,
    seed,
    device,
    use_relation,
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

    if use_relation:
        model = PairwiseRelationalConstructor(
            input_dim=X.shape[1],
            target_count=X.shape[0],
            local_dim=local_dim,
            relational_dim=relational_dim,
            hidden=hidden,
            heads=heads,
        ).to(device)
    else:
        model = IndependentLocalConstructor(
            input_dim=X.shape[1],
            target_count=X.shape[0],
            local_dim=local_dim,
            hidden=hidden,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.0,
    )

    for step in range(
        1,
        steps + 1,
    ):
        prediction = model.decode_all()

        loss = F.mse_loss(
            prediction,
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
                f"  shared step {step:4d}/{steps} | "
                f"factor MSE {loss.item():.8g}"
            )

    return (
        model,
        means.cpu(),
        stds.cpu(),
    )


def optimize_heldout_latent(
    model,
    heldout_factor,
    mean,
    std,
    local_dim,
    steps,
    lr,
    seed,
    device,
    use_relation,
):
    set_seed(seed)

    x = (
        (heldout_factor - mean) / std
    ).reshape(1, -1).to(device)

    z = nn.Parameter(
        torch.randn(
            1,
            local_dim,
            device=device,
        )
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    training_latents = (
        model.local_latents.detach()
    )

    optimizer = torch.optim.AdamW(
        [z],
        lr=lr,
        weight_decay=0.0,
    )

    for step in range(
        1,
        steps + 1,
    ):
        prediction = model.decode_heldout(
            z,
            training_latents,
        )

        loss = F.mse_loss(
            prediction,
            x,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        reconstructed_norm = model.decode_heldout(
            z,
            training_latents,
        )

        reconstructed = (
            reconstructed_norm.cpu()
            * std
            + mean
        )

    return (
        z.detach().cpu().reshape(-1),
        reconstructed.reshape(-1).contiguous(),
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
        target_name.rsplit(
            ".weight",
            1,
        )[0]
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
        int(x)
        for x in a.seeds.split(",")
        if x.strip()
    ]

    if len(targets) < 3:
        raise ValueError(
            "Need at least 3 targets; the last is held out."
        )

    if a.local_latent_dim < 1:
        raise ValueError(
            "local-latent-dim must be >= 1."
        )

    train_targets = targets[:-1]
    heldout_target = targets[-1]

    a.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print(
        "EXPERIMENT 16 — TRUE PAIRWISE RELATIONAL INTERACTION"
    )
    print("=" * 88)
    print(f"Device          : {device}")
    print(f"Rank            : {a.rank}")
    print(
        f"Local latent dim: "
        f"{a.local_latent_dim}"
    )
    print(
        f"Relational dim  : "
        f"{a.relational_dim}"
    )
    print(f"Attention heads : {a.heads}")
    print(f"Train targets   : {len(train_targets)}")
    print(f"Held-out        : {heldout_target}")
    print(f"Seeds           : {seeds}")
    print(f"Run control     : {a.run_control}")

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

    activations = {}

    for target in targets:
        module_name = target.rsplit(
            ".weight",
            1,
        )[0]

        print(
            f"Capturing activations: "
            f"{module_name}"
        )

        activations[target] = capture_q_inputs(
            reference,
            calibration_ids,
            device,
            module_name,
        )

    del reference

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[2] Activation-aware factorization")

    factor_data = {}

    for target in targets:
        print(f"\nTarget: {target}")

        weight = load_weight(
            a.model_dir,
            target,
        )

        L, R = learn_activation_aware_factors(
            weight,
            activations[target],
            a.rank,
            a.train_steps,
            a.lr,
            device,
        )

        factor_data[target] = {
            "weight": weight,
            "L": L,
            "R": R,
            "factors": flatten_factors(L, R),
        }

        print(
            f"  weight shape : "
            f"{tuple(weight.shape)}"
        )

        print(
            f"  factor values: "
            f"{factor_data[target]['factors'].numel():,}"
        )

    architectures = [
        ("pairwise_relational", True)
    ]

    if a.run_control:
        architectures.append(
            ("independent_local_control", False)
        )

    results = []

    for architecture_name, use_relation in architectures:
        for seed in seeds:
            print("\n" + "=" * 88)
            print(
                f"{architecture_name.upper()} | "
                f"SEED {seed}"
            )
            print("=" * 88)

            train_factors = [
                factor_data[t]["factors"]
                for t in train_targets
            ]

            model, means, stds = train_model(
                train_factors,
                a.local_latent_dim,
                a.relational_dim,
                a.hidden,
                a.heads,
                a.train_steps,
                a.lr,
                seed,
                device,
                use_relation,
            )

            print("\n[3] Training targets")

            train_results = []

            with torch.no_grad():
                decoded = model.decode_all()

            for i, target in enumerate(
                train_targets
            ):
                rec = (
                    decoded[i:i + 1].cpu()
                    * stds[i]
                    + means[i]
                )

                original = factor_data[target]["weight"]

                L2, R2 = split_factors(
                    rec,
                    tuple(
                        factor_data[target]["L"].shape
                    ),
                    tuple(
                        factor_data[target]["R"].shape
                    ),
                )

                reconstructed_weight = (
                    L2 @ R2
                ).to(
                    original.dtype
                ).contiguous()

                error = relative_weight_error(
                    original,
                    reconstructed_weight,
                )

                latent_values = (
                    model.local_latents[i]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

                print(
                    f"  {target} | "
                    f"weight error={error:.8f} | "
                    f"local latent={latent_values}"
                )

                train_results.append(
                    {
                        "target": target,
                        "local_latent": latent_values,
                        "weight_error": error,
                    }
                )

            print("\n[4] HELD-OUT TARGET")

            print(
                f"  target = "
                f"{heldout_target}"
            )

            print(
                "  Shared interaction + constructor frozen; "
                "optimizing ONLY the new local latent."
            )

            heldout_factor = factor_data[
                heldout_target
            ]["factors"]

            mean = heldout_factor.mean()

            std = heldout_factor.std().clamp_min(
                1e-8
            )

            z, rec = optimize_heldout_latent(
                model,
                heldout_factor,
                mean,
                std,
                a.local_latent_dim,
                a.latent_steps,
                a.latent_lr,
                seed,
                device,
                use_relation,
            )

            original = factor_data[
                heldout_target
            ]["weight"]

            L2, R2 = split_factors(
                rec,
                tuple(
                    factor_data[
                        heldout_target
                    ]["L"].shape
                ),
                tuple(
                    factor_data[
                        heldout_target
                    ]["R"].shape
                ),
            )

            reconstructed_weight = (
                L2 @ R2
            ).to(
                original.dtype
            ).contiguous()

            weight_error = relative_weight_error(
                original,
                reconstructed_weight,
            )

            behavior = evaluate_weight(
                a.model_dir,
                device,
                heldout_target,
                reconstructed_weight,
                baseline_continuations,
                baseline_logits,
                evaluation_ids,
                tokenizer,
                a.continuation_tokens,
            )

            print("\nHELD-OUT RESULT")

            print(
                f"  local latent = "
                f"{z.tolist()}"
            )

            print(
                f"  weight error = "
                f"{weight_error:.8f}"
            )

            print(
                f"  greedy match = "
                f"{behavior['greedy_token_match_rate']:.4f}"
            )

            print(
                f"  exact        = "
                f"{behavior['greedy_exact_continuation_count']}/"
                f"{behavior['greedy_total_continuations']}"
            )

            print(
                f"  logit RMSE   = "
                f"{behavior['logit_rmse']:.8f}"
            )

            results.append(
                {
                    "architecture": architecture_name,
                    "seed": seed,
                    "local_latent_dim": a.local_latent_dim,
                    "relational_dim": (
                        a.relational_dim
                        if use_relation
                        else None
                    ),
                    "heads": (
                        a.heads
                        if use_relation
                        else None
                    ),
                    "train_targets": train_targets,
                    "heldout_target": heldout_target,
                    "train_results": train_results,
                    "heldout": {
                        "local_latent": z.tolist(),
                        "weight_error": weight_error,
                        "behavior": behavior,
                    },
                }
            )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "experiment": (
            "Experiment 16 — true pairwise relational interaction"
        ),
        "rank": a.rank,
        "local_latent_dim": a.local_latent_dim,
        "relational_dim": a.relational_dim,
        "hidden": a.hidden,
        "heads": a.heads,
        "train_steps": a.train_steps,
        "latent_steps": a.latent_steps,
        "learning_rate": a.lr,
        "latent_learning_rate": a.latent_lr,
        "seeds": seeds,
        "train_targets": train_targets,
        "heldout_target": heldout_target,
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
    print("FINAL HELD-OUT SUMMARY")
    print("=" * 88)

    for row in results:
        h = row["heldout"]
        b = h["behavior"]

        print(
            f"{row['architecture']:28s} | "
            f"seed {row['seed']:3d} | "
            f"weight {h['weight_error']:.6f} | "
            f"greedy {b['greedy_token_match_rate']:.4f} | "
            f"exact "
            f"{b['greedy_exact_continuation_count']:2d}/"
            f"{b['greedy_total_continuations']:2d} | "
            f"RMSE {b['logit_rmse']:.6f}"
        )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
