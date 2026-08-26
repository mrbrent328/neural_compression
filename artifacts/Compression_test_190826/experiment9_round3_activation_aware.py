#!/usr/bin/env python3
"""Experiment 9 Round 3: prompt-aware low-rank compression of Qwen's Q projection.

SVD minimizes error over every direction in weight space.  This experiment
instead learns the same-size low-rank factors from the hidden-state directions
actually produced by calibration prompts, then evaluates only on held-out
prompt continuations.  It is a CPU-friendly proxy for full logit distillation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from experiment9_round2_validation import PROBES as EVALUATION_PROMPTS
from experiment9_round2_validation import assess, forced_logits, greedy, stored
from experiment9_weight_latent_functional import (
    TARGET_WEIGHT,
    inject_weight,
    load_model,
    load_safetensor_weight,
    numerical_metrics,
    set_seed,
    weight_statistics,
)


CALIBRATION_PROMPTS = (
    "Explain gravity to a curious child.",
    "Write a short haiku about rain.",
    "The purpose of a compiler is to",
    "A hexadecimal number uses the digits",
    "Describe the difference between a stack and a queue.",
    "In a healthy ecosystem, predators help",
    "The Pythagorean theorem states that",
    "A reliable experiment should include",
    "What is the difference between an adjective and an adverb?",
    "A REST API commonly uses HTTP methods such as",
    "The moon affects ocean tides because",
    "To make bread rise, bakers usually add",
    "A class in object-oriented programming is",
    "The boiling point of water depends on",
    "An algorithm is efficient when it",
    "The mitochondria are often called",
    "A map is useful because it represents",
    "A recursive function must have",
    "The next prime number after 19 is",
    "In music, a rhythm is",
    "A secure connection on the web usually begins with",
    "The central limit theorem concerns",
    "A balanced argument should acknowledge",
    "The main purpose of version control is to",
    "The opposite of increase is",
    "A dictionary in Python maps keys to",
    "The human heart pumps blood through",
    "A good error message should tell the user",
    "The area of a circle is",
    "A variable in algebra represents",
    "A test set should not be used to",
    "A telescope helps astronomers observe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("./Qwen2-0.5B"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ranks", default="64,128")
    parser.add_argument("--seeds", default="9,10,11")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-calibration-vectors", type=int, default=4096)
    parser.add_argument("--continuation-tokens", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment9_round3_results"))
    return parser.parse_args()


def tokenize(tokenizer, prompts: tuple[str, ...]) -> list[list[int]]:
    return [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]


@torch.inference_mode()
def collect_calibration_inputs(model, prompt_ids: list[list[int]], device: torch.device) -> Tensor:
    """Capture the true hidden states arriving at layer 0's q_proj."""
    q_projection = model.get_submodule(TARGET_WEIGHT.removesuffix(".weight"))
    captured: list[Tensor] = []

    def capture_inputs(_module, inputs) -> None:
        captured.append(inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1]))

    hook = q_projection.register_forward_pre_hook(capture_inputs)
    try:
        for input_ids in prompt_ids:
            tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
            model(input_ids=tokens)
    finally:
        hook.remove()
    return torch.cat(captured, dim=0).contiguous()


def train_activation_aware(
    weight: Tensor,
    inputs: Tensor,
    rank: int,
    svd: tuple[Tensor, Tensor, Tensor],
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_every: int,
) -> tuple[Tensor, dict[str, Any]]:
    left, singular_values, right = svd
    left_start, right_start = left[:, :rank] * singular_values[:rank], right[:rank]
    left = nn.Parameter(left_start.to(device))
    right = nn.Parameter(right_start.to(device))
    optimizer = torch.optim.AdamW((left, right), lr=learning_rate, weight_decay=0.0)
    calibration = inputs.to(device)
    targets = calibration @ weight.float().to(device).T
    for step in range(1, steps + 1):
        rows = torch.randint(calibration.shape[0], (min(batch_size, calibration.shape[0]),), device=device)
        predicted = (calibration[rows] @ right.T) @ left.T
        loss = F.mse_loss(predicted, targets[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            print(f"  rank={rank:4d} step={step:4d}/{steps} activation_mse={loss.item():.8g}")
    with torch.inference_mode():
        reconstruction = (left @ right).cpu().to(weight.dtype).contiguous()
        full_loss = F.mse_loss((calibration @ right.T) @ left.T, targets).item()
    return reconstruction, {"final_batch_activation_mse": float(loss.detach().cpu()), "full_calibration_activation_mse": full_loss}


def evaluate(
    reconstruction: Tensor,
    method: str,
    rank: int,
    seed: int | None,
    training: dict[str, Any] | None,
    args: argparse.Namespace,
    weight: Tensor,
    evaluation_ids: list[list[int]],
    baseline_continuations: list[list[int]],
    baseline_logits: Tensor,
    pad_id: int,
    device: torch.device,
) -> dict[str, Any]:
    clone = load_model(args.model_dir, device, weight.dtype)
    inject_weight(clone, reconstruction)
    functional = assess(clone, evaluation_ids, baseline_continuations, baseline_logits, pad_id, device)
    del clone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "method": method, "rank": rank, "seed": seed, "storage": stored(weight, rank, "autoencoder"),
        "numerical": numerical_metrics(weight, reconstruction), "training": training, "functional": functional,
    }


def aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["method"], run["rank"]), []).append(run)
    table = []
    for (method, rank), group in sorted(groups.items()):
        functionals = [item["functional"] for item in group]
        table.append({
            "method": method, "rank": rank, "runs": len(group),
            "compression_factor": group[0]["storage"]["compression_factor"],
            "relative_frobenius_mean": sum(item["numerical"]["relative_frobenius_error"] for item in group) / len(group),
            "teacher_match_mean": sum(item["teacher_forced_match_rate"] for item in functionals) / len(group),
            "greedy_match_mean": sum(item["greedy_token_match_rate"] for item in functionals) / len(group),
            "exact_continuation_mean": sum(item["greedy_exact_continuation_count"] for item in functionals) / len(group),
            "logit_rmse_min": min(item["logit_rmse"] for item in functionals),
            "logit_rmse_max": max(item["logit_rmse"] for item in functionals),
        })
    return table


def main() -> None:
    args = parse_args()
    ranks = [int(value) for value in args.ranks.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    device = torch.device(args.device)
    if not ranks or not seeds or args.steps < 1 or args.max_calibration_vectors < 1:
        raise ValueError("Ranks, seeds, steps, and calibration-vector limit must be positive.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    model_file = args.model_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Expected local model file: {model_file.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weight = load_safetensor_weight(model_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    calibration_ids = tokenize(tokenizer, CALIBRATION_PROMPTS)
    evaluation_ids = tokenize(tokenizer, EVALUATION_PROMPTS)
    print("=" * 96)
    print("EXPERIMENT 9, ROUND 3 — PROMPT-AWARE ACTIVATION COMPRESSION VS SVD")
    print("=" * 96)
    print(f"Target: {TARGET_WEIGHT} | sha256={weight_statistics(weight)['sha256']}")
    print(f"Calibration: {len(calibration_ids)} prompts | held out: {len(evaluation_ids)} prompts")

    print("Recording reference calibration activations and held-out behavior...")
    reference = load_model(args.model_dir, device, weight.dtype)
    calibration_inputs = collect_calibration_inputs(reference, calibration_ids, device)
    if calibration_inputs.shape[0] > args.max_calibration_vectors:
        generator = torch.Generator().manual_seed(2026)
        selected = torch.randperm(calibration_inputs.shape[0], generator=generator)[: args.max_calibration_vectors]
        calibration_inputs = calibration_inputs[selected]
    baseline_continuations = greedy(reference, evaluation_ids, args.continuation_tokens, tokenizer.pad_token_id, device)
    baseline_logits = forced_logits(reference, evaluation_ids, baseline_continuations, tokenizer.pad_token_id, device)
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Captured {calibration_inputs.shape[0]} calibration hidden-state vectors.")

    print("Computing exact SVD once for initialization and the control...")
    decomposition = torch.linalg.svd(weight.float(), full_matrices=False)
    runs: list[dict[str, Any]] = []
    for rank in ranks:
        if not 1 <= rank < min(weight.shape):
            raise ValueError(f"Invalid rank: {rank}")
        print(f"\nSVD control, rank {rank}...")
        left, singular_values, right = decomposition
        svd = ((left[:, :rank] * singular_values[:rank]) @ right[:rank]).to(weight.dtype).contiguous()
        runs.append(evaluate(svd, "truncated_svd", rank, None, None, args, weight, evaluation_ids, baseline_continuations, baseline_logits, tokenizer.pad_token_id, device))
        for seed in seeds:
            print(f"Activation-aware factors, rank {rank}, seed {seed}...")
            set_seed(seed)
            reconstruction, training = train_activation_aware(weight, calibration_inputs, rank, decomposition, args.steps, args.batch_size, args.learning_rate, device, args.log_every)
            runs.append(evaluate(reconstruction, "activation_aware", rank, seed, training, args, weight, evaluation_ids, baseline_continuations, baseline_logits, tokenizer.pad_token_id, device))

    table = aggregate(runs)
    report = {
        "experiment": "Experiment 9 Round 3: activation-aware low-rank factors versus SVD",
        "source_weight": weight_statistics(weight),
        "calibration_prompts": list(CALIBRATION_PROMPTS), "evaluation_prompts": list(EVALUATION_PROMPTS),
        "configuration": {"ranks": ranks, "seeds": seeds, "steps": args.steps, "calibration_vectors": calibration_inputs.shape[0], "continuation_tokens": args.continuation_tokens},
        "aggregate": table, "runs": runs,
        "decision_rule": "Activation-aware compression is useful only if it beats SVD on held-out greedy continuation agreement and logit RMSE at the same rank.",
    }
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nmethod             rank  runs  compression  teacher-match  greedy-match  exact-cont  logit-RMSE range")
    for row in table:
        print(f"{row['method']:18s} {row['rank']:4d}  {row['runs']:4d}  {row['compression_factor']:9.2f}x  {row['teacher_match_mean']:.3f}          {row['greedy_match_mean']:.3f}        {row['exact_continuation_mean']:6.1f}      {row['logit_rmse_min']:.5f}–{row['logit_rmse_max']:.5f}")
    print(f"\nDetailed JSON results: {output_path}")
    print("The on-disk checkpoint and reference model were never modified.")


if __name__ == "__main__":
    main()
