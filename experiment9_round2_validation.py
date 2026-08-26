#!/usr/bin/env python3
"""Experiment 9 Round 2: repeated autoencoders versus truncated SVD.

Run from the folder containing Qwen2-0.5B and experiment9_weight_latent_functional.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer

from experiment9_weight_latent_functional import (
    TARGET_WEIGHT,
    inject_weight,
    load_model,
    load_safetensor_weight,
    numerical_metrics,
    set_seed,
    train_autoencoder,
    weight_statistics,
)


PROBES = (
    "The capital of France is", "Two plus two equals", "The quick brown fox jumps over the",
    "A small neural network can", "The opposite of hot is", "Water freezes at",
    "The largest planet in our solar system is", "A triangle has", "The author of Romeo and Juliet was",
    "In Python, a list is written using", "def fibonacci(n):", "SELECT name FROM users WHERE",
    "A JSON object usually begins with", "The derivative of x squared is", "A synonym for happy is",
    "The chemical symbol for gold is", "The first month of the year is", "A good password should be",
    "The purpose of a unit test is to", "Machine learning models improve by", "Photosynthesis takes place in",
    "A binary search works best on data that is", "The Earth orbits the", "A database index is used to",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("./Qwen2-0.5B"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-sizes", default="16,32,64,128")
    parser.add_argument("--seeds", default="9,10,11")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--continuation-tokens", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment9_round2_results"))
    return parser.parse_args()


def pad(sequences: list[list[int]], pad_id: int, left: bool = False) -> tuple[Tensor, Tensor, list[int]]:
    lengths, width = [len(item) for item in sequences], max(len(item) for item in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        start = width - len(sequence) if left else 0
        ids[row, start : start + len(sequence)] = torch.tensor(sequence)
        mask[row, start : start + len(sequence)] = 1
    return ids, mask, lengths


@torch.inference_mode()
def greedy(model, prompts: list[list[int]], tokens: int, pad_id: int, device: torch.device) -> list[list[int]]:
    ids, mask, _ = pad(prompts, pad_id, left=True)
    ids, mask = ids.to(device), mask.to(device)
    generated = [[] for _ in prompts]
    for _ in range(tokens):
        next_ids = model(input_ids=ids, attention_mask=mask).logits[:, -1].argmax(dim=-1)
        for row, token_id in enumerate(next_ids.cpu().tolist()):
            generated[row].append(token_id)
        ids = torch.cat((ids, next_ids[:, None]), dim=1)
        mask = torch.cat((mask, torch.ones((len(prompts), 1), dtype=torch.long, device=device)), dim=1)
    return generated


@torch.inference_mode()
def forced_logits(model, prompts: list[list[int]], continuations: list[list[int]], pad_id: int, device: torch.device) -> Tensor:
    sequences = [prompt + continuation for prompt, continuation in zip(prompts, continuations, strict=True)]
    ids, mask, _ = pad(sequences, pad_id)
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float().cpu()
    positions = [torch.arange(len(prompt) - 1, len(prompt) - 1 + len(continuation)) for prompt, continuation in zip(prompts, continuations, strict=True)]
    return torch.stack([logits[row, position] for row, position in enumerate(positions)])


def assess(model, prompts: list[list[int]], baseline_continuations: list[list[int]], baseline_logits: Tensor, pad_id: int, device: torch.device) -> dict[str, Any]:
    candidate_logits = forced_logits(model, prompts, baseline_continuations, pad_id, device)
    candidate_continuations = greedy(model, prompts, len(baseline_continuations[0]), pad_id, device)
    expected = torch.tensor(baseline_continuations)
    forced_predictions = candidate_logits.argmax(dim=-1)
    free_matches = torch.tensor(candidate_continuations).eq(expected)
    delta = candidate_logits - baseline_logits
    top5 = candidate_logits.topk(5, dim=-1).indices
    return {
        "teacher_forced_match_rate": float(forced_predictions.eq(expected).float().mean()),
        "teacher_forced_top5_rate": float(top5.eq(expected.unsqueeze(-1)).any(dim=-1).float().mean()),
        "greedy_token_match_rate": float(free_matches.float().mean()),
        "greedy_exact_continuation_count": int(free_matches.all(dim=1).sum()),
        "prompt_count": len(prompts), "continuation_tokens": expected.shape[1],
        "logit_rmse": float(torch.sqrt(delta.square().mean())),
        "logit_max_abs_error": float(delta.abs().max()),
        "logit_cosine_similarity": float(F.cosine_similarity(baseline_logits.flatten(), candidate_logits.flatten(), dim=0)),
    }


def stored(weight: Tensor, rank: int, method: str) -> dict[str, Any]:
    values = 2 * 896 * rank if method == "autoencoder" else rank * (896 + 896 + 1)
    return {"compressed_values": values, "compression_factor": weight.numel() / values,
            "format": "Z + decoder" if method == "autoencoder" else "U + S + Vh"}


def summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["method"], run["rank"]), []).append(run)
    rows = []
    for (method, rank), group in sorted(groups.items()):
        metric = lambda name: [entry["functional"][name] for entry in group]
        rows.append({"method": method, "rank": rank, "runs": len(group),
                     "compression_factor": group[0]["storage"]["compression_factor"],
                     "relative_frobenius_mean": sum(item["numerical"]["relative_frobenius_error"] for item in group) / len(group),
                     "teacher_match_mean": sum(metric("teacher_forced_match_rate")) / len(group),
                     "greedy_match_mean": sum(metric("greedy_token_match_rate")) / len(group),
                     "exact_continuations_mean": sum(metric("greedy_exact_continuation_count")) / len(group),
                     "logit_rmse_min": min(metric("logit_rmse")), "logit_rmse_max": max(metric("logit_rmse"))})
    return rows


def test_reconstruction(method: str, rank: int, seed: int | None, reconstruction: Tensor, weight: Tensor, args: argparse.Namespace, prompts: list[list[int]], continuations: list[list[int]], baseline_logits: Tensor, pad_id: int, device: torch.device) -> dict[str, Any]:
    clone = load_model(args.model_dir, device, weight.dtype)
    inject_weight(clone, reconstruction)
    functional = assess(clone, prompts, continuations, baseline_logits, pad_id, device)
    del clone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"method": method, "rank": rank, "seed": seed, "storage": stored(weight, rank, method),
            "numerical": numerical_metrics(weight, reconstruction), "functional": functional}


def main() -> None:
    args = arguments()
    ranks = [int(value) for value in args.latent_sizes.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    device = torch.device(args.device)
    if not ranks or not seeds or args.steps < 1 or args.continuation_tokens < 1:
        raise ValueError("Ranks, seeds, steps, and continuation tokens must be positive.")
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
    prompts = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in PROBES]
    print("=" * 96)
    print("EXPERIMENT 9, ROUND 2 — REPEATS + SVD CONTROL + MULTI-TOKEN PROBES")
    print("=" * 96)
    print(f"Target: {TARGET_WEIGHT} | sha256={weight_statistics(weight)['sha256']}")
    print(f"Probes: {len(PROBES)} prompts × {args.continuation_tokens} tokens | ranks={ranks} | AE seeds={seeds}")
    print("Recording untouched reference continuations and logits...")
    reference = load_model(args.model_dir, device, weight.dtype)
    continuations = greedy(reference, prompts, args.continuation_tokens, tokenizer.pad_token_id, device)
    baseline_logits = forced_logits(reference, prompts, continuations, tokenizer.pad_token_id, device)
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("Computing exact SVD once for the matched low-rank control...")
    left, singular, right = torch.linalg.svd(weight.float(), full_matrices=False)
    runs: list[dict[str, Any]] = []
    for rank in ranks:
        print(f"\nSVD control, rank {rank}...")
        svd_weight = ((left[:, :rank] * singular[:rank]) @ right[:rank]).to(weight.dtype).contiguous()
        runs.append(test_reconstruction("truncated_svd", rank, None, svd_weight, weight, args, prompts, continuations, baseline_logits, tokenizer.pad_token_id, device))
        for seed in seeds:
            print(f"Autoencoder, rank {rank}, seed {seed}...")
            set_seed(seed)
            rebuilt, training = train_autoencoder(weight, rank, args.steps, args.batch_size, args.learning_rate, device, args.log_every)
            run = test_reconstruction("autoencoder", rank, seed, rebuilt, weight, args, prompts, continuations, baseline_logits, tokenizer.pad_token_id, device)
            run["training"] = training
            runs.append(run)
    rows = summary_rows(runs)
    report = {"source_weight": weight_statistics(weight), "configuration": {"ranks": ranks, "seeds": seeds, "steps": args.steps, "probe_count": len(PROBES), "continuation_tokens": args.continuation_tokens}, "aggregate": rows, "runs": runs, "decision_rule": "Only pursue rotations if they beat truncated SVD at an equal compression factor across repeated runs, especially on greedy match rate and logit RMSE."}
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nmethod         rank  runs  compression  teacher-match  greedy-match  exact-cont  logit-RMSE range")
    for row in rows:
        print(f"{row['method']:14s} {row['rank']:4d}  {row['runs']:4d}  {row['compression_factor']:9.2f}x  {row['teacher_match_mean']:.3f}          {row['greedy_match_mean']:.3f}        {row['exact_continuations_mean']:6.1f}      {row['logit_rmse_min']:.5f}–{row['logit_rmse_max']:.5f}")
    print(f"\nDetailed JSON results: {output}")
    print("The on-disk checkpoint and reference model were never modified.")


if __name__ == "__main__":
    main()
