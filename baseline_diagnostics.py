#!/usr/bin/env python3
"""
BASELINE DIAGNOSTICS — held-out Q projection

Purpose:
1. Establish the behavioral floor/ceiling of the 24-prompt evaluation.
2. Measure how sensitive each prompt is to the held-out Q projection.
3. Separate degradation caused by low-rank factorization from degradation
   introduced by the latent/constructor system.

Conditions:
    ORIGINAL
    ZERO
    RANDOM
    LOW_RANK_48
    LOW_RANK_96

The script reports both aggregate metrics and per-prompt token matches.
The per-prompt table is the important part for testing the suspected
"12/24 floor" effect.

No constructor or latent optimization is involved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TARGET = "model.layers.4.self_attn.q_proj.weight"

PROMPTS = [
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


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("./Qwen2-0.5B"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--rank-a", type=int, default=48)
    p.add_argument("--rank-b", type=int, default=96)
    p.add_argument("--new-tokens", type=int, default=6)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline_diagnostics"),
    )
    return p.parse_args()


def load_weights(model_dir):
    path = model_dir / "model.safetensors"
    return load_file(str(path), device="cpu")


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


def install_weight(model, target, weight):
    module = model.get_submodule(target.rsplit(".weight", 1)[0])
    with torch.no_grad():
        module.weight.copy_(
            weight.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        )


@torch.inference_mode()
def generate(model, tokenizer, prompts, new_tokens, device):
    results = []

    for prompt in prompts:
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)

        out = model.generate(
            ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        continuation = out[0, ids.shape[1]:].detach().cpu()
        results.append(continuation)

    return results


@torch.inference_mode()
def logits_for_continuations(
    model,
    tokenizer,
    prompts,
    continuations,
    device,
):
    results = []

    for prompt, continuation in zip(
        prompts,
        continuations,
    ):
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        full = torch.tensor(
            prompt_ids,
            dtype=torch.long,
        )

        full = torch.cat(
            [
                full,
                continuation.long(),
            ]
        )

        if continuation.numel() == 0:
            results.append(
                torch.empty((0, 0))
            )
            continue

        logits = model(
            input_ids=full[:-1].unsqueeze(0).to(device)
        ).logits[0]

        results.append(
            logits[-continuation.numel():]
            .float()
            .cpu()
        )

    return results


def compare(reference, candidate):
    rows = []

    total = 0
    matches = 0
    exact = 0
    mse_sum = 0.0
    mse_n = 0

    for i, (ref, cand) in enumerate(
        zip(reference["continuations"], candidate["continuations"])
    ):
        n = min(ref.numel(), cand.numel())

        token_matches = int(
            (ref[:n] == cand[:n]).sum()
        ) if n else 0

        exact_match = (
            ref.numel() == cand.numel()
            and torch.equal(ref, cand)
        )

        ref_logits = reference["logits"][i]
        cand_logits = candidate["logits"][i]

        m = min(
            ref_logits.shape[0],
            cand_logits.shape[0],
        )

        if m:
            diff = ref_logits[:m] - cand_logits[:m]
            mse_sum += float((diff * diff).sum())
            mse_n += diff.numel()

        total += n
        matches += token_matches
        exact += int(exact_match)

        rows.append(
            {
                "prompt_index": i + 1,
                "prompt": PROMPTS[i],
                "reference_tokens": ref.tolist(),
                "candidate_tokens": cand.tolist(),
                "matching_tokens": token_matches,
                "compared_tokens": n,
                "exact": int(exact_match),
            }
        )

    return {
        "greedy_match": matches / total if total else 0.0,
        "exact": exact,
        "total_prompts": len(reference["continuations"]),
        "logit_rmse": (
            mse_sum / mse_n
        ) ** 0.5 if mse_n else float("nan"),
        "per_prompt": rows,
    }


def relative_error(original, candidate):
    return float(
        torch.linalg.vector_norm(
            original.float() - candidate.float()
        )
        / torch.linalg.vector_norm(
            original.float()
        )
    )


def low_rank(weight, rank):
    u, s, vh = torch.linalg.svd(
        weight.float(),
        full_matrices=False,
    )

    r = min(rank, s.numel())

    return (
        u[:, :r] * s[:r]
    ) @ vh[:r]


def main():
    a = args()
    device = torch.device(a.device)

    a.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("BASELINE DIAGNOSTICS — Q-PROJECTION BEHAVIORAL SENSITIVITY")
    print("=" * 88)
    print(f"Model  : {a.model_dir}")
    print(f"Target : {a.target}")
    print(f"Device : {device}")
    print(f"Prompts: {len(PROMPTS)}")
    print(
        f"Controls: original, zero, random, "
        f"rank-{a.rank_a}, rank-{a.rank_b}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        a.model_dir,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tensors = load_weights(a.model_dir)

    if a.target not in tensors:
        raise KeyError(
            f"{a.target} not found in model.safetensors"
        )

    original = tensors[a.target].float().contiguous()

    print(
        f"\nWeight shape: {tuple(original.shape)}"
    )

    # Use a deterministic generator so the random baseline is reproducible.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)

    random_weight = torch.randn(
        original.shape,
        generator=generator,
        dtype=torch.float32,
    )

    # Match the original matrix's mean/std so "random" is not trivially
    # distinguished only by scale.
    random_weight = (
        random_weight
        - random_weight.mean()
    ) / random_weight.std().clamp_min(1e-8)

    random_weight = (
        random_weight * original.std()
        + original.mean()
    )

    zero_weight = torch.zeros_like(original)
    rank_a = low_rank(original, a.rank_a)
    rank_b = low_rank(original, a.rank_b)

    weights = {
        "original": original,
        "zero": zero_weight,
        "random": random_weight,
        f"low_rank_{a.rank_a}": rank_a,
        f"low_rank_{a.rank_b}": rank_b,
    }

    print("\n[1] Numerical controls")

    numerical = {}

    for name, weight in weights.items():
        numerical[name] = {
            "relative_weight_error": relative_error(
                original,
                weight,
            ),
        }

        print(
            f"  {name:12s} | "
            f"weight error = "
            f"{numerical[name]['relative_weight_error']:.8f}"
        )

    print("\n[2] Reference behavior")

    reference_model = load_model(
        a.model_dir,
        device,
    )

    reference_continuations = generate(
        reference_model,
        tokenizer,
        PROMPTS,
        a.new_tokens,
        device,
    )

    reference_logits = logits_for_continuations(
        reference_model,
        tokenizer,
        PROMPTS,
        reference_continuations,
        device,
    )

    reference = {
        "continuations": reference_continuations,
        "logits": reference_logits,
    }

    del reference_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {}

    print("\n[3] Behavioral controls")

    for name, weight in weights.items():
        if name == "original":
            continue

        print("\n" + "-" * 88)
        print(name.upper())
        print("-" * 88)

        model = load_model(
            a.model_dir,
            device,
        )

        install_weight(
            model,
            a.target,
            weight,
        )

        continuations = generate(
            model,
            tokenizer,
            PROMPTS,
            a.new_tokens,
            device,
        )

        logits = logits_for_continuations(
            model,
            tokenizer,
            PROMPTS,
            reference_continuations,
            device,
        )

        candidate = {
            "continuations": continuations,
            "logits": logits,
        }

        metrics = compare(
            reference,
            candidate,
        )

        results[name] = {
            **numerical[name],
            "greedy_match": metrics["greedy_match"],
            "exact": metrics["exact"],
            "total_prompts": metrics["total_prompts"],
            "logit_rmse": metrics["logit_rmse"],
            "per_prompt": metrics["per_prompt"],
        }

        print(
            f"weight error = "
            f"{numerical[name]['relative_weight_error']:.8f}"
        )
        print(
            f"greedy match = "
            f"{metrics['greedy_match']:.4f}"
        )
        print(
            f"exact        = "
            f"{metrics['exact']}/{metrics['total_prompts']}"
        )
        print(
            f"logit RMSE   = "
            f"{metrics['logit_rmse']:.8f}"
        )

        print("\nPer-prompt:")
        for row in metrics["per_prompt"]:
            print(
                f"  {row['prompt_index']:2d} | "
                f"{'EXACT' if row['exact'] else 'diff '} | "
                f"{row['matching_tokens']}/"
                f"{row['compared_tokens']} | "
                f"{row['prompt']}"
            )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print("FINAL BASELINE SUMMARY")
    print("=" * 88)

    print(
        f"{'condition':14s} | "
        f"{'weight':>10s} | "
        f"{'greedy':>8s} | "
        f"{'exact':>8s} | "
        f"{'RMSE':>10s}"
    )

    print("-" * 88)

    for name in ["original", *results.keys()]:
        if name == "original":
            print(
                f"{name:14s} | "
                f"{0.0:10.6f} | "
                f"{1.0:8.4f} | "
                f"{len(PROMPTS):4d}/{len(PROMPTS):<3d} | "
                f"{0.0:10.6f}"
            )
        else:
            r = results[name]
            print(
                f"{name:14s} | "
                f"{r['relative_weight_error']:10.6f} | "
                f"{r['greedy_match']:8.4f} | "
                f"{r['exact']:4d}/{r['total_prompts']:<3d} | "
                f"{r['logit_rmse']:10.6f}"
            )

    # Save a machine-readable result AND a simple per-prompt CSV.
    import json

    summary = {
        "model_dir": str(a.model_dir),
        "target": a.target,
        "rank_a": a.rank_a,
        "rank_b": a.rank_b,
        "new_tokens": a.new_tokens,
        "prompt_count": len(PROMPTS),
        "results": results,
    }

    json_path = a.output_dir / "baseline_summary.json"
    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    csv_path = a.output_dir / "per_prompt_baselines.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "prompt_index",
                "prompt",
                "condition",
                "exact",
                "matching_tokens",
                "compared_tokens",
            ]
        )

        for condition, data in results.items():
            for row in data["per_prompt"]:
                writer.writerow(
                    [
                        row["prompt_index"],
                        row["prompt"],
                        condition,
                        row["exact"],
                        row["matching_tokens"],
                        row["compared_tokens"],
                    ]
                )

    print(
        f"\nSaved:\n"
        f"  {json_path}\n"
        f"  {csv_path}"
    )


if __name__ == "__main__":
    main()
