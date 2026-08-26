#!/usr/bin/env python3
"""Experiment 9: compress one real Qwen2 weight matrix and test behavior.

This script deliberately uses a fresh model instance for every reconstructed
weight.  The reference model is only used to record a behavioral fingerprint;
its parameters are never modified.

Example (CPU-first):
    python experiment9_weight_latent_functional.py

For a quicker dry run:
    python experiment9_weight_latent_functional.py --steps 25 --latent-sizes 16,32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_WEIGHT = "model.layers.0.self_attn.q_proj.weight"
EXPECTED_SHAPE = (896, 896)
PROBES = (
    "The capital of France is",
    "In one sentence, explain why the sky appears blue.",
    "def fibonacci(n):",
    "The quick brown fox jumps over the",
    "A small neural network can",
    "Two plus two equals",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("./Qwen2-0.5B"))
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda, cuda:0")
    parser.add_argument("--latent-sizes", default="16,32,64,128")
    parser.add_argument("--steps", type=int, default=250, help="Optimizer steps per latent size")
    parser.add_argument("--batch-size", type=int, default=128, help="Rows per training step")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment9_results"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_sha256(tensor: Tensor) -> str:
    raw_bytes = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def weight_statistics(weight: Tensor) -> dict[str, Any]:
    values = weight.float()
    return {
        "shape": list(weight.shape),
        "dtype": str(weight.dtype).replace("torch.", ""),
        "numel": weight.numel(),
        "raw_bytes": weight.numel() * weight.element_size(),
        "sha256": tensor_sha256(weight),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "frobenius_norm": float(torch.linalg.vector_norm(values)),
    }


def load_safetensor_weight(model_file: Path) -> Tensor:
    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        if TARGET_WEIGHT not in handle.keys():
            raise KeyError(f"{TARGET_WEIGHT} was not found in {model_file}")
        weight = handle.get_tensor(TARGET_WEIGHT).contiguous()
    if tuple(weight.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Expected {EXPECTED_SHAPE}; got {tuple(weight.shape)}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"Expected bfloat16; got {weight.dtype}")
    return weight


def load_model(model_dir: Path, device: torch.device, source_dtype: torch.dtype):
    # CPU float32 is broadly supported; both reference and reconstructed models
    # use the same inference dtype, while their stored values remain bfloat16.
    inference_dtype = torch.float32 if device.type == "cpu" else source_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=source_dtype,
    )
    model.to(device=device, dtype=inference_dtype)
    model.eval()
    return model


@torch.inference_mode()
def fingerprint(model, tokenizer, device: torch.device) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt in PROBES:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {name: value.to(device) for name, value in encoded.items()}
        final_logits = model(**encoded).logits[0, -1].float().cpu()
        top_values, top_ids = torch.topk(final_logits, k=5)
        records.append(
            {
                "prompt": prompt,
                "token_ids": encoded["input_ids"].cpu().tolist()[0],
                "logits": final_logits,
                "next_token_id": int(top_ids[0]),
                "next_token": tokenizer.decode([int(top_ids[0])]),
                "top5_token_ids": [int(token_id) for token_id in top_ids],
                "top5_tokens": [tokenizer.decode([int(token_id)]) for token_id in top_ids],
                "top5_logits": [float(value) for value in top_values],
            }
        )
    return records


def train_autoencoder(
    weight: Tensor,
    latent_size: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_every: int,
) -> tuple[Tensor, dict[str, Any]]:
    rows, columns = weight.shape
    if not 1 <= latent_size < min(rows, columns):
        raise ValueError(f"latent size must be between 1 and {min(rows, columns) - 1}")
    data = weight.float().to(device)
    encoder = nn.Linear(columns, latent_size, bias=False, device=device)
    decoder = nn.Linear(latent_size, columns, bias=False, device=device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=0.0
    )

    started = time.perf_counter()
    for step in range(1, steps + 1):
        row_indices = torch.randint(rows, (min(batch_size, rows),), device=device)
        batch = data[row_indices]
        reconstruction = decoder(encoder(batch))
        loss = F.mse_loss(reconstruction, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == steps or step % log_every == 0:
            print(f"  latent={latent_size:4d} step={step:4d}/{steps} train_mse={loss.item():.8g}")

    # The compressed artifact is Z plus a decoder.  The encoder is training-only.
    with torch.inference_mode():
        latent_codes = encoder(data)
        reconstruction = decoder(latent_codes).detach().cpu().to(dtype=weight.dtype).contiguous()
    return reconstruction, {
        "training_seconds": time.perf_counter() - started,
        "final_train_mse": float(loss.detach().cpu()),
        "compressed_values": latent_codes.numel() + decoder.weight.numel(),
        "encoder_training_only_values": encoder.weight.numel(),
    }


def numerical_metrics(reference: Tensor, reconstruction: Tensor) -> dict[str, float]:
    delta = reconstruction.float() - reference.float()
    reference_norm = torch.linalg.vector_norm(reference.float())
    return {
        "mse": float(torch.mean(delta.square())),
        "mae": float(torch.mean(delta.abs())),
        "max_abs_error": float(delta.abs().max()),
        "relative_frobenius_error": float(torch.linalg.vector_norm(delta) / reference_norm),
        "cosine_similarity": float(F.cosine_similarity(reference.float().flatten(), reconstruction.float().flatten(), dim=0)),
    }


def functional_metrics(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    per_probe = []
    all_deltas = []
    exact_next_tokens = 0
    for baseline, rebuilt in zip(reference, candidate, strict=True):
        delta = rebuilt["logits"] - baseline["logits"]
        all_deltas.append(delta)
        exact = baseline["next_token_id"] == rebuilt["next_token_id"]
        exact_next_tokens += int(exact)
        baseline_top5 = set(baseline["top5_token_ids"])
        rebuilt_top5 = set(rebuilt["top5_token_ids"])
        per_probe.append(
            {
                "prompt": baseline["prompt"],
                "baseline_next_token": baseline["next_token"],
                "reconstructed_next_token": rebuilt["next_token"],
                "next_token_match": exact,
                "top5_overlap": len(baseline_top5 & rebuilt_top5),
                "logit_rmse": float(torch.sqrt(torch.mean(delta.square()))),
                "logit_max_abs_error": float(delta.abs().max()),
                "logit_cosine_similarity": float(
                    F.cosine_similarity(baseline["logits"], rebuilt["logits"], dim=0)
                ),
            }
        )
    flattened = torch.cat(all_deltas)
    return {
        "next_token_match_count": exact_next_tokens,
        "probe_count": len(reference),
        "next_token_match_rate": exact_next_tokens / len(reference),
        "overall_logit_rmse": float(torch.sqrt(torch.mean(flattened.square()))),
        "overall_logit_max_abs_error": float(flattened.abs().max()),
        "per_probe": per_probe,
    }


def inject_weight(model, reconstruction: Tensor) -> None:
    target = model.get_parameter(TARGET_WEIGHT)
    replacement = reconstruction.to(device=target.device, dtype=target.dtype)
    if replacement.shape != target.shape:
        raise ValueError(f"Replacement shape {tuple(replacement.shape)} != {tuple(target.shape)}")
    with torch.no_grad():
        target.copy_(replacement)


def serializable_fingerprint(fingerprint_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in record.items() if key != "logits"} for record in fingerprint_records]


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    latent_sizes = [int(value) for value in args.latent_sizes.split(",") if value.strip()]
    if not latent_sizes:
        raise ValueError("--latent-sizes must contain at least one integer")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    model_file = args.model_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Expected local model file: {model_file.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("=" * 88)
    print("EXPERIMENT 9 — WEIGHT → LATENT → WEIGHT + FUNCTIONAL EQUIVALENCE")
    print("=" * 88)
    print(f"Model file: {model_file}")
    print(f"Device: {device}")
    source_weight = load_safetensor_weight(model_file)
    source_stats = weight_statistics(source_weight)
    print(f"Target: {TARGET_WEIGHT} {tuple(source_weight.shape)} {source_weight.dtype}")
    print(f"Exact source SHA-256: {source_stats['sha256']}")
    print(f"Stats: min={source_stats['min']:.6g} max={source_stats['max']:.6g} "
          f"mean={source_stats['mean']:.6g} std={source_stats['std']:.6g}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Recording reference behavioral fingerprint (fresh, unmodified model)...")
    reference_model = load_model(args.model_dir, device, source_weight.dtype)
    loaded_reference = reference_model.get_parameter(TARGET_WEIGHT).detach().cpu().to(source_weight.dtype)
    if not torch.equal(loaded_reference, source_weight):
        raise RuntimeError("The loaded reference weight differs from model.safetensors; stopping.")
    baseline = fingerprint(reference_model, tokenizer, device)
    del reference_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results: list[dict[str, Any]] = []
    original_bytes = source_stats["raw_bytes"]
    for latent_size in latent_sizes:
        print(f"\nTraining latent size {latent_size}...")
        reconstruction, training = train_autoencoder(
            source_weight, latent_size, args.steps, args.batch_size, args.learning_rate, device, args.log_every
        )
        numeric = numerical_metrics(source_weight, reconstruction)
        compressed_bytes = training["compressed_values"] * source_weight.element_size()

        print("Evaluating reconstructed weight in a new cloned model...")
        clone = load_model(args.model_dir, device, source_weight.dtype)
        inject_weight(clone, reconstruction)
        rebuilt = fingerprint(clone, tokenizer, device)
        functional = functional_metrics(baseline, rebuilt)
        del clone
        if device.type == "cuda":
            torch.cuda.empty_cache()

        result = {
            "latent_size": latent_size,
            "storage": {
                "original_weight_values": source_weight.numel(),
                "original_weight_bytes": original_bytes,
                "compressed_values": training["compressed_values"],
                "compressed_bytes_assuming_bfloat16": compressed_bytes,
                "compressed_to_original_ratio": compressed_bytes / original_bytes,
                "compression_factor": original_bytes / compressed_bytes,
                "format": "latent_codes[896, latent] + decoder_weight[896, latent]",
                "note": "Encoder parameters are excluded because they are training-only.",
            },
            "training": training,
            "numerical": numeric,
            "functional": functional,
        }
        results.append(result)
        print(f"  relative_Frobenius={numeric['relative_frobenius_error']:.6f} "
              f"next-token matches={functional['next_token_match_count']}/{functional['probe_count']} "
              f"logit_RMSE={functional['overall_logit_rmse']:.6f} "
              f"storage={result['storage']['compression_factor']:.2f}x")

    summary = {
        "experiment": "Experiment 9: Weight → Latent → Weight + Functional Equivalence",
        "configuration": {
            "model_dir": str(args.model_dir), "model_file": str(model_file), "device": str(device),
            "latent_sizes": latent_sizes, "steps": args.steps, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "seed": args.seed,
        },
        "source_weight": source_stats,
        "baseline_fingerprint": serializable_fingerprint(baseline),
        "results": results,
        "interpretation": (
            "Use numerical error, compression factor, next-token match rate, and logit RMSE together. "
            "A promising next rotation/alternate-coordinate experiment should improve the functional metrics "
            "at a fixed compressed-to-original storage ratio, not merely reduce matrix MSE."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 88)
    print("SUMMARY (smaller functional logit error and higher matches are better)")
    print("latent  compression  rel_Frobenius  next-token-match  logit-RMSE")
    for result in results:
        print(f"{result['latent_size']:6d}  {result['storage']['compression_factor']:9.2f}x  "
              f"{result['numerical']['relative_frobenius_error']:13.6f}  "
              f"{result['functional']['next_token_match_count']:2d}/{result['functional']['probe_count']:<15d}  "
              f"{result['functional']['overall_logit_rmse']:.6f}")
    print(f"\nDetailed, JSON-safe results written to: {summary_path}")
    print("The original model instance and on-disk safetensors file were never modified.")


if __name__ == "__main__":
    main()
