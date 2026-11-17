#!/usr/bin/env python3
"""Validate and benchmark the nine trained Neural Compression checkpoints.

Run from the neural_compression repository root:
    python validate_trained_checkpoints.py

The script is deliberately read-only with respect to existing checkpoints. It
loads each checkpoint strictly, checks it against its report, evaluates all
models on the same deterministic held-out batches, verifies latent cache
correctness on trained weights, and benchmarks actual trained inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Match the repository training runner on AMD RDNA2 (RX 6600 / gfx1032).
# This is harmless on CPU/NVIDIA and can be overridden by the caller.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch

ROOT = Path(__file__).resolve().parent
LATENT_DIR = ROOT / "latent_transformer_v0"
if str(LATENT_DIR) not in sys.path:
    sys.path.insert(0, str(LATENT_DIR))

from latent_transformer_v0 import ModelConfig, TinyLM, sample_batch, split_data, load_bytes, count_parameters
from inference_cache import (
    prepare_for_inference,
    clear_inference_cache,
    factor_cache_bytes,
)

SCALES = ("tiny", "small", "medium")
MODES = ("dense", "lowrank", "latent")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    p.add_argument("--data", type=Path, default=None, help="Validation corpus path (default: repository corpus.txt)")
    p.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--reps", type=int, default=7, help="Timed repetitions per strategy")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gen-tokens", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--prompt-tokens", type=int, default=64)
    p.add_argument("--skip-bench", action="store_true", help="Load/evaluate/test only")
    p.add_argument("--skip-eval", action="store_true", help="Skip held-out corpus evaluation")
    p.add_argument("--output", type=Path, default=None, help="JSON output path")
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(scale: str, mode: str, device: torch.device):
    run_name = f"{scale}_{mode}"
    run_dir = ROOT / "runs" / "scaling_comparison" / scale / mode / run_name
    ckpt_path = run_dir / "model.pt"
    report_path = run_dir / "report.json"
    if not ckpt_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint/report for {scale}/{mode}: {run_dir}")

    # These are this project's own checkpoints. weights_only=False is required
    # because the saved object includes ordinary Python metadata alongside tensors.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    report = json.loads(report_path.read_text())
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint structure: {ckpt_path}")

    cfg = ModelConfig(**checkpoint["model_config"])
    if cfg.mode != mode:
        raise ValueError(f"Mode mismatch: path says {mode}, checkpoint says {cfg.mode}")
    if report.get("model_config") != checkpoint["model_config"]:
        raise ValueError(f"Checkpoint/report config mismatch: {ckpt_path}")

    model = TinyLM(cfg)
    # Strict loading is important: no silent random-weight fallback.
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"State-dict mismatch for {ckpt_path}: {incompatible}")
    model.to(device)
    model.eval()

    actual_params = count_parameters(model)
    expected_params = report.get("parameter_counts", {})
    if actual_params != expected_params:
        raise ValueError(
            f"Parameter count mismatch for {ckpt_path}: actual={actual_params}, report={expected_params}"
        )
    actual_size = ckpt_path.stat().st_size
    reported_size = report.get("checkpoint_file_bytes")
    if reported_size is not None and actual_size != reported_size:
        raise ValueError(
            f"File-size mismatch for {ckpt_path}: actual={actual_size}, report={reported_size}"
        )

    return model, checkpoint, report, ckpt_path


@torch.no_grad()
def fixed_validation_loss(model: TinyLM, val_data: torch.Tensor, device: torch.device,
                          batches: int, batch_size: int) -> float:
    model.eval()
    # Reset before each checkpoint so every model sees the same sampled windows.
    torch.manual_seed(20261117)
    losses = []
    for _ in range(batches):
        x, y = sample_batch(val_data, batch_size, model.cfg.context, device)
        _, loss = model(x, y)
        losses.append(float(loss))
    return statistics.mean(losses)


@torch.no_grad()
def verify_cache_equivalence(model: TinyLM, prompt: torch.Tensor) -> dict:
    if model.cfg.mode != "latent":
        return {"applicable": False}

    model.eval()
    clear_inference_cache(model)
    reference_logits, _ = model(prompt)
    reference_tokens = model.generate(prompt.clone(), max_new_tokens=8)

    results = {}
    for strategy in ("all", "layerwise"):
        clear_inference_cache(model)
        prepare_for_inference(model, strategy=strategy)
        logits, _ = model(prompt)
        tokens = model.generate(prompt.clone(), max_new_tokens=8)
        equal_logits = torch.equal(reference_logits, logits)
        equal_tokens = torch.equal(reference_tokens, tokens)
        max_abs = float((reference_logits - logits).abs().max().item())
        cache_bytes = factor_cache_bytes(model)
        results[strategy] = {
            "logits_bit_identical": equal_logits,
            "generated_tokens_identical": equal_tokens,
            "max_abs_logit_difference": max_abs,
            "factor_cache_bytes_after_run": cache_bytes,
        }
        clear_inference_cache(model)
        if not equal_logits or not equal_tokens:
            raise AssertionError(
                f"Trained latent cache mismatch ({strategy}): logits={equal_logits}, tokens={equal_tokens}, max_abs={max_abs}"
            )
    return {"applicable": True, **results}


def timed_forward(model: TinyLM, prompt: torch.Tensor, device: torch.device,
                  reps: int, warmup: int) -> float:
    values = []
    for i in range(warmup + reps):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(prompt)
        sync(device)
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            values.append(elapsed)
    return statistics.median(values) * 1000.0


def timed_generation(model: TinyLM, prompt: torch.Tensor, device: torch.device,
                     tokens: int, reps: int, warmup: int) -> tuple[float, float]:
    durations = []
    for i in range(warmup + reps):
        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt.clone(), max_new_tokens=tokens)
        sync(device)
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            durations.append(elapsed)
    med = statistics.median(durations)
    return med * 1000.0, tokens / max(med, 1e-12)


def benchmark_trained_model(model: TinyLM, prompt: torch.Tensor, device: torch.device,
                            reps: int, warmup: int, gen_tokens: int) -> list[dict]:
    strategies = ["none"] if model.cfg.mode != "latent" else ["none", "all", "layerwise"]
    rows = []
    for strategy in strategies:
        clear_inference_cache(model)
        if strategy != "none":
            prepare_for_inference(model, strategy=strategy)
        sync(device)
        cache_bytes = factor_cache_bytes(model)
        fwd_ms = timed_forward(model, prompt, device, reps, warmup)
        gen_ms, tok_s = timed_generation(model, prompt, device, gen_tokens, reps, warmup)
        rows.append({
            "strategy": strategy,
            "forward_median_ms": round(fwd_ms, 3),
            "generation_median_ms": round(gen_ms, 3),
            "generation_tokens_per_second": round(tok_s, 2),
            "factor_cache_bytes": cache_bytes,
        })
    clear_inference_cache(model)
    model.eval()
    return rows


def main() -> int:
    args = parse_args()
    if args.reps < 1 or args.warmup < 0 or args.gen_tokens < 1:
        raise SystemExit("reps/gen-tokens must be positive and warmup non-negative")

    device = choose_device(args.device)
    corpus_path = args.data.resolve() if args.data else ROOT / "corpus.txt"
    val_data = None
    if not args.skip_eval:
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Expected project corpus at {corpus_path}")
        all_data = load_bytes(corpus_path, smoke=False)
        _, val_data = split_data(all_data)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or (ROOT / "runs" / "checkpoint_validation" / f"validation_{stamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {ROOT}")
    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print(f"Checkpoints: {', '.join(args.scales)} x {', '.join(args.modes)}")
    print(f"Evaluation: {'skipped' if args.skip_eval else f'{args.eval_batches} batches'}")
    print(f"Benchmark: {'skipped' if args.skip_bench else f'{args.reps} reps + {args.warmup} warmups, {args.gen_tokens} generated tokens'}")
    print()

    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repository": "mrbrent328/neural_compression",
        "device": str(device),
        "torch_version": torch.__version__,
        "settings": vars(args) | {"output": str(output)},
        "checkpoints": [],
    }

    for scale in args.scales:
        for mode in args.modes:
            label = f"{scale}/{mode}"
            print(f"[{label}] loading ...", flush=True)
            model, checkpoint, report, ckpt_path = load_checkpoint(scale, mode, device)
            prompt_len = min(args.prompt_tokens, model.cfg.context)
            # A fixed, deterministic prompt. Prefer actual corpus bytes.
            if corpus_path.is_file():
                raw = corpus_path.read_bytes()[:prompt_len]
                ids = list(raw)
                if len(ids) < prompt_len:
                    ids += [32] * (prompt_len - len(ids))
                prompt = torch.tensor([ids], dtype=torch.long, device=device)
            else:
                prompt = torch.arange(prompt_len, device=device).remainder(model.cfg.vocab_size).unsqueeze(0)

            item = {
                "scale": scale,
                "mode": mode,
                "checkpoint": str(ckpt_path.relative_to(ROOT)),
                "checkpoint_bytes": ckpt_path.stat().st_size,
                "checkpoint_sha256": sha256(ckpt_path),
                "strict_state_dict_load": True,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "reported_best_val_loss": min(
                    (x["val_loss"] for x in report.get("history", [])), default=None
                ),
            }

            if val_data is not None:
                item["fixed_validation_loss"] = round(
                    fixed_validation_loss(model, val_data, device,
                                          args.eval_batches, args.eval_batch_size), 6
                )

            item["cache_correctness"] = verify_cache_equivalence(model, prompt)
            if not args.skip_bench:
                item["benchmarks"] = benchmark_trained_model(
                    model, prompt, device, args.reps, args.warmup, args.gen_tokens
                )
            results["checkpoints"].append(item)
            print(f"  PASS strict load | {item['parameter_count']:,} params | {item['checkpoint_bytes']:,} bytes")
            if "fixed_validation_loss" in item:
                print(f"  fixed validation loss: {item['fixed_validation_loss']:.6f}")
            if item["cache_correctness"].get("applicable"):
                print("  PASS trained-weight cache equivalence (all + layerwise)")
            for row in item.get("benchmarks", []):
                print(f"  {row['strategy']:<9} forward={row['forward_median_ms']:>8.3f} ms | generation={row['generation_tokens_per_second']:>8.2f} tok/s | cache={row['factor_cache_bytes']:,} B")
            print()
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"Wrote results: {output}")
    print("Existing model.pt checkpoints were read only; no training or checkpoint files were overwritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
