"""
benchmark_inference.py — Inference latency benchmark for dense / lowrank / latent modes.

Measures:
  - Single-forward latency (median over N_REPS)
  - Time to first token
  - Per-token generation latency (median)
  - Tokens per second
  - Peak RSS (RAM) before and after prepare_for_inference
  - Factor-cache size (latent only)
  - Persistent checkpoint size

Usage:
    # Random-weight benchmark (no checkpoint needed):
    python benchmark_inference.py

    # Benchmark a trained checkpoint:
    python benchmark_inference.py --checkpoint path/to/model.pt

    # Specific device / dtype:
    python benchmark_inference.py --device cpu --dtype float32

Output:
    Prints a table and writes benchmark_inference_results.json in the current dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import List, Optional

import torch
from torch import Tensor

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from latent_transformer_v0 import (
    ModelConfig,
    TinyLM,
    choose_device,
    dtype_from_name,
)
from inference_cache import (
    prepare_for_inference,
    clear_inference_cache,
    factor_cache_bytes,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_REPS      = 31   # number of timed repetitions (median taken)
WARMUP_REPS = 3    # discard first N reps
GEN_TOKENS  = 64   # tokens to generate per timing run
GEN_PROMPT  = 8    # prompt length in tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference benchmark")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to a model.pt checkpoint.  If not given, uses random weights.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype",  default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--n-reps",   type=int, default=N_REPS)
    p.add_argument("--warmup",   type=int, default=WARMUP_REPS)
    p.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    p.add_argument("--gen-prompt", type=int, default=GEN_PROMPT)
    p.add_argument("--out", type=Path, default=Path("benchmark_inference_results.json"))
    # Override model dimensions (only used when no checkpoint is given)
    p.add_argument("--layers",   type=int, default=6)
    p.add_argument("--d-model",  type=int, default=384)
    p.add_argument("--d-ff",     type=int, default=1536)
    p.add_argument("--heads",    type=int, default=6)
    p.add_argument("--rank",     type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--constructor-hidden", type=int, default=128)
    p.add_argument("--modes", nargs="+",
                   choices=["dense","lowrank","latent","latent_all","latent_layerwise"],
                   default=["dense","lowrank","latent","latent_all","latent_layerwise"],
                   help="Which configurations to benchmark")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rss_bytes() -> int:
    """Current RSS in bytes (Linux/macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1024 if sys.platform != "darwin" else 1
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(model: TinyLM, idx: Tensor, device: torch.device) -> float:
    """Return wall-clock seconds for one forward pass."""
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(idx)
    sync(device)
    return time.perf_counter() - t0


def timed_generate(model: TinyLM, idx: Tensor,
                   n_tokens: int, device: torch.device) -> tuple[float, float, List[float]]:
    """
    Return (time_to_first_token, total_seconds, per_token_times).
    per_token_times[0] is first token, rest are subsequent.
    """
    per_token: List[float] = []
    sync(device)
    t_start = time.perf_counter()
    with torch.no_grad():
        cur = idx
        for i in range(n_tokens):
            t0 = time.perf_counter()
            crop   = cur[:, -model.cfg.context:]
            logits, _ = model(crop)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur = torch.cat((cur, next_tok), dim=1)
            sync(device)
            per_token.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_start
    return per_token[0], total, per_token


def checkpoint_bytes(model: TinyLM, tmp_path: Path) -> int:
    """Save a checkpoint and return its file size in bytes."""
    from dataclasses import asdict
    ckpt = {"model_config": asdict(model.cfg), "model_state": model.state_dict()}
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, tmp_path)
    size = tmp_path.stat().st_size
    tmp_path.unlink()
    return size


def load_or_build(args: argparse.Namespace, mode: str,
                  device: torch.device, dtype: torch.dtype) -> TinyLM:
    """Load a checkpoint or build a random-weight model."""
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg  = ModelConfig(**ckpt["model_config"])
        cfg.mode = mode   # allow benchmarking all modes from one checkpoint config
        model = TinyLM(cfg)
        # Only load weights whose keys match (attention / embeddings are shared across modes)
        state = ckpt["model_state"]
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            pass  # random fallback for mismatched modes
        source = f"checkpoint:{args.checkpoint.name}"
    else:
        cfg = ModelConfig(
            vocab_size=256, context=256,
            n_layers=args.layers, d_model=args.d_model,
            n_heads=args.heads, d_ff=args.d_ff,
            dropout=0.0, mode=mode,
            rank=args.rank, latent_dim=args.latent_dim,
            constructor_hidden=args.constructor_hidden,
        )
        torch.manual_seed(1337)
        model = TinyLM(cfg)
        source = "random_weights"

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, source


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def benchmark_one(
    label: str,
    model: TinyLM,
    strategy: Optional[str],   # None / "all" / "layerwise"
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """Run the full benchmark suite for one (model, strategy) pair."""
    torch.manual_seed(42)
    prompt = torch.randint(0, model.cfg.vocab_size,
                           (1, args.gen_prompt), device=device)

    # --- factor cache setup ---
    cache_mb = 0.0
    if model.cfg.mode == "latent" and strategy in ("all", "layerwise"):
        clear_inference_cache(model)
        prepare_for_inference(model, strategy=strategy)
        cache_mb = factor_cache_bytes(model) / 1e6

    # --- checkpoint size ---
    ckpt_mb = checkpoint_bytes(
        model,
        Path("/tmp/_bench_ckpt_tmp.pt")
    ) / 1e6

    # --- forward latency ---
    fwd_times = []
    for i in range(args.warmup + args.n_reps):
        t = timed_forward(model, prompt, device)
        if i >= args.warmup:
            fwd_times.append(t)

    fwd_median_ms = median(fwd_times) * 1000

    # --- generation latency ---
    # Time-to-first-token: single token from scratch
    ttft_times = []
    for i in range(args.warmup + args.n_reps):
        t, _, _ = timed_generate(model, prompt, 1, device)
        if i >= args.warmup:
            ttft_times.append(t)
    ttft_ms = median(ttft_times) * 1000

    # Full generation run for per-token and tok/s
    gen_per_tokens: List[List[float]] = []
    for i in range(args.warmup + args.n_reps):
        _, total, per = timed_generate(model, prompt, args.gen_tokens, device)
        if i >= args.warmup:
            gen_per_tokens.append(per)

    # Median subsequent-token latency (excluding first token)
    sub_medians = [median(r[1:]) for r in gen_per_tokens]
    sub_ms      = median(sub_medians) * 1000

    # Tokens per second: gen_tokens / total_time
    total_times = [sum(r) for r in gen_per_tokens]
    tps         = median([args.gen_tokens / t for t in total_times])

    # --- RAM ---
    gc.collect()
    rss_mb = rss_bytes() / 1e6

    result = {
        "label":           label,
        "mode":            model.cfg.mode,
        "cache_strategy":  strategy or "none",
        "n_layers":        model.cfg.n_layers,
        "d_model":         model.cfg.d_model,
        "d_ff":            model.cfg.d_ff,
        "rank":            model.cfg.rank,
        "latent_dim":      model.cfg.latent_dim,
        "forward_ms":      round(fwd_median_ms, 3),
        "ttft_ms":         round(ttft_ms, 3),
        "subsequent_ms":   round(sub_ms, 3),
        "tokens_per_sec":  round(tps, 1),
        "ckpt_mb":         round(ckpt_mb, 2),
        "factor_cache_mb": round(cache_mb, 2),
        "rss_mb":          round(rss_mb, 1),
        "n_reps":          args.n_reps,
        "gen_tokens":      args.gen_tokens,
    }

    # Clean up
    if model.cfg.mode == "latent":
        clear_inference_cache(model)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = choose_device(args.device)
    dtype  = dtype_from_name(args.dtype)

    print("=" * 72)
    print("INFERENCE BENCHMARK")
    print("=" * 72)
    print(f"device  : {device}")
    print(f"dtype   : {dtype}")
    print(f"reps    : {args.n_reps} (warmup {args.warmup})")
    print(f"gen_tok : {args.gen_tokens}  prompt : {args.gen_prompt} tokens")
    if args.checkpoint:
        print(f"ckpt    : {args.checkpoint}")
    else:
        print(f"weights : random (L={args.layers} D={args.d_model} rank={args.rank})")
    print()

    # Build the job list
    jobs: list[tuple[str, str, Optional[str]]] = []
    for m in args.modes:
        if m == "latent_all":
            jobs.append(("latent_all_cached",   "latent", "all"))
        elif m == "latent_layerwise":
            jobs.append(("latent_layerwise",    "latent", "layerwise"))
        elif m == "latent":
            jobs.append(("latent_uncached",     "latent", None))
        else:
            jobs.append((m,                     m,        None))

    results = []
    for label, mode, strategy in jobs:
        print(f"  Benchmarking {label} ...")
        model, source = load_or_build(args, mode, device, dtype)
        r = benchmark_one(label, model, strategy, device, args)
        r["weight_source"] = source
        results.append(r)
        del model
        gc.collect()

    # --- Print table ---
    print()
    print(f"{'Label':<22} {'Fwd ms':>8} {'TTFT ms':>9} {'Sub ms':>8} "
          f"{'tok/s':>8} {'Ckpt MB':>8} {'Cache MB':>9}")
    print("-" * 78)
    for r in results:
        print(f"{r['label']:<22} {r['forward_ms']:>8.2f} {r['ttft_ms']:>9.2f} "
              f"{r['subsequent_ms']:>8.2f} {r['tokens_per_sec']:>8.1f} "
              f"{r['ckpt_mb']:>8.2f} {r['factor_cache_mb']:>9.2f}")

    print()
    print("Notes:")
    print("  Fwd ms      = median forward-pass wall time (single token prompt)")
    print("  TTFT ms     = median time to first generated token")
    print("  Sub ms      = median per-token latency for tokens 2+ in generation")
    print("  tok/s       = median tokens/sec over full generation run")
    print("  Ckpt MB     = serialized checkpoint file size")
    print("  Cache MB    = runtime factor-tensor memory (latent only)")
    print(f"  All medians over {args.n_reps} reps after {args.warmup} warmup reps")
    print(f"  Weights: {results[0]['weight_source']}")

    # --- Analysis ---
    by_label = {r["label"]: r for r in results}

    if "dense" in by_label and "latent_uncached" in by_label:
        d = by_label["dense"]
        u = by_label["latent_uncached"]
        print()
        print("=== latent_uncached vs dense ===")
        print(f"  Forward overhead:   {u['forward_ms']/d['forward_ms']:.2f}x slower")
        print(f"  Checkpoint saving:  {d['ckpt_mb']/u['ckpt_mb']:.2f}x larger for dense")

    if "latent_uncached" in by_label and "latent_all_cached" in by_label:
        u = by_label["latent_uncached"]
        c = by_label["latent_all_cached"]
        print()
        print("=== latent_all_cached vs latent_uncached ===")
        speedup = u["forward_ms"] / c["forward_ms"]
        print(f"  Forward speedup:    {speedup:.2f}x")
        print(f"  tok/s gain:         {c['tokens_per_sec']/u['tokens_per_sec']:.2f}x")
        print(f"  Cache overhead:     {c['factor_cache_mb']:.2f} MB")

    if "latent_all_cached" in by_label and "latent_layerwise" in by_label:
        a = by_label["latent_all_cached"]
        lw = by_label["latent_layerwise"]
        print()
        print("=== latent_layerwise vs latent_all_cached ===")
        print(f"  Forward ratio:      {lw['forward_ms']/a['forward_ms']:.2f}x")
        print(f"  Persistent cache:   all={a['factor_cache_mb']:.2f}MB  "
              f"layerwise={lw['factor_cache_mb']:.2f}MB (should be ~0 at rest)")

    if "dense" in by_label and "latent_all_cached" in by_label:
        d  = by_label["dense"]
        c  = by_label["latent_all_cached"]
        print()
        print("=== latent_all_cached vs dense ===")
        print(f"  Forward ratio:      {c['forward_ms']/d['forward_ms']:.2f}x")
        print(f"  tok/s ratio:        {c['tokens_per_sec']/d['tokens_per_sec']:.2f}x")
        print(f"  Ckpt ratio:         dense={d['ckpt_mb']:.1f}MB  "
              f"latent={c['ckpt_mb']:.1f}MB  "
              f"(+cache {c['factor_cache_mb']:.1f}MB = {c['ckpt_mb']+c['factor_cache_mb']:.1f}MB)")

    # --- Write JSON ---
    out = {
        "hardware": {
            "platform": sys.platform,
            "device":   str(device),
            "dtype":    str(dtype),
            "torch":    torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "config": {
            "n_reps":     args.n_reps,
            "warmup":     args.warmup,
            "gen_tokens": args.gen_tokens,
            "gen_prompt": args.gen_prompt,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
