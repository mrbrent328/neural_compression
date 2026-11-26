#!/usr/bin/env python3
"""Evaluate E49 checkpoints on the complete validation split deterministically.

Unlike the training runner's former random-batch validation, this scores every
next-token target in validation.bin exactly once using a sliding context.
"""
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from e49_model import E49Config, E49LM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--output-json", type=Path)
    return p.parse_args()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checkpoints(items):
    parsed = []
    names = set()
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid checkpoint {item!r}; expected NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or not raw_path or name in names:
            raise SystemExit(f"Invalid or duplicate checkpoint name: {item!r}")
        names.add(name)
        parsed.append((name, Path(raw_path)))
    return parsed


def load_tokens(path, context, vocab_size):
    if not path.is_file():
        raise SystemExit(f"Missing token file: {path}")
    data = np.memmap(path, dtype=np.uint32, mode="r")
    if data.size <= context + 1:
        raise SystemExit(f"Not enough tokens in {path}: {data.size}")
    if int(data.max()) >= vocab_size:
        raise SystemExit(f"Token ID in {path} exceeds vocabulary size {vocab_size}")
    return data


def evaluate_model(model, data, context, stride, batch_size, device):
    """Score every validation target once, with deterministic sliding windows."""
    model.eval()
    loss_sum = 0.0
    token_count = 0
    pending_x, pending_y = [], []
    started = time.perf_counter()

    def score_batch(xs, ys):
        nonlocal loss_sum, token_count
        inputs = torch.from_numpy(np.stack(xs)).to(device=device, dtype=torch.long)
        labels = torch.from_numpy(np.stack(ys)).to(device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        selected_logits = logits[:, context - stride:, :].contiguous()
        selected_labels = labels[:, context - stride:].contiguous()
        loss = F.cross_entropy(
            selected_logits.view(-1, selected_logits.size(-1)),
            selected_labels.view(-1),
            reduction="sum",
        )
        loss_sum += float(loss.float())
        token_count += selected_labels.numel()

    def score_single(x, y, score_from):
        nonlocal loss_sum, token_count
        inputs = torch.from_numpy(x[None, :]).to(device=device, dtype=torch.long)
        labels = torch.from_numpy(y[None, :]).to(device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        selected_logits = logits[:, score_from:, :].contiguous()
        selected_labels = labels[:, score_from:].contiguous()
        loss = F.cross_entropy(
            selected_logits.view(-1, selected_logits.size(-1)),
            selected_labels.view(-1),
            reduction="sum",
        )
        loss_sum += float(loss.float())
        token_count += selected_labels.numel()

    with torch.inference_mode():
        target_start = 0
        total_targets = len(data) - 1
        while target_start < total_targets:
            target_end = min(target_start + stride, total_targets)
            begin = max(0, target_end - context)
            tokens = np.asarray(data[begin:target_end + 1], dtype=np.int64)
            x, y = tokens[:-1], tokens[1:]
            score_count = target_end - target_start
            score_from = len(y) - score_count

            if len(x) == context and score_count == stride:
                pending_x.append(x)
                pending_y.append(y)
                if len(pending_x) == batch_size:
                    score_batch(pending_x, pending_y)
                    pending_x.clear()
                    pending_y.clear()
            else:
                if pending_x:
                    score_batch(pending_x, pending_y)
                    pending_x.clear()
                    pending_y.clear()
                score_single(x, y, score_from)
            target_start = target_end

        if pending_x:
            score_batch(pending_x, pending_y)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    mean_loss = loss_sum / max(1, token_count)
    return {
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(min(20.0, mean_loss)),
        "scored_tokens": token_count,
        "evaluation_seconds": elapsed,
        "evaluation_tokens_per_second": token_count / max(elapsed, 1e-9),
        "context_tokens": context,
        "evaluation_stride": stride,
        "minimum_context_for_steady_state_tokens": context - stride + 1,
    }


def main():
    args = parse_args()
    if args.batch_size < 1 or args.stride < 1:
        raise SystemExit("--batch-size and --stride must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/ROCm GPU visible to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This evaluator requires GPU BF16 support")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    validation_path = args.data_dir / "validation.bin"
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    validation_sha = sha256_file(validation_path)
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    if manifest:
        recorded_sha = manifest.get("splits", {}).get("validation", {}).get("sha256")
        if recorded_sha and recorded_sha != validation_sha:
            raise SystemExit("validation.bin SHA256 does not match manifest.json")

    results = []
    shared_config = None
    shared_seed = None
    for name, checkpoint_path in parse_checkpoints(args.checkpoint):
        print(f"Loading {name}: {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not {"config", "model", "mode"}.issubset(state):
            raise SystemExit(f"{checkpoint_path} is not a compatible E49 checkpoint")
        cfg = E49Config(**state["config"])
        if cfg.mode != state["mode"]:
            raise SystemExit(f"Checkpoint mode/config mismatch: {checkpoint_path}")
        if args.stride > cfg.context:
            raise SystemExit(f"--stride {args.stride} cannot exceed context {cfg.context}")
        comparable = cfg.to_dict().copy()
        comparable.pop("mode")
        if shared_config is None:
            shared_config = comparable
        elif comparable != shared_config:
            raise SystemExit("Checkpoint architectures differ beyond dense/lowrank/latent mode")

        run_config_path = checkpoint_path.parent / "run_config.json"
        run_config = json.loads(run_config_path.read_text()) if run_config_path.is_file() else None
        if run_config:
            if run_config.get("config") != cfg.to_dict() or run_config.get("mode") != state["mode"]:
                raise SystemExit(f"run_config.json does not match checkpoint: {run_config_path}")
            if manifest_sha and run_config.get("manifest_sha256") != manifest_sha:
                raise SystemExit(f"Training data manifest mismatch: {checkpoint_path}")
            seed = run_config.get("seed")
            if shared_seed is None:
                shared_seed = seed
            elif seed != shared_seed:
                raise SystemExit("Checkpoint seeds differ; refusing an unmatched comparison")

        data = load_tokens(validation_path, cfg.context, cfg.vocab_size)
        model = E49LM(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        torch.cuda.reset_peak_memory_stats(device)
        metrics = evaluate_model(model, data, cfg.context, args.stride, args.batch_size, device)
        row = {
            "name": name,
            "mode": cfg.mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_step": state.get("step"),
            "checkpoint_tokens_seen": state.get("tokens_seen"),
            "config": cfg.to_dict(),
            "parameters": model.parameter_breakdown(),
            "validation_file": str(validation_path),
            "validation_file_bytes": validation_path.stat().st_size,
            "validation_file_sha256": validation_sha,
            "manifest_sha256": manifest_sha,
            "training_seed": run_config.get("seed") if run_config else None,
            "run_config_found": run_config is not None,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            **metrics,
        }
        results.append(row)
        print(
            f"{name}: validation_loss={row['validation_loss']:.6f} "
            f"ppl={row['validation_perplexity']:.4f} "
            f"tokens={row['scored_tokens']:,} "
            f"tok/s={row['evaluation_tokens_per_second']:.1f}",
            flush=True,
        )
        del model, state, data
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "experiment": "E49-full-validation",
        "evaluation": "validation.bin; every next-token target scored once with deterministic sliding context",
        "batch_size": args.batch_size,
        "stride": args.stride,
        "results": results,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered)
        print(f"Wrote {args.output_json}", flush=True)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
