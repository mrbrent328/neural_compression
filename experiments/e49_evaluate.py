#!/usr/bin/env python3
"""Evaluate one or more E49 checkpoints on the untouched test.bin split.

Example:
  python experiments/e49_evaluate.py \\
    --data-dir data/e49_tokens \\
    --checkpoint dense=runs/e49_local_dense/latest.pt \\
    --checkpoint lowrank=runs/e49_local_lowrank/latest.pt \\
    --checkpoint latent=runs/e49_local_latent/latest.pt \\
    --output-json runs/e49_test_comparison.json
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

from e49_model import E49Config, E49LM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Checkpoint to evaluate; repeat for each model, e.g. dense=runs/dense/latest.pt",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--stride", type=int, default=32, help="New targets per window; smaller uses more context")
    p.add_argument("--output-json", type=Path)
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoints(values):
    parsed = []
    names = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --checkpoint {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or name in names:
            raise SystemExit(f"Checkpoint names must be non-empty and unique: {name!r}")
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found: {path}")
        names.add(name)
        parsed.append((name, path))
    return parsed


def load_test_tokens(path, context, vocab_size):
    if not path.is_file():
        raise SystemExit(f"Missing held-out test split: {path}")
    data = np.memmap(path, dtype=np.uint32, mode="r")
    if data.size < 2:
        raise SystemExit(f"Not enough tokens in {path}: {data.size}")
    max_id = int(data.max())
    if max_id >= vocab_size:
        raise SystemExit(
            f"Test token ID {max_id} exceeds checkpoint vocabulary size {vocab_size}"
        )
    return data


def evaluate_model(model, data, context, stride, batch_size, device):
    """Score each next-token target once with a sliding context window.

    The stride controls the context/throughput tradeoff. With context=256 and
    stride=32, each scored token after the initial warm-up has 225-256 prior
    tokens available. Smaller strides use more context but require more
    forward passes.
    """
    import torch.nn.functional as F

    model.eval()
    loss_sum = 0.0
    token_count = 0
    started = time.perf_counter()
    pending_x = []
    pending_y = []

    def score_batch(xs, ys):
        nonlocal loss_sum, token_count
        inputs = torch.from_numpy(np.stack(xs)).to(device=device, dtype=torch.long)
        labels = torch.from_numpy(np.stack(ys)).to(device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        # Every item in a full batch has the same context and scores its last
        # 'stride' targets. Earlier positions are context only.
        score_from = context - stride
        selected_logits = logits[:, score_from:, :].contiguous()
        selected_labels = labels[:, score_from:].contiguous()
        loss = F.cross_entropy(
            selected_logits.view(-1, selected_logits.size(-1)),
            selected_labels.view(-1),
            reduction="sum",
        )
        count = selected_labels.numel()
        loss_sum += float(loss.float())
        token_count += count

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
    mean_loss = loss_sum / token_count
    return {
        "test_loss": mean_loss,
        "test_perplexity": math.exp(min(20.0, mean_loss)),
        "scored_tokens": token_count,
        "evaluation_seconds": elapsed,
        "evaluation_tokens_per_second": token_count / max(elapsed, 1e-9),
        "context_tokens": context,
        "evaluation_stride": stride,
        "minimum_context_for_steady_state_tokens": context - stride + 1,
    }

def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.stride < 1:
        raise SystemExit("--stride must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/ROCm GPU visible to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This evaluator requires GPU BF16 support")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    test_path = args.data_dir / "test.bin"
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    test_sha = sha256_file(test_path)
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None
    if manifest:
        recorded_test_sha = manifest.get("splits", {}).get("test", {}).get("sha256")
        if recorded_test_sha and recorded_test_sha != test_sha:
            raise SystemExit("test.bin SHA256 does not match manifest.json")
    results = []
    shared_config = None
    shared_seed = None

    for name, checkpoint_path in parse_checkpoints(args.checkpoint):
        print(f"Loading {name}: {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" not in state or "model" not in state or "mode" not in state:
            raise SystemExit(f"{checkpoint_path} is not a compatible E49 local checkpoint")

        cfg = E49Config(**state["config"])
        if cfg.mode != state["mode"]:
            raise SystemExit(f"Checkpoint mode/config mismatch in {checkpoint_path}")
        if args.stride > cfg.context:
            raise SystemExit(f"--stride {args.stride} cannot exceed context {cfg.context}")
        comparable_config = cfg.to_dict().copy()
        comparable_config.pop("mode")
        if shared_config is None:
            shared_config = comparable_config
        elif comparable_config != shared_config:
            raise SystemExit("Checkpoint architectures differ beyond the dense/lowrank/latent mode")
        run_config_path = checkpoint_path.parent / "run_config.json"
        run_config = json.loads(run_config_path.read_text()) if run_config_path.is_file() else None
        if run_config:
            if run_config.get("config") != cfg.to_dict() or run_config.get("mode") != state["mode"]:
                raise SystemExit(f"run_config.json does not match checkpoint: {run_config_path}")
            if manifest_sha and run_config.get("manifest_sha256") != manifest_sha:
                raise SystemExit(f"Training data manifest mismatch for {checkpoint_path}")
            seed = run_config.get("seed")
            if shared_seed is None:
                shared_seed = seed
            elif seed != shared_seed:
                raise SystemExit("Checkpoint seeds differ; refusing an unmatched comparison")
        data = load_test_tokens(test_path, cfg.context, cfg.vocab_size)
        model = E49LM(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()

        torch.cuda.reset_peak_memory_stats(device)
        metrics = evaluate_model(model, data, cfg.context, args.stride, args.batch_size, device)
        row = {
            "name": name,
            "mode": cfg.mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config": cfg.to_dict(),
            "parameters": model.parameter_breakdown(),
            "test_file": str(test_path),
            "test_file_bytes": test_path.stat().st_size,
            "test_file_sha256": test_sha,
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
            f"{name}: test_loss={row['test_loss']:.6f} "
            f"ppl={row['test_perplexity']:.4f} "
            f"tokens={row['scored_tokens']:,} "
            f"tok/s={row['evaluation_tokens_per_second']:.1f}",
            flush=True,
        )
        del model, state, data
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "experiment": "E49-held-out-test",
        "evaluation": "test.bin; each next-token target scored once with sliding context",
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
