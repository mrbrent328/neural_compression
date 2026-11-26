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


def evaluate_model(model, data, context, batch_size, device):
    """Score every next-token target in test.bin exactly once.

    Windows overlap by one token: the final token of one input window is the
    first target of the next window. This preserves contiguous context while
    avoiding dropped targets or duplicated target tokens.
    """
    model.eval()
    loss_sum = 0.0
    token_count = 0
    started = time.perf_counter()
    full_windows = (len(data) - 1) // context

    with torch.inference_mode():
        for batch_start in range(0, full_windows, batch_size):
            starts = range(batch_start, min(batch_start + batch_size, full_windows))
            windows = np.stack([
                np.asarray(data[i * context:i * context + context + 1], dtype=np.int64)
                for i in starts
            ])
            tokens = torch.from_numpy(windows).to(device=device, dtype=torch.long)
            inputs, labels = tokens[:, :-1], tokens[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(inputs, labels)
            count = labels.numel()
            loss_sum += float(loss.float()) * count
            token_count += count

        remainder = (len(data) - 1) - full_windows * context
        if remainder:
            start = full_windows * context
            window = np.asarray(data[start:len(data)], dtype=np.int64)[None, :]
            tokens = torch.from_numpy(window).to(device=device, dtype=torch.long)
            inputs, labels = tokens[:, :-1], tokens[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(inputs, labels)
            count = labels.numel()
            loss_sum += float(loss.float()) * count
            token_count += count

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    mean_loss = loss_sum / token_count
    return {
        "test_loss": mean_loss,
        "test_perplexity": math.exp(min(20.0, mean_loss)),
        "scored_tokens": token_count,
        "evaluation_seconds": elapsed,
        "evaluation_tokens_per_second": token_count / max(elapsed, 1e-9),
    }


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
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
    results = []

    for name, checkpoint_path in parse_checkpoints(args.checkpoint):
        print(f"Loading {name}: {checkpoint_path}", flush=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" not in state or "model" not in state or "mode" not in state:
            raise SystemExit(f"{checkpoint_path} is not a compatible E49 local checkpoint")

        cfg = E49Config(**state["config"])
        if cfg.mode != state["mode"]:
            raise SystemExit(f"Checkpoint mode/config mismatch in {checkpoint_path}")
        data = load_test_tokens(test_path, cfg.context, cfg.vocab_size)
        model = E49LM(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()

        torch.cuda.reset_peak_memory_stats(device)
        metrics = evaluate_model(model, data, cfg.context, args.batch_size, device)
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
            "manifest_sha256": (
                hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if manifest_path.is_file() else None
            ),
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
        "evaluation": "test.bin; each next-token target scored once",
        "batch_size": args.batch_size,
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
