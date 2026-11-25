#!/usr/bin/env python3
"""E47 reproducible scale study: dense vs low-rank vs latent MLP parameterizations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

SCALES = {
    "tiny": dict(layers=2, d_model=128, heads=4, d_ff=512, rank=32, latent_dim=8),
    "small": dict(layers=4, d_model=256, heads=4, d_ff=1024, rank=48, latent_dim=16),
    "medium": dict(layers=6, d_model=384, heads=6, d_ff=1536, rank=48, latent_dim=16),
    "large": dict(layers=8, d_model=512, heads=8, d_ff=2048, rank=64, latent_dim=24),
}
EXTS = {".txt", ".md", ".py", ".json", ".csv", ".html", ".xml", ".rst", ".log"}


def collect_sources(path: Path):
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in EXTS
    )
    rows, seen = [], set()
    for file in files:
        try:
            raw = file.read_bytes()
        except OSError:
            continue
        if len(raw) < 256:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in seen:
            seen.add(digest)
            rows.append((str(file), raw, digest))
    if not rows:
        raise SystemExit(f"No usable text sources found under {path}")
    return rows


def split_corpus(rows, outdir: Path, context: int):
    outdir.mkdir(parents=True, exist_ok=True)
    data = None
    if len(rows) >= 3:
        buckets = {"train": [], "val": [], "test": []}
        totals = {key: 0 for key in buckets}
        targets = {"train": .8, "val": .1, "test": .1}
        total = sum(len(row[1]) for row in rows)
        for row in sorted(rows, key=lambda item: len(item[1]), reverse=True):
            key = min(buckets, key=lambda k: totals[k] / max(targets[k] * total, 1))
            buckets[key].append(row)
            totals[key] += len(row[1])
        if all(buckets.values()):
            data = {k: b"\n\n".join(row[1] for row in v) for k, v in buckets.items()}
            method = "source-file-isolated"
            manifest = {k: [row[0] for row in v] for k, v in buckets.items()}
    if data is None:
        raw = b"\n\n".join(row[1] for row in rows)
        n, gap = len(raw), max(context + 1, 1024)
        a, b = int(n * .8), int(n * .9)
        data = {
            "train": raw[:max(0, a-gap)],
            "val": raw[min(n, a+gap):max(a+gap, b-gap)],
            "test": raw[min(n, b+gap):],
        }
        method = "contiguous-with-guard-gaps"
        manifest = {"source_files": [row[0] for row in rows], "guard_gap_bytes": gap}
    for key, raw_part in data.items():
        if len(raw_part) < 4096:
            raise SystemExit(
                f"{key} split is only {len(raw_part):,} bytes; each split needs at least 4 KiB. "
                "Supply a larger and more diverse corpus."
            )
        (outdir / f"{key}.bin").write_bytes(raw_part)
    meta = {
        "method": method,
        "manifest": manifest,
        "source_file_count_after_exact_dedup": len(rows),
        "source_bytes": sum(len(row[1]) for row in rows),
        "split_bytes": {k: len(v) for k, v in data.items()},
        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in data.items()},
    }
    (outdir / "split_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def csv_choices(value, allowed, label):
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if not values or invalid:
        raise SystemExit(f"Invalid {label}: {invalid or value}; choose from {', '.join(allowed)}")
    return values


def main():
    parser = argparse.ArgumentParser(description="E47 multi-scale latent-model benchmark")
    parser.add_argument("--data", type=Path, required=True, help="Text file or directory of source documents")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/e47_scale"))
    parser.add_argument("--scales", default="tiny,small,medium")
    parser.add_argument("--modes", default="dense,lowrank,latent")
    parser.add_argument("--seeds", default="1337", help="Comma-separated seeds; use 3+ for confirmatory results")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--test-batches", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--resume", action="store_true", help="Skip runs with an existing e47_result.json")
    parser.add_argument("--dry-run", action="store_true", help="Prepare splits and print plan without training")
    args = parser.parse_args()

    scales = csv_choices(args.scales, SCALES, "scale")
    modes = csv_choices(args.modes, ["dense", "lowrank", "latent"], "mode")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds or args.steps < 1 or args.context < 2:
        raise SystemExit("--seeds must be nonempty; --steps >= 1 and --context >= 2")
    root = args.repo_root.resolve()
    model_script = root / "latent_transformer_v0" / "latent_transformer_v0.py"
    if not model_script.is_file():
        raise SystemExit(f"Cannot find model script: {model_script}. Set --repo-root to the repository checkout.")
    output = args.output_dir.resolve()
    split_dir = output / "corpus_split"
    split_meta = split_corpus(collect_sources(args.data.resolve()), split_dir, args.context)
    print("E47 CORPUS:", json.dumps(split_meta, indent=2), flush=True)

    plan = [(scale, mode, seed) for scale in scales for mode in modes for seed in seeds]
    print(f"\nE47 PLAN: {len(plan)} runs | scales={scales} | modes={modes} | seeds={seeds} | steps={args.steps}", flush=True)
    if args.dry_run:
        for scale, mode, seed in plan:
            print(f"  {scale:6s} {mode:7s} seed={seed}")
        print("\nDry run only; no model training started.")
        return

    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (scale, mode, seed) in enumerate(plan, 1):
        cfg = SCALES[scale]
        run_name = f"{scale}_{mode}_seed{seed}"
        run_dir = output / run_name
        artifact_dir = run_dir / run_name
        result_path = run_dir / "e47_result.json"
        if args.resume and result_path.exists():
            print(f"[{index}/{len(plan)}] SKIP existing {run_name}", flush=True)
            results.append(json.loads(result_path.read_text()))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(model_script), "--data", str(split_dir / "train.bin"),
            "--val-data", str(split_dir / "val.bin"), "--mode", mode,
            "--device", args.device, "--dtype", args.dtype, "--seed", str(seed),
            "--layers", str(cfg["layers"]), "--d-model", str(cfg["d_model"]),
            "--heads", str(cfg["heads"]), "--d-ff", str(cfg["d_ff"]),
            "--rank", str(cfg["rank"]), "--latent-dim", str(cfg["latent_dim"]),
            "--context", str(args.context), "--steps", str(args.steps),
            "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum),
            "--eval-every", str(args.eval_every), "--eval-batches", str(args.eval_batches),
            "--output-dir", str(run_dir), "--run-name", run_name, "--log-every", "50",
        ]
        print(f"\n[{index}/{len(plan)}] START {run_name}\nCOMMAND: {' '.join(command)}", flush=True)
        started = time.time()
        log_path = run_dir / "training.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise SystemExit(f"{run_name} failed (exit {return_code}); see {log_path}")

        import torch
        sys.path.insert(0, str(model_script.parent))
        import latent_transformer_v0 as lm
        checkpoint = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=False)
        model = lm.TinyLM(lm.ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        device = lm.choose_device(args.device)
        model.to(device=device, dtype=lm.dtype_from_name(args.dtype))
        test_data = lm.load_bytes(split_dir / "test.bin", False)
        generator = torch.Generator(device="cpu").manual_seed(seed + 3)
        test_loss = lm.evaluate(model, test_data, args.batch_size, args.context,
                                args.test_batches, device, generator)
        report = json.loads((artifact_dir / "report.json").read_text())
        history = report.get("history", [])
        best = min(history, key=lambda row: row["val_loss"]) if history else {}
        counts = report.get("parameter_counts", {})
        result = {
            "experiment": "E47_scale", "run_name": run_name, "scale": scale,
            "mode": mode, "seed": seed, "config": cfg, "steps": args.steps,
            "context": args.context, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "train_bytes": split_meta["split_bytes"]["train"],
            "val_bytes": split_meta["split_bytes"]["val"],
            "test_bytes": split_meta["split_bytes"]["test"],
            "test_loss": test_loss, "test_perplexity": math.exp(min(20, test_loss)),
            "best_val_loss": best.get("val_loss"), "best_val_step": best.get("step"),
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "parameter_counts": counts,
            "checkpoint_file_bytes": report.get("checkpoint_file_bytes"),
            "state_dict_bytes": report.get("state_dict_bytes_before_save"),
            "elapsed_wall_seconds_including_test": time.time() - started,
            "device": str(device), "dtype": args.dtype,
            "training_report": str(artifact_dir / "report.json"),
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(
            f"[{index}/{len(plan)}] DONE {run_name}: "
            f"best val={result['best_val_loss']:.4f}, test={test_loss:.4f}, "
            f"params={counts.get('total', 0):,}", flush=True
        )
        summary = {
            "experiment": "E47_scale", "split": split_meta,
            "settings": {**vars(args), "data": str(args.data), "repo_root": str(root),
                         "output_dir": str(output)},
            "runs": results,
        }
        summary["settings"] = {k: str(v) if isinstance(v, Path) else v
                               for k, v in summary["settings"].items()}
        (output / "e47_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nALL RUNS COMPLETE. Summary: {output / 'e47_summary.json'}")


if __name__ == "__main__":
    main()
