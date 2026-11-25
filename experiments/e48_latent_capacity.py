#!/usr/bin/env python3
"""E48: latent capacity sweep at the E47 tiny-model scale.

Varies only latent_dim (default: 8, 16, 32, 64). The model scale, rank,
constructor width, corpus split, optimizer settings, and seed are held fixed.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from e47_scale import collect_sources, split_corpus

DEFAULT_DIMS = (8, 16, 32, 64)


def parse_dims(value: str) -> list[int]:
    try:
        dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("latent dimensions must be comma-separated integers") from exc
    if not dims or any(dim < 1 for dim in dims) or len(set(dims)) != len(dims):
        raise argparse.ArgumentTypeError("provide unique positive latent dimensions")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description="E48 latent-dimension capacity sweep")
    parser.add_argument("--data", type=Path, required=True,
                        help="Text file or directory of source documents")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/e48_latent_capacity"))
    parser.add_argument("--latent-dims", type=parse_dims, default=list(DEFAULT_DIMS),
                        help="Comma-separated latent dimensions (default: 8,16,32,64)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--test-batches", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--constructor-hidden", type=int, default=128)
    parser.add_argument("--constructor-type-dim", type=int, default=16)
    parser.add_argument("--constructor-fourier-bands", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Skip dimensions that already have e48_result.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare the corpus split and print the plan without training")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    model_script = root / "latent_transformer_v0" / "latent_transformer_v0.py"
    if not model_script.is_file():
        raise SystemExit(f"Cannot find model script: {model_script}. Set --repo-root to the repository checkout.")
    if args.steps < 1 or args.context < 2 or args.rank < 1:
        raise SystemExit("--steps and --rank must be positive; --context must be >= 2")

    output = args.output_dir.resolve()
    split_dir = output / "corpus_split"
    split_meta = split_corpus(collect_sources(args.data.resolve()), split_dir, args.context)
    print("E48 CORPUS:", json.dumps(split_meta, indent=2), flush=True)

    plan = args.latent_dims
    print(
        f"\nE48 PLAN: {len(plan)} runs | latent_dims={plan} | seed={args.seed} | "
        f"steps={args.steps} | rank={args.rank} | constructor_hidden={args.constructor_hidden}",
        flush=True,
    )
    if args.dry_run:
        for dim in plan:
            print(f"  tiny latent_dim={dim} rank={args.rank} seed={args.seed}")
        print("\nDry run only; no model training started.")
        return

    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, dim in enumerate(plan, 1):
        run_name = f"tiny_latent_z{dim}_seed{args.seed}"
        run_dir = output / run_name
        artifact_dir = run_dir / run_name
        result_path = run_dir / "e48_result.json"

        if args.resume and result_path.exists():
            print(f"[{index}/{len(plan)}] SKIP existing {run_name}", flush=True)
            results.append(json.loads(result_path.read_text()))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(model_script),
            "--data", str(split_dir / "train.bin"),
            "--val-data", str(split_dir / "val.bin"),
            "--mode", "latent",
            "--device", args.device, "--dtype", args.dtype, "--seed", str(args.seed),
            "--layers", "2", "--d-model", "128", "--heads", "4", "--d-ff", "512",
            "--rank", str(args.rank), "--latent-dim", str(dim),
            "--constructor-hidden", str(args.constructor_hidden),
            "--constructor-type-dim", str(args.constructor_type_dim),
            "--constructor-fourier-bands", str(args.constructor_fourier_bands),
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
            assert process.stdout is not None
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
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 3)
        test_loss = lm.evaluate(
            model, test_data, args.batch_size, args.context,
            args.test_batches, device, generator,
        )
        report = json.loads((artifact_dir / "report.json").read_text())
        history = report.get("history", [])
        best = min(history, key=lambda row: row["val_loss"]) if history else {}
        counts = report.get("parameter_counts", {})
        result = {
            "experiment": "E48_latent_capacity",
            "run_name": run_name,
            "scale": "tiny",
            "mode": "latent",
            "latent_dim": dim,
            "rank": args.rank,
            "constructor_hidden": args.constructor_hidden,
            "constructor_type_dim": args.constructor_type_dim,
            "constructor_fourier_bands": args.constructor_fourier_bands,
            "seed": args.seed,
            "steps": args.steps,
            "context": args.context,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "train_bytes": split_meta["split_bytes"]["train"],
            "val_bytes": split_meta["split_bytes"]["val"],
            "test_bytes": split_meta["split_bytes"]["test"],
            "split_sha256": split_meta["sha256"],
            "test_loss": test_loss,
            "test_perplexity": math.exp(min(20, test_loss)),
            "best_val_loss": best.get("val_loss"),
            "best_val_step": best.get("step"),
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "parameter_counts": counts,
            "checkpoint_file_bytes": report.get("checkpoint_file_bytes"),
            "state_dict_bytes": report.get("state_dict_bytes_before_save"),
            "elapsed_wall_seconds_including_test": time.time() - started,
            "device": str(device),
            "dtype": args.dtype,
            "training_report": str(artifact_dir / "report.json"),
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(
            f"[{index}/{len(plan)}] DONE {run_name}: "
            f"best val={result['best_val_loss']:.4f}, test={test_loss:.4f}, "
            f"params={counts.get('total', 0):,}",
            flush=True,
        )
        summary = {
            "experiment": "E48_latent_capacity",
            "split": split_meta,
            "settings": {
                **vars(args),
                "data": str(args.data),
                "repo_root": str(root),
                "output_dir": str(output),
            },
            "runs": results,
        }
        summary["settings"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in summary["settings"].items()
        }
        (output / "e48_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nALL RUNS COMPLETE. Summary: {output / 'e48_summary.json'}")


if __name__ == "__main__":
    main()
