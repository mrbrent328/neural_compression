#!/usr/bin/env python3
"""Prepare leakage-aware paragraph splits and rerun the medium scaling comparison."""
from __future__ import annotations
import argparse, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "latent_transformer_v0" / "latent_transformer_v0.py"
DEFAULT_SOURCE = ROOT / "corpus.txt"

def normalize(block: str) -> str:
    return " ".join(block.split()).casefold()

def prepare(source: Path, out_dir: Path, seed: int, val_fraction: float) -> dict:
    raw = source.read_text(encoding="utf-8")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    unique, duplicates = {}, 0
    for block in blocks:
        key = normalize(block)
        if not key:
            continue
        if key in unique:
            duplicates += 1
        else:
            unique[key] = block
    items = list(unique.items())
    random.Random(seed).shuffle(items)
    target = int(sum(len(v.encode("utf-8")) for _, v in items) * val_fraction)
    val_items, train_items, val_bytes = [], [], 0
    for item in items:
        size = len(item[1].encode("utf-8"))
        if val_bytes < target:
            val_items.append(item)
            val_bytes += size
        else:
            train_items.append(item)
    if not train_items or not val_items:
        raise RuntimeError("Could not create non-empty train/validation paragraph sets.")
    overlap = {k for k, _ in train_items} & {k for k, _ in val_items}
    if overlap:
        raise RuntimeError("Exact normalized paragraph overlap remains after deduplication.")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.txt", out_dir / "validation.txt"
    train_path.write_text("\n\n".join(v for _, v in train_items) + "\n", encoding="utf-8")
    val_path.write_text("\n\n".join(v for _, v in val_items) + "\n", encoding="utf-8")
    train_size, val_size = train_path.stat().st_size, val_path.stat().st_size
    manifest = {
        "experiment": "SCALING_REVALIDATION_V1",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_bytes": source.stat().st_size,
        "source_paragraphs": len(blocks),
        "unique_normalized_paragraphs": len(items),
        "exact_duplicate_paragraphs_removed": duplicates,
        "split_method": "deterministic shuffled paragraph-level holdout after exact normalized deduplication",
        "source_document_ids_available": False,
        "warning": "Paragraph-level separation only, not source-document-level separation. Related paragraphs from one original document may cross the split.",
        "seed": seed,
        "requested_validation_fraction": val_fraction,
        "train_paragraphs": len(train_items),
        "validation_paragraphs": len(val_items),
        "train_bytes": train_size,
        "validation_bytes": val_size,
        "actual_validation_fraction_bytes": val_size / (train_size + val_size),
        "exact_normalized_paragraph_overlap": len(overlap),
        "train_file": str(train_path),
        "validation_file": str(val_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

def run(args: argparse.Namespace) -> int:
    if not MODEL.exists():
        raise SystemExit(f"Missing model script: {MODEL}")
    run_root = args.output_dir
    split_dir = run_root / "corpus_split"
    manifest = prepare(args.data, split_dir, args.split_seed, args.validation_fraction)
    print("CORPUS PREPARATION")
    print(json.dumps(manifest, indent=2))
    if args.prepare_only:
        print(f"Prepared files in: {split_dir}")
        return 0
    reports, env = [], os.environ.copy()
    env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    for seed in args.seeds:
        for mode in args.modes:
            run_name = f"medium_{mode}_seed{seed}"
            mode_dir = run_root / f"seed_{seed}" / mode
            report_path = mode_dir / run_name / "report.json"
            if report_path.exists() and not args.force:
                raise SystemExit(f"Refusing to overwrite {report_path}; use --force or a new output directory.")
            mode_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(MODEL),
                "--data", str((split_dir / "train.txt").resolve()),
                "--val-data", str((split_dir / "validation.txt").resolve()),
                "--mode", mode, "--layers", "6", "--d-model", "384",
                "--heads", "6", "--d-ff", "1536", "--context", "256",
                "--rank", "48", "--latent-dim", "16", "--steps", str(args.steps),
                "--batch-size", "16", "--lr", "3e-4", "--warmup-steps", "100",
                "--eval-every", "100", "--eval-batches", "20", "--seed", str(seed),
                "--device", args.device, "--dtype", args.dtype,
                "--output-dir", str(mode_dir.resolve()), "--run-name", run_name,
            ]
            log_path = mode_dir / "console.log"
            print(f"\nSTART {run_name}\n" + " ".join(cmd), flush=True)
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.write(line); log.flush()
                    if any(k in line for k in ("step ", "EVAL val", "parameters:", "device/dtype:", "ERROR")):
                        print(line.rstrip(), flush=True)
                code = proc.wait()
            elapsed = round(time.time() - started, 2)
            entry = {"seed": seed, "mode": mode, "exit_code": code,
                     "elapsed_seconds": elapsed, "log": str(log_path)}
            if code == 0 and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                hist = report.get("history", [])
                entry.update({
                    "parameters": report.get("parameter_counts", {}).get("total"),
                    "checkpoint_bytes": report.get("checkpoint_file_bytes"),
                    "best_validation_loss": min((x["val_loss"] for x in hist), default=None),
                    "best_validation_step": min(hist, key=lambda x: x["val_loss"])["step"] if hist else None,
                    "final_validation_loss": hist[-1]["val_loss"] if hist else None,
                    "final_train_loss": hist[-1]["train_loss"] if hist else None,
                    "tokens_per_second": hist[-1].get("tokens_per_second_overall") if hist else None,
                    "report": str(report_path),
                })
                print(f"COMPLETE {run_name}: best_val={entry['best_validation_loss']}")
            else:
                print(f"FAILED {run_name}: exit={code}; inspect {log_path}")
            reports.append(entry)
            (run_root / "progress.json").write_text(json.dumps({
                "experiment": "SCALING_REVALIDATION_V1",
                "corpus_manifest": str(split_dir / "manifest.json"), "runs": reports,
            }, indent=2) + "\n", encoding="utf-8")
            if code != 0:
                print("Stopping after first failed run; inspect the log before continuing.")
                return code
    print("\nALL REQUESTED RUNS FINISHED.")
    print(f"Progress report: {run_root / 'progress.json'}")
    return 0

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "scaling_revalidation_v1")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1337])
    p.add_argument("--modes", nargs="+", choices=["dense", "lowrank", "latent"],
                   default=["dense", "lowrank", "latent"])
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--split-seed", type=int, default=20261124)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not 0.05 <= args.validation_fraction <= 0.30:
        raise SystemExit("--validation-fraction must be between 0.05 and 0.30")
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    run(args)

if __name__ == "__main__":
    main()
