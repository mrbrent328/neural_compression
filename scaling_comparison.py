#!/usr/bin/env python3
"""
Scaling Comparison — Dense vs LowRank vs Latent Transformer
============================================================
Runs all three modes at multiple scales with matched budgets and
writes a combined report to runs/scaling_comparison/.

Usage:
    python scaling_comparison.py --data /path/to/corpus.txt

Scales tested (n_layers, d_model, d_ff, n_heads):
  tiny  : 2 layers, d_model=128, d_ff=512,  heads=4
  small : 4 layers, d_model=256, d_ff=1024, heads=4
  medium: 6 layers, d_model=384, d_ff=1536, heads=6

For each scale, all three modes are trained with the same:
  - corpus, tokenizer, context window
  - steps, batch size, LR schedule, seed
  - rank=48 for lowrank/latent
  - latent_dim=16 for latent

Results are written per-run to runs/scaling_comparison/<scale>/<mode>/
and summarised in runs/scaling_comparison/summary.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Scale definitions
# ---------------------------------------------------------------------------

SCALES = {
    "tiny": dict(
        layers=2, d_model=128, heads=4, d_ff=512,
        steps=500, batch_size=16, context=256, rank=32, latent_dim=8,
    ),
    "small": dict(
        layers=4, d_model=256, heads=4, d_ff=1024,
        steps=500, batch_size=16, context=256, rank=48, latent_dim=16,
    ),
    "medium": dict(
        layers=6, d_model=384, heads=6, d_ff=1536,
        steps=500, batch_size=16, context=256, rank=48, latent_dim=16,
    ),
}

MODES = ["dense", "lowrank", "latent"]
SEED  = 1337
LR    = 3e-4
EVAL_EVERY = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scaling comparison runner")
    p.add_argument("--data", type=Path, required=True, help="UTF-8 training corpus")
    p.add_argument("--scales", nargs="+", default=list(SCALES.keys()),
                   choices=list(SCALES.keys()), help="Which scales to run")
    p.add_argument("--modes", nargs="+", default=MODES,
                   choices=MODES, help="Which modes to run")
    p.add_argument("--steps", type=int, default=None,
                   help="Override steps for all scales")
    p.add_argument("--out-dir", type=Path,
                   default=Path("runs/scaling_comparison"),
                   help="Output root directory")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--force", action="store_true",
                   help="Re-run even if report.json already exists")
    return p.parse_args()


def run_one(
    *,
    scale_name: str,
    mode: str,
    scale: dict,
    data: Path,
    out_dir: Path,
    device: str,
    dtype: str,
    force: bool,
    steps_override: int | None,
) -> dict:
    run_dir = out_dir / scale_name / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    # latent_transformer_v0 writes report.json inside a sub-dir named after
    # the auto-generated run_name. We pass --run-name to control it.
    inner_run_name = f"{scale_name}_{mode}"
    report_path = run_dir / inner_run_name / "report.json"

    if report_path.exists() and not force:
        print(f"  [skip] {scale_name}/{mode} — report.json exists")
        with open(report_path) as f:
            return json.load(f)

    steps = steps_override if steps_override is not None else scale["steps"]

    cmd = [
        sys.executable, "-u",          # -u = unbuffered stdout
        "latent_transformer_v0/latent_transformer_v0.py",
        "--data", str(data),
        "--mode", mode,
        "--layers", str(scale["layers"]),
        "--d-model", str(scale["d_model"]),
        "--heads", str(scale["heads"]),
        "--d-ff", str(scale["d_ff"]),
        "--context", str(scale["context"]),
        "--rank", str(scale["rank"]),
        "--latent-dim", str(scale["latent_dim"]),
        "--steps", str(steps),
        "--batch-size", str(scale["batch_size"]),
        "--lr", str(LR),
        "--seed", str(SEED),
        "--eval-every", str(EVAL_EVERY),
        "--device", device,
        "--dtype", dtype,
        "--output-dir", str(run_dir),
        "--run-name", f"{scale_name}_{mode}",
    ]

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    log_path = run_dir / "console.log"
    t0 = time.time()
    print(f"  [run] {scale_name}/{mode} — {steps} steps …", flush=True)

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            logf.write(line)
            # Print progress lines to terminal
            if any(tok in line for tok in ("step", "loss", "ppl", "val")):
                print(f"    {line.rstrip()}", flush=True)
        proc.wait()

    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [FAIL] {scale_name}/{mode} — exit {proc.returncode} "
              f"(see {log_path})")
        return {"scale_name": scale_name, "mode": mode, "status": "FAILED",
                "elapsed_seconds": elapsed}

    if not report_path.exists():
        print(f"  [WARN] {scale_name}/{mode} — no report.json written")
        return {"scale_name": scale_name, "mode": mode, "status": "NO_REPORT",
                "elapsed_seconds": elapsed}

    with open(report_path) as f:
        report = json.load(f)
    report["scale_name"] = scale_name
    report["run_elapsed_seconds"] = elapsed
    report["status"] = "OK"
    hist = report.get("history", [])
    best_val = min((h["val_loss"] for h in hist), default=float("nan"))
    report["best_val_loss"] = best_val
    total_p = report.get("parameter_counts", {}).get("total", 0)
    ckpt_mb = report.get("checkpoint_file_bytes", 0) / 1e6
    print(f"  [done] {scale_name}/{mode} — "
          f"best_val={best_val:.4f}  "
          f"params={total_p:,}  "
          f"ckpt={ckpt_mb:.1f}MB  "
          f"elapsed={elapsed:.0f}s")
    return report


def summarize(reports: list[dict], out_dir: Path) -> None:
    """Write summary.json and summary.md."""
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(reports, f, indent=2)

    md_lines = [
        "# Scaling Comparison — Summary",
        "",
        "Three transformer modes compared at three scales.",
        "Same corpus, tokenizer, steps, batch size, LR schedule, and seed for every run.",
        "",
        "## Results",
        "",
        "| Scale | Mode | Total params | Ckpt MB | Best val loss | Final val loss | Tokens/s |",
        "|-------|------|-------------|---------|--------------|----------------|---------|",
    ]

    for r in reports:
        if r.get("status") != "OK":
            md_lines.append(
                f"| {r.get('scale_name','?')} | {r.get('mode','?')} "
                f"| — | — | — | — | — | {r.get('status','?')} |"
            )
            continue
        pc = r.get("parameter_counts", {})
        total_p = pc.get("total", 0)
        ckpt_mb = r.get("checkpoint_file_bytes", 0) / 1e6
        hist = r.get("history", [])
        best_val = r.get("best_val_loss",
                         min((h["val_loss"] for h in hist), default=float("nan")))
        final_val = hist[-1]["val_loss"] if hist else float("nan")
        # last recorded throughput
        tps = hist[-1].get("tokens_per_second_overall", 0) if hist else 0
        md_lines.append(
            f"| {r.get('scale_name','?')} | {r.get('mode','?')} "
            f"| {total_p:,} | {ckpt_mb:.1f} | {best_val:.4f} "
            f"| {final_val:.4f} | {tps:,.0f} |"
        )

    md_lines += [
        "",
        "## Parameter breakdown (latent mode)",
        "",
        "| Scale | Constructor | Latents (z) | Attention | Other | Total |",
        "|-------|------------|-------------|-----------|-------|-------|",
    ]
    for r in reports:
        if r.get("mode") != "latent" or r.get("status") != "OK":
            continue
        pc = r.get("parameter_counts", {})
        md_lines.append(
            f"| {r.get('scale_name','?')} "
            f"| {pc.get('constructor',0):,} "
            f"| {pc.get('latents',0):,} "
            f"| {pc.get('attention',0):,} "
            f"| {pc.get('other',0):,} "
            f"| {pc.get('total',0):,} |"
        )

    md_lines += [
        "",
        "## Notes",
        "",
        "* `rank` and `latent_dim` are identical across lowrank and latent at each scale.",
        "* Dense has no rank/latent parameters — its MLP size is `d_model × d_ff × 2 + d_ff × d_model`.",
        "* Constructor parameter count is the same regardless of model depth or width.",
        "* Checkpoint bytes include optimizer state only for the saved model weights.",
        "",
    ]

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"\nSummary written to {summary_path} and {md_path}")


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        sys.exit(f"ERROR: corpus file not found: {args.data}")

    corpus_bytes = args.data.stat().st_size
    print(f"Corpus: {args.data} ({corpus_bytes:,} bytes)")
    print(f"Scales: {args.scales}")
    print(f"Modes:  {args.modes}")
    print(f"Output: {args.out_dir}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []
    for scale_name in args.scales:
        scale = SCALES[scale_name]
        print(f"\n=== Scale: {scale_name} "
              f"(L={scale['layers']}, D={scale['d_model']}, "
              f"d_ff={scale['d_ff']}) ===")
        for mode in args.modes:
            report = run_one(
                scale_name=scale_name,
                mode=mode,
                scale=scale,
                data=args.data,
                out_dir=args.out_dir,
                device=args.device,
                dtype=args.dtype,
                force=args.force,
                steps_override=args.steps,
            )
            all_reports.append(report)
            # Write incremental summary after each run
            summarize(all_reports, args.out_dir)

    print("\n" + "=" * 60)
    print("ALL RUNS COMPLETE")
    summarize(all_reports, args.out_dir)

    # Print a quick terminal table
    print()
    print(f"{'Scale':<8} {'Mode':<10} {'Params':>12} {'CkptMB':>8} "
          f"{'BestVal':>9} {'FinalVal':>9}")
    print("-" * 60)
    for r in all_reports:
        if r.get("status") != "OK":
            print(f"{r.get('scale_name','?'):<8} {r.get('mode','?'):<10} "
                  f"{'FAILED':>12}")
            continue
        pc = r.get("parameter_counts", {})
        total_p = pc.get("total", 0)
        ckpt_mb = r.get("checkpoint_file_bytes", 0) / 1e6
        hist = r.get("history", [])
        best_val = r.get("best_val_loss",
                         min((h["val_loss"] for h in hist), default=float("nan")))
        final_val = hist[-1]["val_loss"] if hist else float("nan")
        print(f"{r.get('scale_name','?'):<8} {r.get('mode','?'):<10} "
              f"{total_p:>12,} {ckpt_mb:>8.1f} "
              f"{best_val:>9.4f} {final_val:>9.4f}")


if __name__ == "__main__":
    main()
