#!/usr/bin/env python3
"""
Latent Transformer v0 configuration sweep
==========================================
Runs controlled grids of dense, low-rank, and latent configurations and keeps
all results under ONE sweep directory with continuously refreshed summaries.

Designed for long unattended runs:
  * unique run names (no accidental overwrites)
  * resumable: completed reports are skipped unless --force is used
  * console output is tee'd to each run's console.log
  * summary.csv, summary.json, leaderboard.md update after every run
  * failures are recorded and the sweep continues

Examples:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
      --data Gorgias.txt --profile core --name gorgias_core

  HSA_OVERRIDE_GFX_VERSION=10.3.0 python sweep_latent_configs.py \
      --data Gorgias.txt --profile broad --name gorgias_broad

Use --dry-run first if you want to see the exact grid without training.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "latent_transformer_v0.py"


@dataclass(frozen=True)
class RunSpec:
    mode: str
    heads: int
    rank: int = 64
    latent_dim: int = 32
    constructor_hidden: int = 128
    fourier_bands: int = 8
    seed: int = 1337

    def name(self) -> str:
        if self.mode == "dense":
            return f"dense_H{self.heads}_S{self.seed}"
        if self.mode == "lowrank":
            return f"lowrank_H{self.heads}_R{self.rank}_S{self.seed}"
        return (
            f"latent_H{self.heads}_R{self.rank}_Z{self.latent_dim}"
            f"_CH{self.constructor_hidden}_FB{self.fourier_bands}_S{self.seed}"
        )


def parse_int_list(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep Latent Transformer v0 configurations")
    p.add_argument("--data", type=Path, required=True, help="UTF-8 corpus passed to latent_transformer_v0.py")
    p.add_argument("--name", default="config_sweep", help="Sweep folder name")
    p.add_argument("--output-root", type=Path, default=Path("sweeps"))
    p.add_argument("--profile", choices=("core", "broad", "constructor", "custom"), default="core")

    # Custom/profile overrides. For core/broad these defaults are replaced by the profile.
    p.add_argument("--heads", type=parse_int_list, default=None, help="comma list, e.g. 4,6,8")
    p.add_argument("--ranks", type=parse_int_list, default=None, help="comma list, e.g. 32,48,64")
    p.add_argument("--latent-dims", type=parse_int_list, default=None, help="comma list, e.g. 1,2,4,8,16,32")
    p.add_argument("--constructor-hiddens", type=parse_int_list, default=None, help="comma list")
    p.add_argument("--fourier-bands", type=parse_int_list, default=None, help="comma list")
    p.add_argument("--seeds", type=parse_int_list, default=[1337], help="comma list")

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--device", default="auto")

    p.add_argument("--include-dense", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-lowrank", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reports-only", action="store_true",
                   help="Delete model.pt after each successful run; keeps reports/samples/logs")
    p.add_argument("--force", action="store_true", help="Rerun configurations even when report.json exists")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def profile_values(a: argparse.Namespace):
    if a.profile == "core":
        heads = [4, 6, 8]
        ranks = [32, 48, 64]
        dims = [1, 2, 4, 8, 16, 32]
        hiddens = [128]
        bands = [8]
    elif a.profile == "broad":
        # Wide enough to map attention geometry, rank and latent capacity in one overnight-ish sweep.
        heads = [3, 4, 6, 8, 12]
        ranks = [16, 32, 48, 64]
        dims = [1, 2, 4, 8, 16, 32]
        hiddens = [128]
        bands = [8]
    elif a.profile == "constructor":
        # Focused sweep of the shared constructor while keeping the current best-known neighborhood.
        heads = [6]
        ranks = [64]
        dims = [1, 2, 4, 8, 16, 32]
        hiddens = [64, 128, 256]
        bands = [4, 8, 12]
    else:
        heads = [6]
        ranks = [64]
        dims = [32]
        hiddens = [128]
        bands = [8]

    # Any explicit list overrides its profile dimension.
    heads = a.heads or heads
    ranks = a.ranks or ranks
    dims = a.latent_dims or dims
    hiddens = a.constructor_hiddens or hiddens
    bands = a.fourier_bands or bands
    return heads, ranks, dims, hiddens, bands


def build_specs(a: argparse.Namespace) -> List[RunSpec]:
    heads, ranks, dims, hiddens, bands = profile_values(a)
    specs: List[RunSpec] = []

    # Dense does not depend on rank/latent/constructor settings.
    if a.include_dense:
        for h, seed in itertools.product(heads, a.seeds):
            specs.append(RunSpec(mode="dense", heads=h, seed=seed))

    # Low-rank is the critical control for whether latent construction adds value beyond factorization.
    if a.include_lowrank:
        for h, r, seed in itertools.product(heads, ranks, a.seeds):
            specs.append(RunSpec(mode="lowrank", heads=h, rank=r, seed=seed))

    for h, r, z, ch, fb, seed in itertools.product(
        heads, ranks, dims, hiddens, bands, a.seeds
    ):
        specs.append(RunSpec(
            mode="latent", heads=h, rank=r, latent_dim=z,
            constructor_hidden=ch, fourier_bands=fb, seed=seed,
        ))

    # Stable unique order, just in case custom lists contain duplicates.
    seen = set()
    unique = []
    for s in specs:
        if s.name() not in seen:
            seen.add(s.name())
            unique.append(s)
    return unique


def command_for(a: argparse.Namespace, spec: RunSpec, runs_dir: Path) -> List[str]:
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--data", str(a.data.resolve()),
        "--mode", spec.mode,
        "--heads", str(spec.heads),
        "--rank", str(spec.rank),
        "--latent-dim", str(spec.latent_dim),
        "--constructor-hidden", str(spec.constructor_hidden),
        "--constructor-fourier-bands", str(spec.fourier_bands),
        "--steps", str(a.steps),
        "--warmup-steps", str(a.warmup_steps),
        "--batch-size", str(a.batch_size),
        "--context", str(a.context),
        "--d-model", str(a.d_model),
        "--layers", str(a.layers),
        "--d-ff", str(a.d_ff),
        "--lr", str(a.lr),
        "--dtype", a.dtype,
        "--device", a.device,
        "--seed", str(spec.seed),
        "--output-dir", str(runs_dir),
        "--run-name", spec.name(),
    ]
    return cmd


def read_report(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def extract_row(spec: RunSpec, report: Dict[str, Any], status: str = "ok") -> Dict[str, Any]:
    history = report.get("history") or []
    final = history[-1] if history else {}
    if history:
        best = min(history, key=lambda x: x.get("val_loss", float("inf")))
    else:
        best = {}
    counts = report.get("parameter_counts") or {}
    cfg = report.get("model_config") or {}
    return {
        "status": status,
        "run": spec.name(),
        "mode": spec.mode,
        "heads": cfg.get("n_heads", spec.heads),
        "rank": cfg.get("rank", spec.rank),
        "latent_dim": cfg.get("latent_dim", spec.latent_dim),
        "constructor_hidden": cfg.get("constructor_hidden", spec.constructor_hidden),
        "fourier_bands": cfg.get("constructor_fourier_bands", spec.fourier_bands),
        "seed": report.get("seed", spec.seed),
        "total_params": counts.get("total"),
        "attention_params": counts.get("attention"),
        "mlp_params": counts.get("mlp"),
        "constructor_params": counts.get("constructor"),
        "latent_params": counts.get("latents"),
        "checkpoint_bytes": report.get("checkpoint_file_bytes"),
        "best_val_loss": best.get("val_loss"),
        "best_step": best.get("step"),
        "final_val_loss": final.get("val_loss"),
        "final_train_loss": final.get("train_loss"),
        "final_perplexity": final.get("perplexity"),
        "tokens_per_second": final.get("tokens_per_second_overall"),
        "peak_device_memory_bytes": final.get("peak_device_memory_bytes"),
        "sample": (report.get("sample") or "").replace("\r", " ").replace("\n", " ")[:180],
    }


def write_summaries(sweep_dir: Path, rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("best_val_loss") is None,
            r.get("best_val_loss") if r.get("best_val_loss") is not None else float("inf"),
            r.get("run", ""),
        ),
    )
    (sweep_dir / "summary.json").write_text(json.dumps(sorted_rows, indent=2) + "\n")

    fieldnames = [
        "status", "run", "mode", "heads", "rank", "latent_dim",
        "constructor_hidden", "fourier_bands", "seed", "total_params",
        "attention_params", "mlp_params", "constructor_params", "latent_params",
        "checkpoint_bytes", "best_val_loss", "best_step", "final_val_loss",
        "final_train_loss", "final_perplexity", "tokens_per_second",
        "peak_device_memory_bytes", "sample",
    ]
    with (sweep_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted_rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    ok = [r for r in sorted_rows if r.get("best_val_loss") is not None]
    md = [
        "# Latent Transformer Sweep Leaderboard",
        "",
        f"Completed results: **{len(ok)} / {manifest['run_count']}**",
        "",
        "Sorted by best observed validation loss. Final loss is also shown so overfitting is visible.",
        "",
        "| # | run | mode | H | R | Z | CH | FB | best val | step | final val | tok/s | params |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ok[:40], 1):
        md.append(
            f"| {i} | `{r['run']}` | {r['mode']} | {r['heads']} | {r['rank']} | "
            f"{r['latent_dim']} | {r['constructor_hidden']} | {r['fourier_bands']} | "
            f"{r['best_val_loss']:.4f} | {r['best_step']} | {r['final_val_loss']:.4f} | "
            f"{(r['tokens_per_second'] or 0):.0f} | {r['total_params']:,} |"
        )

    # Latent-only view makes the architecture search easier to scan.
    latent = [r for r in ok if r.get("mode") == "latent"]
    if latent:
        md += [
            "",
            "## Best latent configurations",
            "",
            "| # | H | R | Z | CH | FB | best val | final val | latent params | constructor params | tok/s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, r in enumerate(latent[:30], 1):
            md.append(
                f"| {i} | {r['heads']} | {r['rank']} | {r['latent_dim']} | "
                f"{r['constructor_hidden']} | {r['fourier_bands']} | {r['best_val_loss']:.4f} | "
                f"{r['final_val_loss']:.4f} | {r['latent_params']} | {r['constructor_params']} | "
                f"{(r['tokens_per_second'] or 0):.0f} |"
            )

    (sweep_dir / "leaderboard.md").write_text("\n".join(md) + "\n")


def run_one(a: argparse.Namespace, spec: RunSpec, runs_dir: Path) -> bool:
    run_dir = runs_dir / spec.name()
    report_path = run_dir / "report.json"
    if report_path.exists() and not a.force:
        print(f"SKIP complete: {spec.name()}")
        return True

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "console.log"
    cmd = command_for(a, spec, runs_dir)
    print("\n" + "=" * 100)
    print(f"RUN {spec.name()}")
    print(" ".join(shlex.quote(x) for x in cmd))
    print("=" * 100)

    env = os.environ.copy()
    with log_path.open("w") as log:
        log.write("COMMAND:\n" + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
        log.write(f"\nEXIT CODE: {code}\n")

    if code != 0 or not report_path.exists():
        print(f"FAILED {spec.name()} (exit={code})")
        return False

    if a.reports_only:
        checkpoint = run_dir / "model.pt"
        if checkpoint.exists():
            checkpoint.unlink()
    return True


def collect_existing(specs: Iterable[RunSpec], runs_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for spec in specs:
        report = read_report(runs_dir / spec.name() / "report.json")
        if report is not None:
            rows.append(extract_row(spec, report))
    return rows


def main() -> None:
    a = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"training script not found: {TRAIN_SCRIPT}")
    if not a.data.exists():
        raise SystemExit(f"data file not found: {a.data}")
    if a.d_model <= 0:
        raise SystemExit("--d-model must be positive")

    specs = build_specs(a)
    bad_heads = sorted({s.heads for s in specs if a.d_model % s.heads != 0})
    if bad_heads:
        raise SystemExit(f"d_model={a.d_model} is not divisible by head counts: {bad_heads}")
    bad_ranks = sorted({s.rank for s in specs if s.rank > min(a.d_model, a.d_ff)})
    if bad_ranks:
        raise SystemExit(f"ranks exceed min(d_model,d_ff): {bad_ranks}")

    sweep_dir = a.output_root / a.name
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": a.name,
        "profile": a.profile,
        "run_count": len(specs),
        "data": str(a.data.resolve()),
        "training": {
            "steps": a.steps,
            "warmup_steps": a.warmup_steps,
            "batch_size": a.batch_size,
            "context": a.context,
            "d_model": a.d_model,
            "layers": a.layers,
            "d_ff": a.d_ff,
            "lr": a.lr,
            "dtype": a.dtype,
            "device": a.device,
        },
        "runs": [asdict(s) | {"run_name": s.name()} for s in specs],
    }
    write_summaries(sweep_dir, collect_existing(specs, runs_dir), manifest)

    print(f"Sweep:   {a.name}")
    print(f"Profile: {a.profile}")
    print(f"Runs:    {len(specs)}")
    print(f"Folder:  {sweep_dir}")
    print(f"Data:    {a.data.resolve()}")
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        print(f"HSA_OVERRIDE_GFX_VERSION={os.environ['HSA_OVERRIDE_GFX_VERSION']}")
    else:
        print("WARNING: HSA_OVERRIDE_GFX_VERSION is not set in this shell.")

    if a.dry_run:
        for i, spec in enumerate(specs, 1):
            print(f"{i:4d}. {spec.name()}")
        return

    start = time.time()
    failures = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] {spec.name()}")
        ok = run_one(a, spec, runs_dir)
        if not ok:
            failures.append(spec.name())
        rows = collect_existing(specs, runs_dir)
        write_summaries(sweep_dir, rows, manifest)
        elapsed = time.time() - start
        print(f"Progress: {len(rows)}/{len(specs)} complete | elapsed {elapsed/3600:.2f} h")
        print(f"Leaderboard: {sweep_dir / 'leaderboard.md'}")

    print("\n" + "=" * 100)
    print("SWEEP COMPLETE")
    print("=" * 100)
    print(f"Results:     {sweep_dir}")
    print(f"Leaderboard: {sweep_dir / 'leaderboard.md'}")
    print(f"CSV:         {sweep_dir / 'summary.csv'}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for name in failures:
            print(f"  - {name}")
    else:
        print("Failures:    none")


if __name__ == "__main__":
    main()
