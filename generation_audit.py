#!/usr/bin/env python3
"""Read-only generation audit for latent_transformer_v0 checkpoints.

Run from the neural_compression repository root:
  python generation_audit.py --device auto

Loads existing dense/lowrank/latent checkpoints; does not train or overwrite them.
Writes a timestamped JSON report and readable TXT transcript under
runs/generation_audit/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
TRAINER_DIR = ROOT / "latent_transformer_v0"
sys.path.insert(0, str(TRAINER_DIR))
from latent_transformer_v0 import ModelConfig, TinyLM, split_data, load_bytes, count_parameters

DEFAULT_PROMPTS = [
    ("documentation_english", "The component is "),
    ("documentation_english_2", "A value set may "),
    ("technical_requirement", "The system SHALL "),
    ("csharp", "public static void Main(string[] args) {\n"),
    ("python", "def calculate_total(items):\n"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(root: Path, mode: str, override: str | None) -> Path:
    if override:
        p = Path(override).expanduser()
        return p if p.is_absolute() else ROOT / p
    return root / f"{mode}_s1337" / "model.pt"


def load_model(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Pass --dense, --lowrank, or --latent with the correct model.pt path."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = TinyLM(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model, ckpt, {
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint_bytes": path.stat().st_size,
        "parameters": count_parameters(model),
        "config": ckpt["model_config"],
        "strict_load": True,
    }


def make_validation_prompts(corpus_path: Path | None, count: int, context: int, seed: int):
    if corpus_path is None or not corpus_path.is_file() or count <= 0:
        return []
    data = load_bytes(corpus_path, smoke=False)
    _, val = split_data(data)
    raw = bytes(val.tolist())
    rng = random.Random(seed)
    prompts = []
    # Sample windows from the held-out tail. Prefix is model input; following
    # bytes are retained only as a reference continuation for human comparison.
    min_prefix, target_len = 48, 96
    max_start = max(0, len(raw) - min_prefix - target_len - 1)
    attempts = 0
    while len(prompts) < count and attempts < count * 100:
        attempts += 1
        start = rng.randint(0, max_start)
        prefix = raw[start:start + min_prefix]
        target = raw[start + min_prefix:start + min_prefix + target_len]
        try:
            ptxt = prefix.decode("utf-8")
            ttxt = target.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # Prefer readable source/documentation windows, not binary-looking text.
        if sum(ch.isprintable() or ch in "\n\r\t" for ch in ptxt) / max(1, len(ptxt)) < 0.92:
            continue
        prompts.append({
            "id": f"heldout_{len(prompts)+1}",
            "prompt": ptxt,
            "reference_continuation": ttxt,
            "source": "held-out final 10% of corpus",
        })
    return prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-root", default="runs/latent_transformer_v0",
                    help="Directory containing dense_s1337/model.pt etc.")
    ap.add_argument("--dense", default=None, help="Optional explicit dense model.pt path")
    ap.add_argument("--lowrank", default=None, help="Optional explicit lowrank model.pt path")
    ap.add_argument("--latent", default=None, help="Optional explicit latent model.pt path")
    ap.add_argument("--corpus", default="corpus.txt", help="Training corpus for held-out prompt sampling")
    ap.add_argument("--heldout-prompts", type=int, default=4)
    ap.add_argument("--new-bytes", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--output-dir", default="runs/generation_audit")
    args = ap.parse_args()

    if args.new_bytes < 1:
        ap.error("--new-bytes must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    corpus = Path(args.corpus) if args.corpus else None
    if corpus is not None and not corpus.is_absolute():
        corpus = ROOT / corpus

    heldout = make_validation_prompts(corpus, args.heldout_prompts, 0, args.seed)
    prompts = [
        {"id": name, "prompt": prompt, "source": "fixed diagnostic prompt"}
        for name, prompt in DEFAULT_PROMPTS
    ] + heldout

    modes = ("dense", "lowrank", "latent")
    overrides = {"dense": args.dense, "lowrank": args.lowrank, "latent": args.latent}
    results = {
        "experiment": "latent_transformer_v0_generation_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repository_root": str(ROOT),
        "device": str(device),
        "settings": vars(args),
        "prompts": prompts,
        "models": {},
        "read_only": True,
    }
    transcript = [
        "LATENT TRANSFORMER V0 — GENERATION AUDIT",
        f"Created: {results['created_at']}", f"Device: {device}",
        "Greedy decoding; same prompt and byte budget for each checkpoint.",
        "Existing checkpoints are loaded read-only.\n",
    ]

    for mode in modes:
        path = resolve_checkpoint(run_root, mode, overrides[mode])
        print(f"Loading {mode}: {path}", flush=True)
        model, ckpt, meta = load_model(path, device)
        model_results = {"metadata": meta, "generations": []}
        transcript.extend(["\n" + "=" * 88, f"MODEL: {mode.upper()}",
                           f"Parameters: {meta['parameters']['total']:,}",
                           f"Checkpoint: {meta['checkpoint_bytes']:,} bytes",
                           f"SHA-256: {meta['sha256']}"])
        for item in prompts:
            prompt_bytes = item["prompt"].encode("utf-8", errors="replace")
            if len(prompt_bytes) > model.cfg.context:
                prompt_bytes = prompt_bytes[-model.cfg.context:]
            prompt_ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
            with torch.inference_mode():
                generated = model.generate(prompt_ids.clone(), args.new_bytes)[0].cpu().tolist()
            continuation = bytes(generated[len(prompt_bytes):]).decode("utf-8", errors="replace")
            row = {
                "prompt_id": item["id"],
                "prompt": prompt_bytes.decode("utf-8", errors="replace"),
                "generated_continuation": continuation,
                "reference_continuation": item.get("reference_continuation"),
                "source": item.get("source"),
                "generated_bytes": args.new_bytes,
            }
            model_results["generations"].append(row)
            transcript.extend([
                f"\n--- {item['id']} ---",
                "PROMPT:", row["prompt"],
                "GENERATED:", continuation,
            ])
            if row["reference_continuation"] is not None:
                transcript.extend(["REFERENCE (held-out actual text; not shown to model):",
                                   row["reference_continuation"]])
        results["models"][mode] = model_results
        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = outdir / f"generation_audit_{stamp}.json"
    txt_path = outdir / f"generation_audit_{stamp}.txt"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(transcript) + "\n", encoding="utf-8")
    print("\nAUDIT COMPLETE")
    print(f"JSON: {json_path}")
    print(f"TEXT: {txt_path}")
    print("No training was performed; no checkpoint was modified.")


if __name__ == "__main__":
    main()
