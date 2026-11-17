#!/usr/bin/env python3
"""Read-only greedy/temperature/top-p generation audit for latent_transformer_v0."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
TRAINER_DIR = ROOT / "latent_transformer_v0"
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

from generation_audit import DEFAULT_PROMPTS, load_model, make_validation_prompts, resolve_checkpoint  # noqa: E402
from inference_cache import clear_inference_cache, factor_cache_bytes, prepare_for_inference  # noqa: E402

SAMPLING_CONFIGS = (
    {"name": "temperature_0.7", "kind": "temperature", "temperature": 0.7, "top_p": None},
    {"name": "temperature_0.9", "kind": "temperature", "temperature": 0.9, "top_p": None},
    {"name": "temperature_1.0", "kind": "temperature", "temperature": 1.0, "top_p": None},
    {"name": "top_p_0.90", "kind": "top_p", "temperature": 1.0, "top_p": 0.90},
    {"name": "top_p_0.95", "kind": "top_p", "temperature": 1.0, "top_p": 0.95},
)
METRIC_KEYS = (
    "unique_byte_fraction", "distinct_2", "repeat_3gram_rate",
    "whitespace_fraction", "printable_fraction", "utf8_valid",
    "replacement_char_count", "mean_raw_entropy_bits",
    "mean_sampling_entropy_bits", "mean_raw_top1_probability",
    "mean_sampling_top1_probability",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-root", default="runs/latent_transformer_v0")
    p.add_argument("--dense", default=None)
    p.add_argument("--lowrank", default=None)
    p.add_argument("--latent", default=None)
    p.add_argument("--corpus", default="corpus.txt")
    p.add_argument("--heldout-prompts", type=int, default=4)
    p.add_argument("--new-bytes", type=int, default=128)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--output-dir", default="runs/generation_audit")
    return p.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def top_p_filter_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest descending-probability prefix reaching top_p."""
    if logits.ndim != 1:
        raise ValueError("Expected one-dimensional logits")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if top_p == 1.0:
        return logits.clone()
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the token that crosses the threshold; remove only later tokens.
    remove = (cumulative - sorted_probs) >= top_p
    filtered_sorted = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, sorted_indices, filtered_sorted)
    return filtered


def derive_sampling_seed(base_seed, strategy, prompt_id, seed_index):
    payload = f"{base_seed}|{strategy}|{prompt_id}|{seed_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def entropy_bits(probabilities):
    p = probabilities.float()
    return -(p * torch.log2(p.clamp_min(1e-30))).sum()


def ngram_repeat_rate(data: bytes, n: int) -> float:
    if len(data) < n:
        return 0.0
    grams = [data[i:i + n] for i in range(len(data) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def mean_tensor(values):
    return float(torch.stack(values).mean().item()) if values else 0.0


def continuation_metrics(data, raw_h, sample_h, raw_p1, sample_p1):
    try:
        decoded = data.decode("utf-8")
        utf8_valid = True
    except UnicodeDecodeError:
        decoded = data.decode("utf-8", errors="replace")
        utf8_valid = False
    size = len(data)
    bigrams = [data[i:i + 2] for i in range(max(0, size - 1))]
    printable = sum(b in (9, 10, 13) or 32 <= b <= 126 for b in data)
    return {
        "unique_byte_fraction": len(set(data)) / max(1, size),
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "repeat_3gram_rate": ngram_repeat_rate(data, 3),
        "whitespace_fraction": sum(b in (9, 10, 11, 12, 13, 32) for b in data) / max(1, size),
        "printable_fraction": printable / max(1, size),
        "utf8_valid": utf8_valid,
        "replacement_char_count": decoded.count("\ufffd"),
        "mean_raw_entropy_bits": mean_tensor(raw_h),
        "mean_sampling_entropy_bits": mean_tensor(sample_h),
        "mean_raw_top1_probability": mean_tensor(raw_p1),
        "mean_sampling_top1_probability": mean_tensor(sample_p1),
    }


def choose_next_token(logits, strategy, generator):
    raw_probs = torch.softmax(logits.float(), dim=-1)
    raw_h, raw_p1 = entropy_bits(raw_probs), raw_probs.max()
    if strategy["kind"] == "greedy":
        sample_probs = raw_probs
        token = torch.argmax(logits, dim=-1)
    else:
        scaled = logits.float() / float(strategy["temperature"])
        if strategy.get("top_p") is not None:
            scaled = top_p_filter_logits(scaled, float(strategy["top_p"]))
        sample_probs = torch.softmax(scaled, dim=-1)
        token = torch.multinomial(sample_probs, 1, generator=generator).squeeze(0)
    return token, raw_h, entropy_bits(sample_probs), raw_p1, sample_probs.max()


@torch.inference_mode()
def generate_continuation(model, prompt_ids, new_bytes, strategy, generator=None):
    if new_bytes < 1:
        raise ValueError("new_bytes must be positive")
    idx = prompt_ids.clone()
    output = bytearray()
    raw_h, sample_h, raw_p1, sample_p1 = [], [], [], []
    for _ in range(new_bytes):
        logits, _ = model(idx[:, -model.cfg.context:])
        token, rh, sh, rp, sp = choose_next_token(logits[0, -1, :], strategy, generator)
        token_id = int(token.item())
        output.append(token_id)
        idx = torch.cat((idx, token.reshape(1, 1)), dim=1)
        raw_h.append(rh)
        sample_h.append(sh)
        raw_p1.append(rp)
        sample_p1.append(sp)
    data = bytes(output)
    return data, continuation_metrics(data, raw_h, sample_h, raw_p1, sample_p1)


def summarize_rows(rows):
    result = {"count": len(rows)}
    for key in METRIC_KEYS:
        values = [float(row["metrics"][key]) for row in rows]
        result[key] = {
            "mean": statistics.fmean(values) if values else None,
            "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        }
    return result


def pairwise_seed_agreement(rows):
    outputs = [bytes.fromhex(r["generated_bytes_hex"]) for r in rows
               if r.get("seed_index") is not None]
    scores = []
    for left, right in combinations(outputs, 2):
        n = min(len(left), len(right))
        if n:
            scores.append(sum(a == b for a, b in zip(left[:n], right[:n])) / n)
    return statistics.fmean(scores) if scores else None


def build_model_summary(generations, prompt_ids):
    names = ["greedy"] + [x["name"] for x in SAMPLING_CONFIGS]
    summary = {}
    for name in names:
        rows = [r for r in generations if r["strategy"] == name]
        by_prompt, agreements = {}, []
        for prompt_id in prompt_ids:
            subset = [r for r in rows if r["prompt_id"] == prompt_id]
            agreement = pairwise_seed_agreement(subset)
            if agreement is not None:
                agreements.append(agreement)
            by_prompt[prompt_id] = {
                "metrics": summarize_rows(subset),
                "mean_pairwise_seed_byte_agreement": agreement,
            }
        summary[name] = {
            "overall": summarize_rows(rows),
            "mean_pairwise_seed_byte_agreement": statistics.fmean(agreements) if agreements else None,
            "by_prompt": by_prompt,
        }
    return summary


def verify_greedy_parity(model, prompt_ids, new_bytes):
    count = min(8, new_bytes)
    expected = model.generate(prompt_ids.clone(), max_new_tokens=count)[0, prompt_ids.shape[1]:]
    generated, _ = generate_continuation(
        model, prompt_ids, count, {"name": "greedy", "kind": "greedy"}
    )
    actual = torch.tensor(list(generated), dtype=torch.long, device=expected.device)
    if not torch.equal(expected, actual):
        raise AssertionError("Custom greedy decoder differs from TinyLM.generate().")
    return {"checked_tokens": count, "bit_identical": True}


def main():
    args = parse_args()
    if args.new_bytes < 1 or args.seeds < 1 or args.heldout_prompts < 0:
        raise SystemExit("--new-bytes and --seeds must be positive; heldout-prompts cannot be negative")
    device = choose_device(args.device)
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
    if not prompts:
        raise SystemExit("No prompts available.")

    overrides = {"dense": args.dense, "lowrank": args.lowrank, "latent": args.latent}
    results = {
        "experiment": "latent_transformer_v0_generation_sampling_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repository": "mrbrent328/neural_compression",
        "device": str(device),
        "torch_version": torch.__version__,
        "settings": vars(args),
        "methodology": {
            "prompt_count": len(prompts),
            "fixed_prompt_count": len(DEFAULT_PROMPTS),
            "heldout_prompt_count": len(heldout),
            "sampling_seeds_per_prompt_and_configuration": args.seeds,
            "greedy_repeats": 1,
            "generated_bytes_per_case": args.new_bytes,
            "seed_derivation": "SHA-256(base_seed|strategy|prompt_id|seed_index), first 8 bytes",
            "raw_entropy": "Unmodified next-byte distribution at temperature 1.0",
            "sampling_entropy": "Distribution after temperature and optional top-p filtering",
            "top_p_rule": "Smallest descending-probability prefix whose cumulative mass reaches top_p",
            "read_only": True,
        },
        "prompts": prompts,
        "decoding_configurations": [
            {"name": "greedy", "kind": "greedy", "temperature": None, "top_p": None}
        ] + list(SAMPLING_CONFIGS),
        "models": {},
    }
    transcript = [
        "LATENT TRANSFORMER V0 — GENERATION SAMPLING AUDIT",
        f"Created: {results['created_at']}", f"Device: {device} | PyTorch: {torch.__version__}",
        "Greedy baseline; temperature 0.7/0.9/1.0; top-p 0.90/0.95.",
        f"Sampling seeds per prompt/configuration: {args.seeds}",
        f"Generated bytes per case: {args.new_bytes}",
        "Read-only checkpoints. Latent factor cache is enabled only after bit-identical logits check.",
    ]

    for mode in ("dense", "lowrank", "latent"):
        path = resolve_checkpoint(run_root, mode, overrides[mode])
        print(f"Loading {mode}: {path}", flush=True)
        model, checkpoint, metadata = load_model(path, device)
        cache_info = {"enabled": False, "strategy": "none", "factor_cache_bytes": 0}
        if mode == "latent":
            probe_bytes = prompts[0]["prompt"].encode("utf-8", errors="replace")[-model.cfg.context:]
            probe = torch.tensor([list(probe_bytes)], dtype=torch.long, device=device)
            with torch.inference_mode():
                uncached, _ = model(probe)
            prepare_for_inference(model, strategy="all")
            with torch.inference_mode():
                cached, _ = model(probe)
            if not torch.equal(uncached, cached):
                raise AssertionError("Latent factor cache changed logits.")
            cache_info = {
                "enabled": True, "strategy": "all",
                "factor_cache_bytes": factor_cache_bytes(model),
                "logits_bit_identical": True,
            }

        first = prompts[0]["prompt"].encode("utf-8", errors="replace")[-model.cfg.context:]
        first_ids = torch.tensor([list(first)], dtype=torch.long, device=device)
        parity = verify_greedy_parity(model, first_ids, args.new_bytes)
        model_result = {
            "metadata": metadata,
            "inference_factor_cache": cache_info,
            "greedy_parity_check": parity,
            "generations": [],
        }
        transcript.extend([
            "\n" + "=" * 88, f"MODEL: {mode.upper()}",
            f"Parameters: {metadata['parameters']['total']:,}",
            f"Checkpoint bytes: {metadata['checkpoint_bytes']:,}",
            f"SHA-256: {metadata['sha256']}",
            f"Factor cache bytes: {cache_info['factor_cache_bytes']:,}",
            "Greedy parity with TinyLM.generate(): PASS",
        ])

        for prompt_item in prompts:
            prompt_bytes = prompt_item["prompt"].encode("utf-8", errors="replace")[-model.cfg.context:]
            prompt_text = prompt_bytes.decode("utf-8", errors="replace")
            prompt_ids = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
            cases = [({"name": "greedy", "kind": "greedy", "temperature": None, "top_p": None}, None, None)]
            for config in SAMPLING_CONFIGS:
                for seed_index in range(args.seeds):
                    seed = derive_sampling_seed(args.seed, config["name"], prompt_item["id"], seed_index)
                    cases.append((config, seed, seed_index))

            for config, sampling_seed, seed_index in cases:
                generator = None
                if sampling_seed is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(sampling_seed)
                generated, metrics = generate_continuation(
                    model, prompt_ids, args.new_bytes, config, generator
                )
                row = {
                    "prompt_id": prompt_item["id"], "prompt": prompt_text,
                    "source": prompt_item.get("source"), "strategy": config["name"],
                    "kind": config["kind"], "temperature": config.get("temperature"),
                    "top_p": config.get("top_p"), "seed_index": seed_index,
                    "sampling_seed": sampling_seed, "generated_bytes": len(generated),
                    "generated_bytes_hex": generated.hex(),
                    "generated_continuation": generated.decode("utf-8", errors="replace"),
                    "reference_continuation": prompt_item.get("reference_continuation"),
                    "metrics": metrics,
                }
                model_result["generations"].append(row)
                transcript.extend([
                    f"\n--- {prompt_item['id']} | {config['name']} | seed={seed_index} ---",
                    "PROMPT:", prompt_text, "GENERATED:", row["generated_continuation"],
                    "METRICS:", json.dumps(metrics, sort_keys=True),
                ])
                if row["reference_continuation"] is not None and config["name"] == "greedy":
                    transcript.extend(["REFERENCE (not shown to model):", row["reference_continuation"]])
            print(f"  {mode}: completed {prompt_item['id']}", flush=True)

        model_result["summary"] = build_model_summary(
            model_result["generations"], [p["id"] for p in prompts]
        )
        results["models"][mode] = model_result
        clear_inference_cache(model)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = outdir / f"generation_sampling_audit_{stamp}.json"
    txt_path = outdir / f"generation_sampling_audit_{stamp}.txt"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(transcript) + "\n", encoding="utf-8")
    print("\nSAMPLING AUDIT COMPLETE")
    print(f"JSON: {json_path}")
    print(f"TEXT: {txt_path}")
    print("No training was performed; no checkpoint was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
