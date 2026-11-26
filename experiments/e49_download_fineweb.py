#!/usr/bin/env python3
"""Stream a reproducible, deduplicated FineWeb sample into E49 uint32 splits.

The script does not download the full FineWeb corpus. It streams the
HuggingFaceFW/fineweb sample-10BT configuration, tokenizes documents with the
E49 tokenizer, and writes train.bin, validation.bin, test.bin, and manifest.json.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np


DATASET_ID = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"
TOKENIZER_ID = "mistralai/Mistral-7B-v0.1"
SPLIT_THRESHOLDS = (100, 110, 120)  # 100:10:10 = 20M:2M:2M


class BKTree:
    """Small Hamming-distance BK-tree for conservative 64-bit SimHash filtering."""

    def __init__(self):
        self.root = None

    def contains_near(self, value, max_distance):
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node_value, children = stack.pop()
            distance = (value ^ node_value).bit_count()
            if distance <= max_distance:
                return True
            low = max(0, distance - max_distance)
            high = distance + max_distance
            for edge, child in children.items():
                if low <= edge <= high:
                    stack.append(child)
        return False

    def add(self, value):
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            node_value, children = node
            distance = (value ^ node_value).bit_count()
            if distance == 0:
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def simhash64(normalized_text):
    """Compute a 64-bit word-5-gram SimHash, sampling long documents evenly."""
    words = re.findall(r"\w+", normalized_text, flags=re.UNICODE)
    if len(words) < 5:
        shingles = [" ".join(words)]
    else:
        step = max(1, math.ceil((len(words) - 4) / 2048))
        sampled = words[::step]
        if len(sampled) < 5:
            sampled = words[:5]
        shingles = (
            " ".join(sampled[index:index + 5])
            for index in range(len(sampled) - 4)
        )

    weights = np.zeros(64, dtype=np.int32)
    used = 0
    for shingle in shingles:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8), bitorder="little")
        weights += bits.astype(np.int32) * 2 - 1
        used += 1
    if used == 0:
        return 0
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return signature


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/e49_fineweb_v1"))
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--train-tokens", type=int, default=20_000_000)
    parser.add_argument("--validation-tokens", type=int, default=2_000_000)
    parser.add_argument("--test-tokens", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--min-document-tokens", type=int, default=64)
    parser.add_argument("--max-documents", type=int, default=1_000_000)
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    targets = {
        "train": args.train_tokens,
        "validation": args.validation_tokens,
        "test": args.test_tokens,
    }
    if min(targets.values()) <= 0:
        raise SystemExit("All split token targets must be positive.")
    if args.shuffle_buffer < 1 or args.max_documents < 1:
        raise SystemExit("Shuffle buffer and max documents must be positive.")
    if not 0 <= args.near_duplicate_distance <= 8:
        raise SystemExit("--near-duplicate-distance must be between 0 and 8.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {name: output_dir / f"{name}.bin" for name in targets}
    manifest_path = output_dir / "manifest.json"
    occupied = [path for path in [*final_paths.values(), manifest_path] if path.exists()]
    if occupied:
        raise SystemExit(
            "Output already contains E49 dataset files; choose a new --output-dir. "
            "No existing files were changed: " + ", ".join(str(path) for path in occupied)
        )

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing packages. Install with: "
            "python -m pip install 'datasets>=3.0' 'transformers>=4.45' "
            "tokenizers numpy"
        ) from exc

    print("Resolving immutable dataset revision...", flush=True)
    dataset_info = HfApi().dataset_info(args.dataset, revision=args.dataset_revision)
    dataset_sha = dataset_info.sha
    print(f"Dataset revision: {dataset_sha}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision, use_fast=True
    )
    vocab_size = len(tokenizer)
    if vocab_size != 32000:
        raise SystemExit(
            f"E49's current data/model setup expects vocab size 32000; "
            f"tokenizer reports {vocab_size}. Stop rather than silently mismatch."
        )
    eos_id = tokenizer.eos_token_id
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")

    print("Opening FineWeb streaming dataset (no full-corpus download)...", flush=True)
    stream = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split="train",
        streaming=True,
        revision=dataset_sha,
    )
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    temporary_paths = {
        name: output_dir / f"{name}.bin.partial" for name in targets
    }
    counts = {name: 0 for name in targets}
    documents = {name: [] for name in targets}
    exact_hashes = set()
    near_duplicate_index = BKTree()
    examined = 0
    accepted = 0
    rejected = {
        "not_english": 0,
        "low_language_score": 0,
        "short_or_long": 0,
        "too_few_tokens": 0,
        "exact_duplicate": 0,
        "near_duplicate": 0,
    }

    handles = {}
    try:
        handles = {
            name: temporary_paths[name].open("wb")
            for name in targets
        }
        for row in stream:
            examined += 1
            if examined > args.max_documents:
                break

            text = row.get("text")
            if not isinstance(text, str):
                rejected["short_or_long"] += 1
                continue
            if row.get("language") not in (None, "en"):
                rejected["not_english"] += 1
                continue
            language_score = row.get("language_score")
            if language_score is not None and float(language_score) < 0.65:
                rejected["low_language_score"] += 1
                continue
            if len(text) < args.min_chars or len(text) > args.max_chars:
                rejected["short_or_long"] += 1
                continue

            normalized = normalize_text(text)
            if not normalized:
                rejected["short_or_long"] += 1
                continue
            text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if text_hash in exact_hashes:
                rejected["exact_duplicate"] += 1
                continue

            signature = simhash64(normalized)
            if near_duplicate_index.contains_near(
                signature, args.near_duplicate_distance
            ):
                rejected["near_duplicate"] += 1
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < args.min_document_tokens:
                rejected["too_few_tokens"] += 1
                continue

            # Stable content-hash assignment prevents the same document from
            # drifting between train/validation/test across different stream orders.
            bucket = int(text_hash[:16], 16) % SPLIT_THRESHOLDS[-1]
            if bucket < SPLIT_THRESHOLDS[0]:
                split = "train"
            elif bucket < SPLIT_THRESHOLDS[1]:
                split = "validation"
            else:
                split = "test"

            # Index every accepted unique document, including documents for a
            # split that has already reached its target, to prevent leakage.
            exact_hashes.add(text_hash)
            near_duplicate_index.add(signature)
            accepted += 1

            remaining = targets[split] - counts[split]
            if remaining <= 0:
                continue

            take = min(len(token_ids), remaining)
            written_ids = token_ids[:take]
            truncated = take < len(token_ids)
            if eos_id is not None and not truncated and take < remaining:
                written_ids = written_ids + [eos_id]
            token_array = np.asarray(written_ids, dtype=np.uint32)
            token_array.tofile(handles[split])
            counts[split] += int(token_array.size)

            documents[split].append({
                "id": row.get("id"),
                "url": row.get("url"),
                "dump": row.get("dump"),
                "date": row.get("date"),
                "normalized_text_sha256": text_hash,
                "tokens_written": int(token_array.size),
                "truncated_at_split_target": truncated,
            })

            if examined % 1000 == 0:
                status = " ".join(
                    f"{name}={counts[name]:,}/{targets[name]:,}"
                    for name in targets
                )
                print(
                    f"documents={examined:,} accepted={accepted:,} "
                    f"{status}",
                    flush=True,
                )

            if all(counts[name] >= targets[name] for name in targets):
                break

        for handle in handles.values():
            handle.flush()
            handle.close()
        handles = {}

        incomplete = {
            name: (counts[name], targets[name])
            for name in targets if counts[name] < targets[name]
        }
        if incomplete:
            raise RuntimeError(
                "Stream ended or document limit reached before all targets "
                f"were filled: {incomplete}. Increase --max-documents or "
                "choose another dataset configuration."
            )

        split_metadata = {}
        for name, path in temporary_paths.items():
            final_path = final_paths[name]
            path.replace(final_path)
            split_metadata[name] = {
                "tokens": counts[name],
                "bytes": final_path.stat().st_size,
                "sha256": sha256_file(final_path),
                "documents_written": len(documents[name]),
                "sources": documents[name],
            }

        manifest = {
            "experiment": "E49",
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "dataset_revision_requested": args.dataset_revision,
            "dataset_revision_resolved": dataset_sha,
            "dataset_license": "ODC-By-1.0",
            "dataset_license_url": "https://opendatacommons.org/licenses/by/1-0/",
            "dataset_card": "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
            "dataset_access": "Hugging Face datasets streaming; full corpus not downloaded",
            "tokenizer": args.tokenizer,
            "tokenizer_revision_requested": args.tokenizer_revision,
            "tokenizer_revision_resolved": tokenizer_commit,
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "seed": args.seed,
            "shuffle_buffer": args.shuffle_buffer,
            "split_method": (
                "stable normalized-content SHA256 bucket assignment; "
                "100:10:10 buckets for train:validation:test"
            ),
            "deduplication": {
                "exact": "SHA256 of NFKC-normalized, lowercased, whitespace-collapsed text",
                "near_duplicate": (
                    "64-bit word-5-gram SimHash with BK-tree Hamming threshold "
                    f"{args.near_duplicate_distance}; global across all splits"
                ),
                "note": (
                    "FineWeb is already deduplicated within each crawl; this "
                    "additional conservative filter screens the sampled stream."
                ),
            },
            "filtering": {
                "language": "English; rows with absent language metadata are retained",
                "minimum_language_score": 0.65,
                "minimum_characters": args.min_chars,
                "maximum_characters": args.max_chars,
                "minimum_document_tokens": args.min_document_tokens,
            },
            "targets": targets,
            "actual_split_data": split_metadata,
            "stream_statistics": {
                "documents_examined": examined,
                "unique_documents_accepted": accepted,
                "rejected": rejected,
            },
            "raw_document_text_saved": False,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            "dataset_revision": dataset_sha,
            "splits": {
                name: {
                    "tokens": counts[name],
                    "bytes": final_paths[name].stat().st_size,
                    "documents": len(documents[name]),
                }
                for name in targets
            },
            "documents_examined": examined,
            "rejected": rejected,
        }, indent=2), flush=True)
        print("Dataset preparation complete.", flush=True)

    except BaseException:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        # Remove only final split files created by this invocation if the
        # manifest was not committed; never touch unrelated user files.
        if not manifest_path.exists():
            for path in final_paths.values():
                if path.exists():
                    path.unlink()
        raise


if __name__ == "__main__":
    main()
