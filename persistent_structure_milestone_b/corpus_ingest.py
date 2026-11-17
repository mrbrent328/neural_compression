#!/usr/bin/env python3
"""Build a bounded, provenance-preserving Persistent Structure text sample.

This is a research data-preparation tool, not a legal clearance system.
Only exact license strings in the reviewed allowlist are eligible. Composite,
ambiguous, missing, and unrecognized license strings are rejected by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

DATASET_DEFAULT = "PleIAs/common_corpus"
SPLIT_DEFAULT = "train"

# Exact strings only. Do not add aliases or composite expressions here without
# a separate review. In particular, "Public Domain", "CC0", and "CC-By" are
# intentionally NOT aliases for versioned identifiers.
EXACT_LICENSE_ALLOWLIST = frozenset({
    "CC0-1.0",
    "CC-BY-4.0",
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
})

# CC-BY requires attribution. We retain these source fields and reject rows
# where the minimum attribution metadata is absent.
ATTRIBUTION_REQUIRED_LICENSES = frozenset({"CC-BY-4.0"})

PROVENANCE_FIELDS = (
    "identifier",
    "collection",
    "open_type",
    "license",
    "language",
    "language_type",
    "title",
    "creator",
    "curator",
    "date",
    "word_count",
    "token_count",
)


def clean_metadata_value(value: Any) -> Any:
    """Keep source metadata types where possible; normalize empty values to None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    if isinstance(value, (int, float, bool)):
        return value
    # Dataset metadata should be JSON-friendly; stringify unusual scalar values.
    return str(value)


def normalize_text(text: str) -> str:
    """Conservative normalization: NFC and newline normalization only.

    Whitespace and indentation are preserved to avoid damaging code examples.
    """
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _source_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: clean_metadata_value(record.get(field)) for field in PROVENANCE_FIELDS}


def prepare_record(
    record: Mapping[str, Any],
    dataset: str,
    revision: str,
    split: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return a normalized JSONL object or a stable rejection reason."""
    raw_license = record.get("license")
    license_value = raw_license.strip() if isinstance(raw_license, str) else ""

    if license_value not in EXACT_LICENSE_ALLOWLIST:
        return None, "license_not_allowlisted"

    raw_text = record.get("text")
    if not isinstance(raw_text, str):
        return None, "text_not_string"
    if "\x00" in raw_text:
        return None, "text_contains_nul"

    normalized = normalize_text(raw_text)
    if not normalized.strip():
        return None, "text_empty"

    source = _source_metadata(record)
    if license_value in ATTRIBUTION_REQUIRED_LICENSES:
        if not all(_nonempty(source.get(key)) for key in ("identifier", "title", "creator")):
            return None, "required_attribution_metadata_missing"

    raw_bytes = raw_text.encode("utf-8")
    normalized_bytes = normalized.encode("utf-8")
    text_hash = sha256_bytes(normalized_bytes)

    item = {
        "schema_version": "persistent_structure_corpus_record_v1",
        "text": normalized,
        "text_sha256": text_hash,
        "source_text_sha256": sha256_bytes(raw_bytes),
        "source": {
            "dataset": dataset,
            "revision": revision,
            "split": split,
            **source,
        },
        "attribution": {
            "creator": source.get("creator"),
            "title": source.get("title"),
            "identifier": source.get("identifier"),
            "license": license_value,
        },
    }
    return item, None


def ingest_records(
    records: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    dataset: str,
    revision: str,
    split: str,
    max_records: int = 100_000,
    target_bytes: int = 20 * 1024 * 1024,
    max_document_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """Write a deterministic bounded JSONL sample and a reproducibility manifest."""
    if max_records < 1:
        raise ValueError("max_records must be >= 1")
    if target_bytes < 1:
        raise ValueError("target_bytes must be >= 1")
    if max_document_bytes < 1:
        raise ValueError("max_document_bytes must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "corpus.jsonl"
    temp_path = output_dir / "corpus.jsonl.tmp"

    scanned = written = output_bytes = text_bytes = duplicates = 0
    rejected: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    collection_counts: Counter[str] = Counter()
    open_type_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    size_limit_reached = False
    max_records_reached = False

    try:
        with temp_path.open("wb") as handle:
            for record in records:
                if scanned >= max_records:
                    max_records_reached = True
                    break
                scanned += 1

                raw_license = record.get("license")
                license_value = raw_license.strip() if isinstance(raw_license, str) else ""
                if license_value not in EXACT_LICENSE_ALLOWLIST:
                    rejected["license_not_allowlisted"] += 1
                    continue

                item, reason = prepare_record(record, dataset, revision, split)
                if reason:
                    rejected[reason] += 1
                    continue

                assert item is not None
                digest = item["text_sha256"]
                if digest in seen_hashes:
                    duplicates += 1
                    rejected["duplicate_normalized_text"] += 1
                    continue

                text_size = len(item["text"].encode("utf-8"))
                if text_size > max_document_bytes:
                    rejected["document_exceeds_max_document_bytes"] += 1
                    continue

                line = (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                if output_bytes + len(line) > target_bytes:
                    size_limit_reached = True
                    break

                handle.write(line)
                seen_hashes.add(digest)
                written += 1
                output_bytes += len(line)
                text_bytes += text_size
                license_counts[license_value] += 1

                source = item["source"]
                collection_counts[str(source.get("collection") or "<MISSING>")] += 1
                open_type_counts[str(source.get("open_type") or "<MISSING>")] += 1
                language_counts[str(source.get("language") or "<MISSING>")] += 1

            handle.flush()
            os.fsync(handle.fileno())

        temp_path.replace(jsonl_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    manifest = {
        "manifest": "persistent_structure_corpus_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "revision": revision,
        "split": split,
        "policy": {
            "name": "strict_exact_license_allowlist_v1",
            "exact_allowlist": sorted(EXACT_LICENSE_ALLOWLIST),
            "composite_license_strings_allowed": False,
            "unknown_or_missing_license_default": "reject",
            "public_domain_requires_separate_documented_basis": True,
            "cc_by_required_metadata": ["identifier", "title", "creator"],
            "legal_clearance_claimed": False,
        },
        "sampling": {
            "max_records_scanned": max_records,
            "target_output_bytes": target_bytes,
            "max_document_bytes": max_document_bytes,
            "selection_method": "bounded_stream_shuffle_seeded_before_ingestion",
            "deduplication": "sha256_of_NFC_and_newline_normalized_text",
        },
        "results": {
            "records_scanned": scanned,
            "records_written": written,
            "duplicates_rejected": duplicates,
            "rejected_by_reason": dict(sorted(rejected.items())),
            "written_by_exact_license": dict(sorted(license_counts.items())),
            "written_by_collection": dict(sorted(collection_counts.items())),
            "written_by_open_type": dict(sorted(open_type_counts.items())),
            "written_by_language": dict(sorted(language_counts.items())),
            "jsonl_bytes": output_bytes,
            "text_utf8_bytes": text_bytes,
            "target_size_reached_or_next_record_would_exceed": size_limit_reached,
            "max_records_reached": max_records_reached,
        },
        "artifacts": {
            "corpus_jsonl": jsonl_path.name,
            "corpus_jsonl_sha256": sha256_file(jsonl_path),
        },
        "limitations": [
            "Eligibility means only that the exact metadata string matched this project's allowlist.",
            "This manifest is not legal advice or a claim of legal clearance.",
            "Source metadata and attribution fields are preserved, but underlying source rights and notice requirements may require separate verification.",
            "The sample is bounded and is not an exhaustive inventory of the source dataset.",
        ],
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def resolve_dataset_revision(dataset: str, requested_revision: str) -> str:
    """Resolve a branch/tag to an immutable dataset commit SHA."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Missing huggingface_hub. Install with: python -m pip install huggingface_hub") from exc

    info = HfApi().dataset_info(repo_id=dataset, revision=requested_revision)
    if not getattr(info, "sha", None):
        raise RuntimeError("Hugging Face did not return an immutable dataset commit SHA.")
    return info.sha


def load_stream(dataset: str, split: str, revision: str, shuffle_buffer: int, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing datasets. Install with: python -m pip install datasets") from exc

    stream = load_dataset(dataset, split=split, streaming=True, revision=revision)
    if shuffle_buffer:
        if shuffle_buffer < 2:
            raise ValueError("shuffle_buffer must be 0 or >= 2")
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_DEFAULT)
    parser.add_argument("--split", default=SPLIT_DEFAULT)
    parser.add_argument("--revision", default="main", help="Branch/tag/commit; resolved to an immutable SHA and recorded")
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--shuffle-buffer", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target-mb", type=float, default=20.0, help="Target JSONL size in MiB; default 20")
    parser.add_argument("--max-document-mb", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("persistent_structure_corpus_sample"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_records < 1 or args.shuffle_buffer < 0 or args.shuffle_buffer == 1:
        print("error: invalid record or shuffle-buffer limit", file=sys.stderr)
        return 2
    if args.target_mb <= 0 or args.max_document_mb <= 0:
        print("error: size limits must be positive", file=sys.stderr)
        return 2

    try:
        resolved_sha = resolve_dataset_revision(args.dataset, args.revision)
        stream = load_stream(args.dataset, args.split, resolved_sha, args.shuffle_buffer, args.seed)
        manifest = ingest_records(
            stream,
            args.output_dir,
            dataset=args.dataset,
            revision=resolved_sha,
            split=args.split,
            max_records=args.max_records,
            target_bytes=int(args.target_mb * 1024 * 1024),
            max_document_bytes=int(args.max_document_mb * 1024 * 1024),
        )
    except Exception as exc:
        print(f"ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "dataset_revision": resolved_sha,
        "records_scanned": manifest["results"]["records_scanned"],
        "records_written": manifest["results"]["records_written"],
        "jsonl_bytes": manifest["results"]["jsonl_bytes"],
        "manifest": str(args.output_dir / "manifest.json"),
        "legal_clearance_claimed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
