#!/usr/bin/env python3
"""Bounded, metadata-only audit of a Hugging Face text dataset.

This tool intentionally does NOT download the full dataset, print document text,
select training examples, or make licensing/legal determinations. It samples a
bounded number of streamed records and reports the exact metadata values seen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_DATASET = "PleIAs/common_corpus"
DEFAULT_SPLIT = "train"
METADATA_FIELDS = ("identifier", "collection", "open_type", "license", "language", "date", "title", "creator")


def _display_value(value: Any) -> str:
    """Return a stable, bounded string representation for metadata reports."""
    if value is None:
        return "<NULL>"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "<BLANK>"
    return str(value)


def summarize_records(records: Iterable[Mapping[str, Any]], limit: int) -> dict[str, Any]:
    """Summarize at most `limit` records without retaining document text."""
    if limit < 1:
        raise ValueError("limit must be >= 1")

    field_presence: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    open_types: Counter[str] = Counter()
    collections: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    missing_or_blank: Counter[str] = Counter()
    examples: dict[str, dict[str, str]] = {}
    inspected = 0
    first_record_fields: list[str] = []

    for record in records:
        if inspected >= limit:
            break
        inspected += 1
        if inspected == 1:
            first_record_fields = sorted(str(k) for k in record.keys())

        for field in record.keys():
            field_presence[str(field)] += 1

        for field in METADATA_FIELDS:
            value = _display_value(record.get(field))
            if value in ("<NULL>", "<BLANK>"):
                missing_or_blank[field] += 1
            if field == "license":
                licenses[value] += 1
                if value not in examples:
                    examples[value] = {
                        "identifier": _display_value(record.get("identifier")),
                        "collection": _display_value(record.get("collection")),
                        "open_type": _display_value(record.get("open_type")),
                    }
            elif field == "open_type":
                open_types[value] += 1
            elif field == "collection":
                collections[value] += 1
            elif field == "language":
                languages[value] += 1

    return {
        "records_inspected": inspected,
        "first_record_fields": first_record_fields,
        "field_presence_counts": dict(sorted(field_presence.items())),
        "license_values": [
            {"value": value, "count": count, "example_metadata": examples[value]}
            for value, count in sorted(licenses.items(), key=lambda item: (-item[1], item[0]))
        ],
        "open_type_values": dict(sorted(open_types.items())),
        "collection_values": dict(sorted(collections.items())),
        "language_values": dict(sorted(languages.items())),
        "missing_or_blank_counts": dict(sorted(missing_or_blank.items())),
        "limitations": [
            "This is a bounded sample, not an exhaustive inventory of dataset metadata.",
            "A license string is reported verbatim after trimming surrounding whitespace; no license equivalence is inferred.",
            "No record is approved for ingestion by this audit.",
            "No document text is included in the report.",
        ],
    }


def load_stream(dataset: str, split: str, revision: str, shuffle_buffer: int, seed: int):
    """Load a bounded-access streaming dataset; imports optional dependency lazily."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install it with: python -m pip install datasets"
        ) from exc

    stream = load_dataset(dataset, split=split, streaming=True, revision=revision)
    if shuffle_buffer:
        if shuffle_buffer < 2:
            raise ValueError("shuffle_buffer must be 0 or >= 2")
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default="main", help="Dataset revision; use a commit hash for reproducibility")
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--shuffle-buffer", type=int, default=0, help="0 = deterministic dataset prefix; otherwise bounded shuffle buffer")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=Path("corpus_audit_report.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_records < 1:
        print("error: --max-records must be >= 1", file=sys.stderr)
        return 2
    if args.shuffle_buffer == 1 or args.shuffle_buffer < 0:
        print("error: --shuffle-buffer must be 0 or >= 2", file=sys.stderr)
        return 2

    try:
        stream = load_stream(args.dataset, args.split, args.revision, args.shuffle_buffer, args.seed)
        summary = summarize_records(stream, args.max_records)
    except Exception as exc:  # report actionable failure; do not hide it
        print(f"audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report = {
        "audit": "persistent_structure_corpus_metadata_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "requested_revision": args.revision,
        "sampling": {
            "method": "streaming_bounded_shuffle" if args.shuffle_buffer else "streaming_sequential_prefix",
            "max_records": args.max_records,
            "shuffle_buffer": args.shuffle_buffer,
            "seed": args.seed if args.shuffle_buffer else None,
        },
        "summary": summary,
        "decision": "INSPECTION_ONLY_NO_LICENSES_APPROVED",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "records_inspected": summary["records_inspected"],
        "distinct_license_values_seen": len(summary["license_values"]),
        "missing_or_blank_license_records": summary["missing_or_blank_counts"].get("license", 0),
        "decision": report["decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
