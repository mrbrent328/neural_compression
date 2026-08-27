#!/usr/bin/env python3
"""Compare representations for communicating synchronized state changes.

The same exact state change is encoded several ways and then zlib-compressed:
absolute indices, index differences, contiguous ranges, factored values, and a
combined representation. Every representation is decoded and checked for exact
recovery before its size is considered valid.
"""

import json
import struct
import zlib
from pathlib import Path

import numpy as np

SEED = 314
STATE_SIZE = 1024


def pack_f32(values):
    return np.asarray(values, dtype="<f4").tobytes()


def encode_absolute(indices, values):
    return struct.pack("<BI", 1, len(indices)) + indices.astype("<i4").tobytes() + pack_f32(values)


def encode_index_delta(indices, values):
    if len(indices):
        gaps = np.diff(np.concatenate(([indices[0]], indices))).astype(np.int32)
    else:
        gaps = np.array([], dtype=np.int32)
    return struct.pack("<BI", 2, len(indices)) + gaps.astype("<i4").tobytes() + pack_f32(values)


def find_ranges(indices):
    if len(indices) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    starts = [int(indices[0])]
    lengths = []
    run_start = 0
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            lengths.append(i - run_start)
            starts.append(int(indices[i]))
            run_start = i
    lengths.append(len(indices) - run_start)
    return np.asarray(starts, dtype=np.int32), np.asarray(lengths, dtype=np.int32)


def encode_ranges(indices, values):
    starts, lengths = find_ranges(indices)
    return (struct.pack("<BI", 3, len(starts)) + starts.astype("<i4").tobytes() +
            lengths.astype("<i4").tobytes() + pack_f32(values))


def encode_factored(indices, values):
    if len(values) == 0:
        return struct.pack("<BI", 4, 0)
    base = float(np.mean(values))
    residuals = (values - base).astype(np.float32)
    return (struct.pack("<BI f", 4, len(indices), base) +
            indices.astype("<i4").tobytes() + pack_f32(residuals))


def encode_combined(indices, values):
    starts, lengths = find_ranges(indices)
    base = float(np.mean(values)) if len(values) else 0.0
    residuals = (values - base).astype(np.float32) if len(values) else np.array([], dtype=np.float32)
    return (struct.pack("<BI fI", 5, len(starts), base, len(indices)) +
            starts.astype("<i4").tobytes() + lengths.astype("<i4").tobytes() +
            pack_f32(residuals))


def decode_and_reconstruct(payload):
    kind, count = struct.unpack("<BI", payload[:5])
    offset = 5

    if kind == 1:
        indices = np.frombuffer(payload[offset:offset + 4 * count], dtype="<i4").copy()
        offset += 4 * count
        values = np.frombuffer(payload[offset:offset + 4 * count], dtype="<f4").copy()
        return indices, values

    if kind == 2:
        gaps = np.frombuffer(payload[offset:offset + 4 * count], dtype="<i4").copy()
        offset += 4 * count
        indices = np.cumsum(gaps, dtype=np.int32) if count else np.array([], dtype=np.int32)
        values = np.frombuffer(payload[offset:offset + 4 * count], dtype="<f4").copy()
        return indices, values

    if kind == 3:
        starts = np.frombuffer(payload[offset:offset + 4 * count], dtype="<i4").copy()
        offset += 4 * count
        lengths = np.frombuffer(payload[offset:offset + 4 * count], dtype="<i4").copy()
        offset += 4 * count
        if count:
            indices = np.concatenate([np.arange(s, s + n, dtype=np.int32) for s, n in zip(starts, lengths)])
        else:
            indices = np.array([], dtype=np.int32)
        values = np.frombuffer(payload[offset:offset + 4 * len(indices)], dtype="<f4").copy()
        return indices, values

    if kind == 4:
        (base,) = struct.unpack("<f", payload[offset:offset + 4])
        offset += 4
        indices = np.frombuffer(payload[offset:offset + 4 * count], dtype="<i4").copy()
        offset += 4 * count
        residuals = np.frombuffer(payload[offset:offset + 4 * count], dtype="<f4").copy()
        return indices, (base + residuals).astype(np.float32)

    if kind == 5:
        range_count, base, value_count = struct.unpack("<I f I", payload[offset:offset + 12])
        offset += 12
        starts = np.frombuffer(payload[offset:offset + 4 * range_count], dtype="<i4").copy()
        offset += 4 * range_count
        lengths = np.frombuffer(payload[offset:offset + 4 * range_count], dtype="<i4").copy()
        offset += 4 * range_count
        if range_count:
            indices = np.concatenate([np.arange(s, s + n, dtype=np.int32) for s, n in zip(starts, lengths)])
        else:
            indices = np.array([], dtype=np.int32)
        residuals = np.frombuffer(payload[offset:offset + 4 * value_count], dtype="<f4").copy()
        return indices, (base + residuals).astype(np.float32)

    raise ValueError(f"unknown representation {kind}")


ENCODERS = {
    "absolute": encode_absolute,
    "index_delta": encode_index_delta,
    "ranges": encode_ranges,
    "factored_values": encode_factored,
    "combined": encode_combined,
}


def run_case(rng, changed, pattern):
    if pattern == "clustered":
        start = int(rng.integers(0, STATE_SIZE - changed + 1))
        indices = np.arange(start, start + changed, dtype=np.int32)
        values = rng.normal(0.01, 0.01, changed).astype(np.float32)
    else:
        indices = np.sort(rng.choice(STATE_SIZE, changed, replace=False)).astype(np.int32)
        if pattern == "similar":
            values = rng.normal(0.01, 0.0001, changed).astype(np.float32)
        elif pattern == "random":
            values = rng.normal(0.0, 0.01, changed).astype(np.float32)
        else:
            raise ValueError(pattern)

    results = {}
    for name, encoder in ENCODERS.items():
        raw = encoder(indices, values)
        compressed = zlib.compress(raw, level=9)
        decoded_indices, decoded_values = decode_and_reconstruct(zlib.decompress(compressed))
        exact = bool(np.array_equal(indices, decoded_indices) and np.array_equal(values, decoded_values))
        results[name] = {
            "raw_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "reduction_vs_full_percent": 100.0 * (1.0 - len(compressed) / (STATE_SIZE * 4)),
            "exact": exact,
        }

    best = min(results, key=lambda name: results[name]["compressed_bytes"])
    return {
        "pattern": pattern,
        "changed_values": changed,
        "representations": results,
        "best_representation": best,
        "best_compressed_bytes": results[best]["compressed_bytes"],
    }


def main():
    rng = np.random.default_rng(SEED)
    results = [run_case(rng, changed, pattern)
               for pattern in ("random", "clustered", "similar")
               for changed in (1, 4, 16, 64, 256, 512, 1024)]
    output = {
        "experiment": "structured delta representation",
        "seed": SEED,
        "state_values": STATE_SIZE,
        "results": results,
    }
    Path("structured_delta_result.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
