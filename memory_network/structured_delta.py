#!/usr/bin/env python3
"""Compare representations for communicating synchronized state changes."""

import json
import struct
import zlib
from pathlib import Path
import numpy as np

SEED = 314
STATE_SIZE = 1024

# Packet header: magic, version, representation, index_count, range_count, value_count.
HEADER = struct.Struct("<4sBBIII")
MAGIC = b"SDLT"
VERSION = 1


def pack_f32(values):
    return np.asarray(values, dtype="<f4").tobytes()


def pack_packet(kind, index_count, range_count, value_count, payload):
    return HEADER.pack(MAGIC, VERSION, kind, index_count, range_count, value_count) + payload


def encode_absolute(indices, values):
    return pack_packet(1, len(indices), 0, len(values), indices.astype("<i4").tobytes() + pack_f32(values))


def encode_index_delta(indices, values):
    if len(indices):
        gaps = np.empty(len(indices), dtype=np.int32)
        gaps[0] = indices[0]
        if len(indices) > 1:
            gaps[1:] = np.diff(indices)
    else:
        gaps = np.array([], dtype=np.int32)
    return pack_packet(2, len(indices), 0, len(values), gaps.astype("<i4").tobytes() + pack_f32(values))


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
    payload = starts.astype("<i4").tobytes() + lengths.astype("<i4").tobytes() + pack_f32(values)
    return pack_packet(3, len(indices), len(starts), len(values), payload)


def encode_factored(indices, values):
    base = float(np.mean(values)) if len(values) else 0.0
    residuals = (values - base).astype(np.float32) if len(values) else np.array([], dtype=np.float32)
    payload = struct.pack("<f", base) + indices.astype("<i4").tobytes() + pack_f32(residuals)
    return pack_packet(4, len(indices), 0, len(values), payload)


def encode_combined(indices, values):
    starts, lengths = find_ranges(indices)
    base = float(np.mean(values)) if len(values) else 0.0
    residuals = (values - base).astype(np.float32) if len(values) else np.array([], dtype=np.float32)
    payload = (struct.pack("<f", base) + starts.astype("<i4").tobytes() +
               lengths.astype("<i4").tobytes() + pack_f32(residuals))
    return pack_packet(5, len(indices), len(starts), len(values), payload)


def read_packet(payload):
    if len(payload) < HEADER.size:
        raise ValueError("payload is shorter than packet header")
    magic, version, kind, index_count, range_count, value_count = HEADER.unpack(payload[:HEADER.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("invalid structured delta packet header")
    if index_count < 0 or range_count < 0 or value_count < 0:
        raise ValueError("negative packet count")
    return kind, index_count, range_count, value_count, payload[HEADER.size:]


def take_array(payload, offset, count, dtype):
    item_size = np.dtype(dtype).itemsize
    end = offset + item_size * count
    if end > len(payload):
        raise ValueError("structured delta packet is truncated")
    return np.frombuffer(payload[offset:end], dtype=dtype).copy(), end


def ranges_to_indices(starts, lengths, expected_count):
    if len(starts) != len(lengths):
        raise ValueError("range metadata length mismatch")
    if np.any(lengths <= 0):
        raise ValueError("range lengths must be positive")
    total = int(np.sum(lengths, dtype=np.int64))
    if total != expected_count:
        raise ValueError("range lengths do not match index count")
    indices = np.empty(total, dtype=np.int32)
    offset = 0
    for start, length in zip(starts, lengths):
        length = int(length)
        indices[offset:offset + length] = np.arange(int(start), int(start) + length, dtype=np.int32)
        offset += length
    return indices


def decode_and_reconstruct(payload):
    kind, index_count, range_count, value_count, data = read_packet(payload)
    offset = 0

    if kind == 1:
        if range_count != 0 or index_count != value_count:
            raise ValueError("invalid absolute packet counts")
        indices, offset = take_array(data, offset, index_count, "<i4")
        values, offset = take_array(data, offset, value_count, "<f4")
        return indices, values

    if kind == 2:
        if range_count != 0 or index_count != value_count:
            raise ValueError("invalid index-delta packet counts")
        gaps, offset = take_array(data, offset, index_count, "<i4")
        indices = np.cumsum(gaps, dtype=np.int32) if index_count else np.array([], dtype=np.int32)
        values, offset = take_array(data, offset, value_count, "<f4")
        return indices, values

    if kind == 3:
        starts, offset = take_array(data, offset, range_count, "<i4")
        lengths, offset = take_array(data, offset, range_count, "<i4")
        indices = ranges_to_indices(starts, lengths, index_count)
        values, offset = take_array(data, offset, value_count, "<f4")
        return indices, values

    if kind == 4:
        if range_count != 0 or index_count != value_count:
            raise ValueError("invalid factored-value packet counts")
        if offset + 4 > len(data):
            raise ValueError("factored-value packet is missing its base")
        (base,) = struct.unpack("<f", data[offset:offset + 4])
        offset += 4
        indices, offset = take_array(data, offset, index_count, "<i4")
        residuals, offset = take_array(data, offset, value_count, "<f4")
        return indices, (base + residuals).astype(np.float32)

    if kind == 5:
        if range_count < 0 or index_count != value_count:
            raise ValueError("invalid combined packet counts")
        if offset + 4 > len(data):
            raise ValueError("combined packet is missing its base")
        (base,) = struct.unpack("<f", data[offset:offset + 4])
        offset += 4
        starts, offset = take_array(data, offset, range_count, "<i4")
        lengths, offset = take_array(data, offset, range_count, "<i4")
        indices = ranges_to_indices(starts, lengths, index_count)
        residuals, offset = take_array(data, offset, value_count, "<f4")
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

    if not all(result["exact"] for result in results.values()):
        raise RuntimeError("a delta representation failed exact reconstruction")
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
