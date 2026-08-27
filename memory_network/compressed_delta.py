#!/usr/bin/env python3
"""Measure compression of sparse state deltas.

Two synchronized nodes start from the same state. One node changes a controlled
number of values. The delta is serialized, compressed with zlib, decompressed,
and applied to the receiver. We compare full-state bytes, raw-delta bytes, and
compressed-delta bytes while requiring exact synchronization.
"""

from dataclasses import dataclass
import json
import struct
import zlib
from pathlib import Path

import numpy as np

SEED = 314
STATE_SIZE = 1024


@dataclass
class Delta:
    indices: np.ndarray
    values: np.ndarray

    def encode(self) -> bytes:
        count = len(self.indices)
        header = struct.pack("<I", count)
        return header + self.indices.astype("<i4", copy=False).tobytes() + self.values.astype("<f4", copy=False).tobytes()

    @staticmethod
    def decode(payload: bytes) -> "Delta":
        (count,) = struct.unpack("<I", payload[:4])
        offset = 4
        index_bytes = 4 * count
        indices = np.frombuffer(payload[offset:offset + index_bytes], dtype="<i4").copy()
        offset += index_bytes
        values = np.frombuffer(payload[offset:offset + 4 * count], dtype="<f4").copy()
        return Delta(indices, values)


class MemoryNode:
    def __init__(self, state: np.ndarray):
        self.reference = state.astype(np.float32, copy=True)
        self.state = self.reference.copy()

    def change(self, indices: np.ndarray, values: np.ndarray) -> None:
        self.state[indices] += values

    def make_delta(self) -> Delta:
        difference = self.state - self.reference
        indices = np.flatnonzero(difference != 0).astype(np.int32)
        return Delta(indices, difference[indices].astype(np.float32))

    def apply_delta(self, delta: Delta) -> None:
        self.state[delta.indices] = self.reference[delta.indices] + delta.values


def run_case(rng: np.random.Generator, changed: int, pattern: str) -> dict:
    initial = rng.normal(size=STATE_SIZE).astype(np.float32)
    source = MemoryNode(initial)
    receiver = MemoryNode(initial)

    indices = np.sort(rng.choice(STATE_SIZE, size=changed, replace=False)).astype(np.int32)
    if pattern == "random":
        values = rng.normal(scale=0.01, size=changed).astype(np.float32)
    elif pattern == "similar":
        values = (0.01 + rng.normal(scale=0.0001, size=changed)).astype(np.float32)
    elif pattern == "clustered":
        start = int(rng.integers(0, STATE_SIZE - changed + 1))
        indices = np.arange(start, start + changed, dtype=np.int32)
        values = rng.normal(scale=0.01, size=changed).astype(np.float32)
    else:
        raise ValueError(pattern)

    source.change(indices, values)
    delta = source.make_delta()
    raw = delta.encode()
    compressed = zlib.compress(raw, level=9)
    decoded = Delta.decode(zlib.decompress(compressed))
    receiver.apply_delta(decoded)

    full_bytes = int(source.state.nbytes)
    raw_bytes = len(raw)
    compressed_bytes = len(compressed)
    return {
        "pattern": pattern,
        "changed_values": changed,
        "state_values": STATE_SIZE,
        "full_state_bytes": full_bytes,
        "raw_delta_bytes": raw_bytes,
        "compressed_delta_bytes": compressed_bytes,
        "raw_reduction_percent": 100.0 * (1.0 - raw_bytes / full_bytes),
        "compressed_reduction_percent": 100.0 * (1.0 - compressed_bytes / full_bytes),
        "compression_of_delta_percent": 100.0 * (1.0 - compressed_bytes / raw_bytes),
        "synchronized": bool(np.array_equal(source.state, receiver.state)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = []
    for pattern in ("random", "similar", "clustered"):
        for changed in (1, 4, 16, 64, 256, 512, 1024):
            results.append(run_case(rng, changed, pattern))

    output = {"experiment": "compressed delta communication", "seed": SEED, "results": results}
    Path("compressed_delta_result.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
