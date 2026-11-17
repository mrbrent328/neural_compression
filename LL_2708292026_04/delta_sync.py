#!/usr/bin/env python3
"""Measure compact state deltas between synchronized memory nodes.

This is the next step after the initial synchronization prototype. Two nodes
begin with identical state, one node changes, and only the changed coordinates
are transmitted. The receiver applies the delta and we measure communication
cost versus sending the complete state.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

SEED = 314
STATE_SIZE = 1024


@dataclass
class Delta:
    indices: np.ndarray
    values: np.ndarray

    @property
    def bytes(self) -> int:
        return int(self.indices.nbytes + self.values.nbytes)


class MemoryNode:
    def __init__(self, state: np.ndarray):
        self.state = state.astype(np.float32, copy=True)
        self.reference = self.state.copy()

    def change(self, indices: np.ndarray, values: np.ndarray) -> None:
        self.state[indices] += values

    def make_delta(self, threshold: float = 0.0) -> Delta:
        difference = self.state - self.reference
        indices = np.flatnonzero(np.abs(difference) > threshold).astype(np.int32)
        values = difference[indices].astype(np.float32)
        return Delta(indices, values)

    def apply_delta(self, delta: Delta) -> None:
        self.state[delta.indices] = self.reference[delta.indices] + delta.values

    def synchronized_with(self, other: "MemoryNode") -> bool:
        return bool(np.array_equal(self.state, other.state))


def run_case(changed: int) -> dict:
    rng = np.random.default_rng(SEED)
    initial = rng.normal(size=STATE_SIZE).astype(np.float32)
    source = MemoryNode(initial)
    receiver = MemoryNode(initial)

    indices = np.arange(changed, dtype=np.int32)
    values = rng.normal(scale=0.01, size=changed).astype(np.float32)
    source.change(indices, values)

    delta = source.make_delta()
    receiver.apply_delta(delta)

    full_bytes = int(source.state.nbytes)
    delta_bytes = delta.bytes
    return {
        "changed_values": changed,
        "state_values": STATE_SIZE,
        "full_state_bytes": full_bytes,
        "delta_bytes": delta_bytes,
        "communication_ratio": delta_bytes / full_bytes,
        "reduction_percent": 100.0 * (1.0 - delta_bytes / full_bytes),
        "synchronized": source.synchronized_with(receiver),
    }


def main() -> None:
    results = [run_case(changed) for changed in (1, 4, 16, 64, 256, 512, 1024)]
    output = {
        "experiment": "delta communication cost",
        "seed": SEED,
        "results": results,
    }
    Path("delta_sync_result.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
