#!/usr/bin/env python3
"""First runnable Neural Memory Network node.

This is intentionally not a neural network yet. It establishes the smallest
usable unit: a persistent local state that can produce and consume compact
state deltas.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass
class MemoryNode:
    """A persistent local memory with a synchronizable state."""

    name: str
    state: dict[str, float]

    def delta_from(self, reference: dict[str, float]) -> dict[str, float]:
        """Return only state values that differ from the reference."""
        keys = set(reference) | set(self.state)
        return {
            key: self.state.get(key, 0.0) - reference.get(key, 0.0)
            for key in sorted(keys)
            if self.state.get(key, 0.0) != reference.get(key, 0.0)
        }

    def apply_delta(self, delta: dict[str, float]) -> None:
        """Apply an additive state delta received from a synchronized peer."""
        for key, change in delta.items():
            self.state[key] = self.state.get(key, 0.0) + change

    def snapshot(self) -> dict[str, float]:
        return dict(self.state)


def main() -> None:
    # Both nodes begin from exactly the same reference state.
    reference = {"memory_a": 1.0, "memory_b": 2.0, "memory_c": 3.0}
    sender = MemoryNode("sender", reference.copy())
    receiver = MemoryNode("receiver", reference.copy())

    # Only the sender changes.
    sender.state["memory_b"] += 0.75
    sender.state["memory_c"] -= 0.25

    delta = sender.delta_from(reference)
    receiver.apply_delta(delta)

    result = {
        "sender": sender.snapshot(),
        "receiver": receiver.snapshot(),
        "delta": delta,
        "synchronized": sender.snapshot() == receiver.snapshot(),
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    assert result["synchronized"]


if __name__ == "__main__":
    main()
