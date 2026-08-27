#!/usr/bin/env python3
"""Minimal persistent relational memory prototype.

The observer never receives object IDs or absolute world coordinates.
It sees local features and relative positions, builds persistent units,
records sparse transitions between units, and predicts the next observation.

This is intentionally a small, inspectable baseline rather than a neural model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import json
from pathlib import Path
import numpy as np

SEED = 314
FEATURE_GATE = 0.18
RELATION_GATE = 2.0
RELATIONS_PER_STEP = 2

@dataclass
class Unit:
    uid: int
    feature: np.ndarray
    count: int = 1

@dataclass
class Relation:
    src: int
    dst: int
    delta_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    count: int = 0

    @property
    def delta(self):
        return self.delta_sum / max(self.count, 1)

class Memory:
    def __init__(self):
        self.units: dict[int, Unit] = {}
        self.relations: dict[tuple[int, int], Relation] = {}
        self.next_uid = 0
        self.previous: list[tuple[int, np.ndarray]] = []

    def identify(self, feature: np.ndarray) -> tuple[int, bool]:
        if not self.units:
            return self._new(feature), True
        best = min(self.units.values(), key=lambda u: np.linalg.norm(u.feature-feature))
        if np.linalg.norm(best.feature-feature) > FEATURE_GATE:
            return self._new(feature), True
        best.feature = (best.feature * best.count + feature) / (best.count + 1)
        best.count += 1
        return best.uid, False

    def _new(self, feature):
        uid = self.next_uid
        self.next_uid += 1
        self.units[uid] = Unit(uid, feature.copy())
        return uid

    def observe(self, observations):
        current = []
        for feature, position in observations:
            uid, _ = self.identify(feature)
            current.append((uid, position.copy()))

        candidates = []
        for src, src_pos in self.previous:
            for dst, dst_pos in current:
                delta = dst_pos - src_pos
                d = float(np.linalg.norm(delta))
                if d <= RELATION_GATE:
                    candidates.append((d, src, dst, delta))
        candidates.sort(key=lambda x: x[0])
        used = set()
        for _, src, dst, delta in candidates:
            if dst in used:
                continue
            key = (src, dst)
            rel = self.relations.setdefault(key, Relation(src, dst))
            rel.delta_sum += delta
            rel.count += 1
            used.add(dst)
            if len(used) >= RELATIONS_PER_STEP:
                break
        self.previous = current

    def predict_next(self):
        predictions = []
        for rel in self.relations.values():
            if rel.count:
                predictions.append((rel.src, rel.dst, rel.delta.copy()))
        return predictions

    def snapshot(self):
        return {
            "units": len(self.units),
            "relations": len(self.relations),
            "relation_table": [
                {"src": r.src, "dst": r.dst, "delta": r.delta.tolist(), "count": r.count}
                for r in self.relations.values()
            ],
        }

def world(t, phase=0.0):
    """Three persistent objects moving on a circle; IDs stay evaluator-only."""
    base = [0.0, 2.1, 4.2]
    out = []
    for i, offset in enumerate(base):
        a = phase + offset + 0.12*t
        pos = np.array([math.cos(a), math.sin(a)])
        # Intrinsic feature identifies the local object, not its location.
        feature = np.array([math.cos(offset), math.sin(offset)])
        out.append((feature, pos))
    return out

def run():
    np.random.seed(SEED)
    mem = Memory()

    # Pass A builds the persistent structure.
    for t in range(30):
        mem.observe(world(t, phase=0.0))

    before = mem.snapshot()
    predictions = mem.predict_next()

    # Pass B changes observation phase. No world IDs or absolute positions enter memory.
    mem.previous = []
    for t in range(30):
        mem.observe(world(t, phase=math.pi/3))

    after = mem.snapshot()
    reused = sum(u.count > 30 for u in mem.units.values())
    result = {
        "experiment": "minimal persistent relational memory",
        "seed": SEED,
        "pass_a": before,
        "pass_b": after,
        "units_reused_across_passes": reused,
        "initial_predictions": len(predictions),
    }
    Path("persistent_relational_memory_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run()
