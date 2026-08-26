
#!/usr/bin/env python3
"""
E36 — Local Observer / Adaptive Information Memory

Concept:
    The observer moves through a 3-D world over time.

    At each timestep it receives ONLY information that is currently
    visible from its position/orientation. Hidden geometry is never
    passed to the memory system.

    Memory is allowed to grow dynamically:
        - update an existing chunk
        - create a new chunk
        - merge compatible chunks

    The experiment asks whether a useful information topology can
    emerge from traversal rather than being imposed by giving the
    observer the whole world.

This is deliberately a small conceptual prototype before integrating
the mechanism into the existing latent experiments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# World
# ============================================================

@dataclass
class Voxel:
    pos: np.ndarray
    feature: np.ndarray


@dataclass
class World:
    voxels: list[Voxel]


def make_world(seed: int = 36) -> World:
    rng = np.random.default_rng(seed)

    voxels = []

    # Three spatial structures with partially overlapping feature
    # statistics. They are deliberately not exposed as object IDs.
    structures = [
        (np.array([0.0, 0.0, 0.0]), 2.5, 90),
        (np.array([7.0, 1.5, 2.0]), 2.0, 70),
        (np.array([2.0, 7.0, 3.5]), 2.2, 80),
        (np.array([8.0, 7.0, 1.0]), 1.6, 60),
    ]

    for center, radius, count in structures:
        for _ in range(count):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            r = radius * (rng.random() ** (1.0 / 3.0))
            pos = center + d * r

            # Feature is local appearance, not an object label.
            feature = np.array([
                math.sin(pos[0] * 0.7),
                math.cos(pos[1] * 0.6),
                math.sin(pos[2] * 0.9),
                r / 3.0,
            ])

            feature += rng.normal(0, 0.035, size=4)
            voxels.append(Voxel(pos.astype(float), feature.astype(float)))

    return World(voxels)


# ============================================================
# Observer / visibility
# ============================================================

@dataclass
class Observer:
    position: np.ndarray
    direction: np.ndarray
    fov_degrees: float = 75.0
    max_range: float = 7.0


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def line_of_sight(origin, target, world: World, ignore_index: int):
    """
    Approximate occlusion with spherical blockers around world points.

    The observer receives no information about a target if another
    world point lies sufficiently close to the ray before it.
    """
    ray = target - origin
    distance = np.linalg.norm(ray)
    if distance < 1e-9:
        return True

    ray = ray / distance

    for i, voxel in enumerate(world.voxels):
        if i == ignore_index:
            continue

        rel = voxel.pos - origin
        along = np.dot(rel, ray)

        if along <= 0.0 or along >= distance:
            continue

        closest = origin + ray * along
        lateral = np.linalg.norm(voxel.pos - closest)

        # Larger points block more strongly at close range.
        blocker_radius = 0.20 + 0.015 * along

        if lateral < blocker_radius:
            return False

    return True


def observe(observer: Observer, world: World):
    visible = []

    forward = normalize(observer.direction)
    cos_half_fov = math.cos(math.radians(observer.fov_degrees / 2))

    for i, voxel in enumerate(world.voxels):
        rel = voxel.pos - observer.position
        distance = np.linalg.norm(rel)

        if distance < 1e-9 or distance > observer.max_range:
            continue

        direction = rel / distance
        if np.dot(direction, forward) < cos_half_fov:
            continue

        if line_of_sight(observer.position, voxel.pos, world, i):
            visible.append({
                "world_index": i,
                "relative_position": rel.copy(),
                "feature": voxel.feature.copy(),
                "distance": float(distance),
            })

    return visible


# ============================================================
# Adaptive memory
# ============================================================

@dataclass
class Chunk:
    centroid: np.ndarray
    feature_centroid: np.ndarray
    observations: list[dict] = field(default_factory=list)
    last_seen: int = -1


class AdaptiveMemory:
    """
    No fixed spatial buckets.

    A new observation is compared with existing memory using only
    information the observer has already acquired.

    If it is sufficiently compatible with a remembered region,
    that chunk expands. Otherwise a new chunk is created.

    Periodically, compatible chunks may merge.

    The thresholds are deliberately simple and exposed so the next
    experiment can replace them with a learned decision mechanism.
    """

    def __init__(
        self,
        position_gate=1.35,
        feature_gate=0.65,
        merge_gate=1.10,
    ):
        self.chunks: list[Chunk] = []
        self.position_gate = position_gate
        self.feature_gate = feature_gate
        self.merge_gate = merge_gate

        self.events = []

    def _score(self, obs, chunk):
        p = np.linalg.norm(
            obs["relative_position"] - chunk.centroid
        )
        f = np.linalg.norm(
            obs["feature"] - chunk.feature_centroid
        )
        return p, f

    def update(self, observations, timestep, observer_position):
        created = 0
        updated = 0

        for obs in observations:
            # Convert relative observation into the observer's accumulated
            # world-relative coordinate frame. The observer is allowed to
            # know where it currently is and where it has been.
            absolute_estimate = (
                observer_position + obs["relative_position"]
            )

            candidate = None
            best_score = float("inf")

            for chunk in self.chunks:
                p = np.linalg.norm(
                    absolute_estimate - chunk.centroid
                )
                f = np.linalg.norm(
                    obs["feature"] - chunk.feature_centroid
                )

                if p <= self.position_gate and f <= self.feature_gate:
                    score = p + f
                    if score < best_score:
                        best_score = score
                        candidate = chunk

            record = {
                "position": absolute_estimate.copy(),
                "feature": obs["feature"].copy(),
                "timestep": timestep,
            }

            if candidate is None:
                self.chunks.append(
                    Chunk(
                        centroid=absolute_estimate.copy(),
                        feature_centroid=obs["feature"].copy(),
                        observations=[record],
                        last_seen=timestep,
                    )
                )
                created += 1
            else:
                n = len(candidate.observations)
                candidate.centroid = (
                    candidate.centroid * n + absolute_estimate
                ) / (n + 1)

                candidate.feature_centroid = (
                    candidate.feature_centroid * n
                    + obs["feature"]
                ) / (n + 1)

                candidate.observations.append(record)
                candidate.last_seen = timestep
                updated += 1

        merged = self.merge_compatible(timestep)

        self.events.append({
            "timestep": timestep,
            "visible": len(observations),
            "chunks": len(self.chunks),
            "created": created,
            "updated": updated,
            "merged": merged,
        })

    def merge_compatible(self, timestep):
        merged = 0
        changed = True

        while changed:
            changed = False

            for i in range(len(self.chunks)):
                if changed:
                    break

                for j in range(i + 1, len(self.chunks)):
                    a = self.chunks[i]
                    b = self.chunks[j]

                    p = np.linalg.norm(a.centroid - b.centroid)
                    f = np.linalg.norm(
                        a.feature_centroid - b.feature_centroid
                    )

                    if p <= self.merge_gate and f <= self.feature_gate:
                        all_obs = a.observations + b.observations
                        n = len(all_obs)

                        positions = np.asarray(
                            [x["position"] for x in all_obs]
                        )
                        features = np.asarray(
                            [x["feature"] for x in all_obs]
                        )

                        a.centroid = positions.mean(axis=0)
                        a.feature_centroid = features.mean(axis=0)
                        a.observations = all_obs
                        a.last_seen = max(
                            a.last_seen,
                            b.last_seen,
                        )

                        del self.chunks[j]
                        merged += 1
                        changed = True
                        break

        return merged

    def memory_size(self):
        # Structural cost + stored observations.
        return sum(
            8 + 4 * len(c.observations)
            for c in self.chunks
        )


# ============================================================
# Trajectory
# ============================================================

def make_trajectory(steps=96):
    points = []

    for t in range(steps):
        a = 2 * math.pi * t / steps

        # A looping path that passes through regions from different
        # directions, then revisits the starting area.
        x = 4.0 + 4.4 * math.cos(a)
        y = 3.8 + 3.2 * math.sin(a)
        z = 2.0 + 0.8 * math.sin(2 * a)

        points.append(np.array([x, y, z]))

    return points


def direction_for_path(path, i):
    if i == 0:
        return path[1] - path[0]
    if i == len(path) - 1:
        return path[-1] - path[-2]
    return path[i + 1] - path[i - 1]


# ============================================================
# Experiment
# ============================================================

def run():
    print("=" * 80)
    print("E36 — LOCAL OBSERVER / ADAPTIVE INFORMATION MEMORY")
    print("=" * 80)

    world = make_world()
    path = make_trajectory()

    memory = AdaptiveMemory()

    rows = []

    total_world_points = len(world.voxels)
    ever_seen = set()

    for t, position in enumerate(path):
        direction = direction_for_path(path, t)

        observer = Observer(
            position=position,
            direction=direction,
        )

        visible = observe(observer, world)

        for item in visible:
            ever_seen.add(item["world_index"])

        memory.update(
            visible,
            t,
            position,
        )

        rows.append({
            "timestep": t,
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "visible": len(visible),
            "unique_seen": len(ever_seen),
            "chunks": len(memory.chunks),
            "memory_size": memory.memory_size(),
        })

        if t % 12 == 0 or t == len(path) - 1:
            print(
                f"t={t:3d} | "
                f"visible={len(visible):3d} | "
                f"unique_seen={len(ever_seen):3d}/{total_world_points} | "
                f"chunks={len(memory.chunks):3d} | "
                f"memory={memory.memory_size():4d}"
            )

    # --------------------------------------------------------
    # Revisit analysis
    # --------------------------------------------------------

    midpoint = len(path) // 2
    first_half = rows[:midpoint]
    second_half = rows[midpoint:]

    first_unique = first_half[-1]["unique_seen"]
    final_unique = rows[-1]["unique_seen"]

    first_chunks = first_half[-1]["chunks"]
    final_chunks = rows[-1]["chunks"]

    revisit_new_information = final_unique - first_unique
    revisit_chunk_growth = final_chunks - first_chunks

    # A crude locality check:
    # the observer never receives hidden points.
    observed_indices = sorted(ever_seen)

    results = {
        "experiment": "E36 — local observer adaptive information memory",
        "world_points": total_world_points,
        "timesteps": len(path),
        "final_unique_seen": final_unique,
        "coverage_fraction": final_unique / total_world_points,
        "final_chunks": final_chunks,
        "final_memory_size": memory.memory_size(),
        "first_half_chunks": first_chunks,
        "second_half_new_information": revisit_new_information,
        "second_half_chunk_growth": revisit_chunk_growth,
        "locality_enforced": True,
        "observed_world_indices": observed_indices,
        "trajectory": rows,
        "memory_events": memory.events,
    }

    out = Path("synthetic_latent_e36_local_observer")
    out.mkdir(exist_ok=True)

    (out / "e36_summary.json").write_text(
        json.dumps(results, indent=2)
    )

    # --------------------------------------------------------
    # Plot observer trajectory and world
    # --------------------------------------------------------

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    world_xyz = np.asarray([v.pos for v in world.voxels])
    path_xyz = np.asarray(path)

    ax.scatter(
        world_xyz[:, 0],
        world_xyz[:, 1],
        world_xyz[:, 2],
        s=8,
        alpha=0.18,
    )

    ax.plot(
        path_xyz[:, 0],
        path_xyz[:, 1],
        path_xyz[:, 2],
        linewidth=2,
    )

    ax.scatter(
        [path_xyz[0, 0]],
        [path_xyz[0, 1]],
        [path_xyz[0, 2]],
        s=60,
        marker="o",
    )

    ax.set_title("E36 — World and observer trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig(out / "e36_world_trajectory.png", dpi=160)
    plt.close()

    # --------------------------------------------------------
    # Plot information growth
    # --------------------------------------------------------

    t = [r["timestep"] for r in rows]
    unique = [r["unique_seen"] for r in rows]
    chunks = [r["chunks"] for r in rows]
    mem = [r["memory_size"] for r in rows]

    plt.figure(figsize=(9, 6))
    plt.plot(t, unique, label="unique information acquired")
    plt.plot(t, chunks, label="adaptive chunks")
    plt.plot(t, mem, label="memory cost")
    plt.xlabel("time")
    plt.ylabel("count / cost")
    plt.title("E36 — Information acquisition vs adaptive memory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_information_growth.png", dpi=160)
    plt.close()

    print()
    print("=" * 80)
    print("E36 COMPLETE")
    print("=" * 80)
    print(f"World points:       {total_world_points}")
    print(f"Unique seen:        {final_unique}/{total_world_points}")
    print(f"Coverage:           {final_unique / total_world_points:.3f}")
    print(f"Final chunks:       {final_chunks}")
    print(f"Final memory cost:  {memory.memory_size()}")
    print(f"New info on revisit:{revisit_new_information}")
    print(f"Chunk growth:       {revisit_chunk_growth}")
    print(f"Results:            {out.resolve()}")


if __name__ == "__main__":
    run()
