
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
class KnowledgeChunk:
    """
    Adaptive region of knowledge.

    A chunk is a compact model of a region the observer has encountered.
    It is NOT an object identity and it is NOT an observation history.

    The chunk stores sufficient statistics:
        centroid
        feature centroid
        spatial spread
        feature spread
        observation count
        compact lineage / split metadata

    If new information no longer fits the current region, the chunk is
    allowed to expand the representation by splitting into smaller chunks.
    """
    centroid: np.ndarray
    feature_centroid: np.ndarray
    position_m2: float = 0.0
    feature_m2: float = 0.0
    observations_seen: int = 0
    first_seen: int = -1
    last_seen: int = -1
    split_count: int = 0
    parent_id: int | None = None
    chunk_id: int = -1


class AdaptiveKnowledgeMemory:
    """
    Adaptive local knowledge topology.

    Core rule:

        If the current knowledge can explain an observation,
            update that knowledge.

        If it cannot,
            expand the representation.

    Expansion happens by creating a new local chunk. Later, nearby chunks
    can merge when the accumulated evidence says they represent one
    coherent region.

    There is deliberately NO temporal-change bookkeeping in the static
    world. Repeatedly seeing the same static region is confirmation, not
    new data.

    This version therefore tests the user's original idea more directly:
    the representation decides how much structure it needs based on the
    stream of locally available information.
    """

    def __init__(
        self,
        position_gate=1.35,
        feature_gate=0.65,
        spread_gate=0.95,
        merge_gate=0.90,
        feature_merge_gate=0.50,
    ):
        self.chunks: list[KnowledgeChunk] = []

        self.position_gate = position_gate
        self.feature_gate = feature_gate
        self.spread_gate = spread_gate
        self.merge_gate = merge_gate
        self.feature_merge_gate = feature_merge_gate

        self.next_id = 0
        self.events = []

    def _new_chunk(self, position, feature, timestep):
        chunk = KnowledgeChunk(
            centroid=position.copy(),
            feature_centroid=feature.copy(),
            position_m2=0.0,
            feature_m2=0.0,
            observations_seen=1,
            first_seen=timestep,
            last_seen=timestep,
            split_count=0,
            parent_id=None,
            chunk_id=self.next_id,
        )
        self.next_id += 1
        self.chunks.append(chunk)
        return chunk

    def _candidate(self, position, feature):
        best = None
        best_score = float("inf")

        for chunk in self.chunks:
            p = np.linalg.norm(position - chunk.centroid)
            f = np.linalg.norm(feature - chunk.feature_centroid)

            if p <= self.position_gate and f <= self.feature_gate:
                # Prefer chunks whose current compact state is closest.
                score = p + f
                if score < best_score:
                    best_score = score
                    best = chunk

        return best

    def _update_chunk(self, chunk, position, feature, timestep):
        """
        Incrementally update sufficient statistics.

        We do NOT retain the raw observation.
        """
        n_old = chunk.observations_seen

        dp = position - chunk.centroid
        df = feature - chunk.feature_centroid

        n_new = n_old + 1

        # Welford-style running second moment.
        chunk.position_m2 += (
            np.dot(dp, dp) * n_old / n_new
        )
        chunk.feature_m2 += (
            np.dot(df, df) * n_old / n_new
        )

        chunk.centroid = (
            chunk.centroid
            + dp / n_new
        )

        chunk.feature_centroid = (
            chunk.feature_centroid
            + df / n_new
        )

        chunk.observations_seen = n_new
        chunk.last_seen = timestep

    def update(self, observations, timestep, observer_position):
        created = 0
        updated = 0
        expanded = 0
        merged = 0

        for obs in observations:
            absolute_estimate = (
                observer_position
                + obs["relative_position"]
            )

            candidate = self._candidate(
                absolute_estimate,
                obs["feature"],
            )

            if candidate is None:
                self._new_chunk(
                    absolute_estimate,
                    obs["feature"],
                    timestep,
                )
                created += 1
                continue

            # Determine whether this observation still fits inside the
            # candidate's compact knowledge region.
            p = np.linalg.norm(
                absolute_estimate - candidate.centroid
            )
            f = np.linalg.norm(
                obs["feature"] - candidate.feature_centroid
            )

            if (
                p > self.spread_gate
                or f > self.feature_gate
            ):
                # The current chunk cannot explain this observation.
                # Expand the representation by creating a new local region.
                child = self._new_chunk(
                    absolute_estimate,
                    obs["feature"],
                    timestep,
                )
                child.parent_id = candidate.chunk_id
                candidate.split_count += 1
                expanded += 1
            else:
                self._update_chunk(
                    candidate,
                    absolute_estimate,
                    obs["feature"],
                    timestep,
                )
                updated += 1

        merged = self.merge_compatible(timestep)

        self.events.append({
            "timestep": timestep,
            "visible": len(observations),
            "chunks": len(self.chunks),
            "created": created,
            "updated": updated,
            "expanded": expanded,
            "merged": merged,
        })

    def merge_compatible(self, timestep):
        merged = 0
        did_merge = True

        while did_merge:
            did_merge = False

            for i in range(len(self.chunks)):
                if did_merge:
                    break

                for j in range(i + 1, len(self.chunks)):
                    a = self.chunks[i]
                    b = self.chunks[j]

                    p = np.linalg.norm(
                        a.centroid - b.centroid
                    )
                    f = np.linalg.norm(
                        a.feature_centroid
                        - b.feature_centroid
                    )

                    if (
                        p <= self.merge_gate
                        and f <= self.feature_merge_gate
                    ):
                        total = (
                            a.observations_seen
                            + b.observations_seen
                        )

                        if total:
                            a.centroid = (
                                a.centroid * a.observations_seen
                                + b.centroid * b.observations_seen
                            ) / total

                            a.feature_centroid = (
                                a.feature_centroid
                                * a.observations_seen
                                + b.feature_centroid
                                * b.observations_seen
                            ) / total

                        a.position_m2 += b.position_m2
                        a.feature_m2 += b.feature_m2
                        a.observations_seen = total
                        a.first_seen = min(
                            a.first_seen,
                            b.first_seen,
                        )
                        a.last_seen = max(
                            a.last_seen,
                            b.last_seen,
                        )
                        a.split_count += b.split_count

                        del self.chunks[j]
                        merged += 1
                        did_merge = True
                        break

        return merged

    def knowledge_cost_slots(self):
        """
        Cost of persistent knowledge, not perception history.

        Each chunk:
            position centroid       3
            feature centroid        4
            spatial spread          1
            feature spread          1
            observation count       1
            timestamps              2
            ---------------------------
                               =   12

        A chunk's 100th confirmation costs zero additional storage.

        This is intentionally a simple structural cost, not a claim about
        a particular binary serialization format.
        """
        return len(self.chunks) * 12

    @staticmethod
    def raw_observation_cost_slots(observations):
        # Every raw observation carries 3 position + 4 feature values.
        return observations * 7


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
    print("E36 — LOCAL OBSERVER / ADAPTIVE KNOWLEDGE TOPOLOGY v3")
    print("=" * 80)

    world = make_world()
    path = make_trajectory()

    memory = AdaptiveKnowledgeMemory()

    rows = []

    total_world_points = len(world.voxels)
    ever_seen = set()
    total_raw_observations = 0

    for t, position in enumerate(path):
        direction = direction_for_path(path, t)

        observer = Observer(
            position=position,
            direction=direction,
        )

        visible = observe(observer, world)

        for item in visible:
            ever_seen.add(item["world_index"])

        total_raw_observations += len(visible)

        memory.update(
            visible,
            t,
            position,
        )

        knowledge_cost = memory.knowledge_cost_slots()
        raw_cost = memory.raw_observation_cost_slots(
            total_raw_observations
        )

        rows.append({
            "timestep": t,
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "visible": len(visible),
            "unique_seen": len(ever_seen),
            "knowledge_chunks": len(memory.chunks),
            "knowledge_cost_slots": knowledge_cost,
            "raw_observation_cost_slots": raw_cost,
            "compression_ratio": (
                knowledge_cost / raw_cost
                if raw_cost else 0.0
            ),
            "created": memory.events[-1]["created"],
            "updated": memory.events[-1]["updated"],
            "expanded": memory.events[-1]["expanded"],
            "merged": memory.events[-1]["merged"],
        })

        if t % 12 == 0 or t == len(path) - 1:
            print(
                f"t={t:3d} | "
                f"visible={len(visible):3d} | "
                f"unique={len(ever_seen):3d}/{total_world_points} | "
                f"chunks={len(memory.chunks):3d} | "
                f"expand={memory.events[-1]['expanded']:2d} | "
                f"merge={memory.events[-1]['merged']:2d} | "
                f"cost={knowledge_cost:4d} | "
                f"raw={raw_cost:4d}"
            )

    midpoint = len(path) // 2
    first_half = rows[:midpoint]

    first_unique = first_half[-1]["unique_seen"]
    final_unique = rows[-1]["unique_seen"]

    first_chunks = first_half[-1]["knowledge_chunks"]
    final_chunks = rows[-1]["knowledge_chunks"]

    revisit_new_information = final_unique - first_unique
    revisit_chunk_growth = final_chunks - first_chunks

    final_knowledge_cost = memory.knowledge_cost_slots()
    final_raw_cost = memory.raw_observation_cost_slots(
        total_raw_observations
    )

    results = {
        "experiment": (
            "E36 — local observer adaptive knowledge topology v3"
        ),
        "world_points": total_world_points,
        "timesteps": len(path),
        "final_unique_seen": final_unique,
        "coverage_fraction": final_unique / total_world_points,
        "final_knowledge_chunks": final_chunks,
        "total_raw_observations": total_raw_observations,
        "final_knowledge_cost_slots": final_knowledge_cost,
        "counterfactual_raw_observation_cost_slots": final_raw_cost,
        "knowledge_to_raw_ratio": (
            final_knowledge_cost / final_raw_cost
            if final_raw_cost else 0.0
        ),
        "estimated_slot_savings": (
            final_raw_cost - final_knowledge_cost
        ),
        "first_half_chunks": first_chunks,
        "second_half_new_information": revisit_new_information,
        "second_half_chunk_growth": revisit_chunk_growth,
        "locality_enforced": True,
        "hidden_world_is_never_given_to_memory": True,
        "repeated_static_observations_do_not_create_history": True,
        "adaptive_expansion_enabled": True,
        "trajectory": rows,
        "memory_events": memory.events,
    }

    out = Path("synthetic_latent_e36_local_topology_v3")
    out.mkdir(exist_ok=True)

    (out / "e36_summary.json").write_text(
        json.dumps(results, indent=2)
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    t = [r["timestep"] for r in rows]
    unique = [r["unique_seen"] for r in rows]
    chunks = [r["knowledge_chunks"] for r in rows]
    raw = [r["raw_observation_cost_slots"] for r in rows]
    knowledge = [r["knowledge_cost_slots"] for r in rows]
    expansions = [r["expanded"] for r in rows]
    ratio = [r["compression_ratio"] for r in rows]

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

    plt.figure(figsize=(9, 6))
    plt.plot(t, unique, label="unique information acquired")
    plt.plot(t, chunks, label="adaptive knowledge chunks")
    plt.xlabel("time")
    plt.ylabel("count")
    plt.title("E36 — Information acquisition vs adaptive topology")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_information_vs_topology.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(t, raw, label="raw observation storage")
    plt.plot(t, knowledge, label="persistent knowledge storage")
    plt.xlabel("time")
    plt.ylabel("scalar slots")
    plt.title("E36 — Perception history vs persistent knowledge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_memory_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(t, expansions, label="new expansions this step")
    plt.xlabel("time")
    plt.ylabel("new chunks")
    plt.title("E36 — When the representation decides it needs more structure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_adaptive_expansion.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(t, ratio)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("time")
    plt.ylabel("knowledge / raw observation cost")
    plt.title("E36 — Persistent knowledge compression ratio")
    plt.tight_layout()
    plt.savefig(out / "e36_compression_ratio.png", dpi=160)
    plt.close()

    print()
    print("=" * 80)
    print("E36 COMPLETE")
    print("=" * 80)
    print(f"World points:          {total_world_points}")
    print(f"Unique seen:           {final_unique}/{total_world_points}")
    print(f"Coverage:              {final_unique / total_world_points:.3f}")
    print(f"Knowledge chunks:      {final_chunks}")
    print(f"Knowledge cost:        {final_knowledge_cost} scalar slots")
    print(f"Raw observation cost:  {final_raw_cost} scalar slots")
    print(
        f"Estimated savings:     "
        f"{final_raw_cost - final_knowledge_cost} slots"
    )
    print(
        f"Knowledge/raw ratio:   "
        f"{final_knowledge_cost / final_raw_cost:.3f}"
    )
    print(f"New info on revisit:   {revisit_new_information}")
    print(f"Chunk growth on revisit:{revisit_chunk_growth}")
    print(f"Results:               {out.resolve()}")



if __name__ == "__main__":
    run()
