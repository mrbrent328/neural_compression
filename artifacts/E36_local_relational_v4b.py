#!/usr/bin/env python3
"""
E36 v4b — Local relational memory / double traversal (sparse)

Purpose
-------
Test whether a moving observer can build a reusable representation from
only locally available information and movement, without giving memory
absolute world coordinates or hidden world identifiers.

The simulator itself still uses world coordinates to perform visibility.
That is allowed. The information boundary is at the observer -> memory
interface.

Memory receives ONLY:
    - relative position of a visible feature to the observer
    - local feature vector
    - distance (derived locally from relative position)
    - observer movement delta
    - observer rotation delta
    - timestep as ordering, not as a spatial/world coordinate

Memory never receives:
    - world coordinates
    - world_index
    - the complete world
    - hidden geometry
    - position-derived feature vectors

Two identical traversals are run through the same world. Memory persists
between passes. The key measurement is whether pass 2 reuses the relational
structure built during pass 1 and therefore requires less expansion.

Relations are deliberately sparse: each observation connects only to
a small number of persistent local anchors and to its own previous
state. This prevents the representation from degenerating into an
all-pairs observation history.
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

    structures = [
        (np.array([0.0, 0.0, 0.0]), 2.5, 90),
        (np.array([7.0, 1.5, 2.0]), 2.0, 70),
        (np.array([2.0, 7.0, 3.5]), 2.2, 80),
        (np.array([8.0, 7.0, 1.0]), 1.6, 60),
    ]

    # IMPORTANT: feature is now an intrinsic local signature.
    # It is sampled independently of world position. The observer can
    # use it to recognize a previously encountered feature, but cannot
    # invert it to recover x/y/z.
    for center, radius, count in structures:
        for _ in range(count):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            r = radius * (rng.random() ** (1.0 / 3.0))
            pos = center + d * r

            # Stable intrinsic appearance/signature.
            feature = rng.normal(0.0, 1.0, size=4)
            feature /= np.linalg.norm(feature)
            feature += rng.normal(0.0, 0.012, size=4)

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


def rotation_delta(prev_direction, current_direction):
    if prev_direction is None:
        return np.zeros(3)

    a = normalize(prev_direction)
    b = normalize(current_direction)

    # A compact local rotation descriptor: cross product.
    # It carries relative orientation change, not world orientation.
    return np.cross(a, b)


def line_of_sight(origin, target, world: World, ignore_index: int):
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
            # Evaluator metadata is intentionally separate from what the
            # memory receives. This lets us measure coverage without leaking
            # an object/world ID into the representation.
            visible.append({
                "world_index": i,
                "relative_position": rel.copy(),
                "feature": voxel.feature.copy(),
                "distance": float(distance),
            })

    return visible


def memory_observations(visible):
    """Strip evaluator/world metadata before crossing the memory boundary."""
    return [
        {
            "relative_position": item["relative_position"].copy(),
            "feature": item["feature"].copy(),
            "distance": item["distance"],
        }
        for item in visible
    ]


# ============================================================
# Relational memory
# ============================================================

@dataclass
class Landmark:
    landmark_id: int
    feature: np.ndarray
    feature_m2: float = 0.0
    observations: int = 0
    first_seen: int = -1
    last_seen: int = -1
    passes_seen: set[int] = field(default_factory=set)


@dataclass
class Relation:
    from_landmark: int
    to_landmark: int
    # Relative motion accumulated between observations.
    motion_sum: np.ndarray
    motion_count: int
    # Relative landmark movement in observer coordinates.
    relative_shift_sum: np.ndarray
    relation_count: int
    first_pass: int
    last_pass: int


class RelationalMemory:
    """
    Local relational memory.

    A landmark is identified by intrinsic perceptual state, not position.
    Relations record how observations connect through the observer's
    movement. No absolute location is stored.
    """

    def __init__(self, feature_gate=0.22, new_landmark_gate=0.42):
        self.feature_gate = feature_gate
        self.new_landmark_gate = new_landmark_gate

        self.landmarks: list[Landmark] = []
        self.relations: dict[tuple[int, int], Relation] = {}
        self.next_landmark_id = 0

        self.previous_visible: list[tuple[int, np.ndarray]] = []
        self.previous_motion = np.zeros(3)
        self.previous_rotation = np.zeros(3)
        self.events = []

    def _match_landmark(self, feature):
        if not self.landmarks:
            return None, float("inf")

        best = None
        best_distance = float("inf")
        for landmark in self.landmarks:
            d = np.linalg.norm(feature - landmark.feature)
            if d < best_distance:
                best_distance = d
                best = landmark

        if best_distance <= self.feature_gate:
            return best, best_distance
        return None, best_distance

    def _create_landmark(self, feature, timestep, pass_id):
        landmark = Landmark(
            landmark_id=self.next_landmark_id,
            feature=feature.copy(),
            observations=1,
            first_seen=timestep,
            last_seen=timestep,
            passes_seen={pass_id},
        )
        self.next_landmark_id += 1
        self.landmarks.append(landmark)
        return landmark

    def _update_landmark(self, landmark, feature, timestep, pass_id):
        n = landmark.observations
        delta = feature - landmark.feature
        n_new = n + 1
        landmark.feature_m2 += np.dot(delta, delta) * n / n_new
        landmark.feature = landmark.feature + delta / n_new
        landmark.observations = n_new
        landmark.last_seen = timestep
        landmark.passes_seen.add(pass_id)

    def update(self, observations, timestep, pass_id, motion_delta, rotation_delta):
        matched = 0
        created = 0
        expanded = 0
        reused = 0
        relation_updates = 0

        current_visible = []

        for obs in observations:
            landmark, feature_distance = self._match_landmark(obs["feature"])

            if landmark is None:
                landmark = self._create_landmark(
                    obs["feature"], timestep, pass_id
                )
                created += 1
                expanded += 1
            else:
                was_seen_in_pass = pass_id in landmark.passes_seen
                self._update_landmark(
                    landmark,
                    obs["feature"],
                    timestep,
                    pass_id,
                )
                matched += 1
                if not was_seen_in_pass:
                    reused += 1

            current_visible.append(
                (landmark.landmark_id, obs["relative_position"].copy())
            )

        # Build a SPARSE relational topology.
        #
        # We do not store every co-visible pair. That would turn a local
        # state into a quadratic observation history. Instead:
        #   1. retain short temporal edges for landmarks that persist across
        #      consecutive viewpoints;
        #   2. retain at most a few nearest cross-landmark anchors per view.
        #
        # This is much closer to the user's "six degrees" intuition:
        # the system can trace a route through a small relational graph
        # instead of remembering every pair it ever saw.
        if self.previous_visible:
            previous_by_id = {lid: rel for lid, rel in self.previous_visible}
            for curr_id, curr_rel in current_visible:
                if curr_id not in previous_by_id:
                    continue

                prev_rel = previous_by_id[curr_id]
                key = (curr_id, curr_id)
                relation = self.relations.get(key)
                if relation is None:
                    relation = Relation(
                        from_landmark=curr_id,
                        to_landmark=curr_id,
                        motion_sum=np.zeros(3),
                        motion_count=0,
                        relative_shift_sum=np.zeros(3),
                        relation_count=0,
                        first_pass=pass_id,
                        last_pass=pass_id,
                    )
                    self.relations[key] = relation

                relation.motion_sum += motion_delta
                relation.motion_count += 1
                relation.relative_shift_sum += curr_rel - prev_rel
                relation.relation_count += 1
                relation.last_pass = pass_id
                relation_updates += 1

        # Sparse spatial anchors: for each current landmark, connect it to
        # up to two nearest OTHER landmarks visible at this viewpoint.
        # Relative positions only; no absolute world coordinate is stored.
        max_anchors = 2
        for i in range(len(current_visible)):
            a_id, a_rel = current_visible[i]
            candidates = []
            for j in range(len(current_visible)):
                if i == j:
                    continue
                b_id, b_rel = current_visible[j]
                d = np.linalg.norm(b_rel - a_rel)
                candidates.append((d, b_id, b_rel))

            candidates.sort(key=lambda x: x[0])
            for _, b_id, b_rel in candidates[:max_anchors]:
                key = (a_id, b_id)
                relation = self.relations.get(key)
                if relation is None:
                    relation = Relation(
                        from_landmark=a_id,
                        to_landmark=b_id,
                        motion_sum=np.zeros(3),
                        motion_count=0,
                        relative_shift_sum=np.zeros(3),
                        relation_count=0,
                        first_pass=pass_id,
                        last_pass=pass_id,
                    )
                    self.relations[key] = relation

                relation.relative_shift_sum += b_rel - a_rel
                relation.relation_count += 1
                relation.last_pass = pass_id
                relation_updates += 1

        self.previous_visible = current_visible
        self.previous_motion = motion_delta.copy()
        self.previous_rotation = rotation_delta.copy()

        self.events.append({
            "timestep": timestep,
            "pass": pass_id,
            "visible": len(observations),
            "landmarks": len(self.landmarks),
            "relations": len(self.relations),
            "created": created,
            "matched": matched,
            "reused": reused,
            "expanded": expanded,
            "relation_updates": relation_updates,
        })

    def begin_pass(self):
        # We retain knowledge, but reset only the short-term traversal state.
        self.previous_visible = []
        self.previous_motion = np.zeros(3)
        self.previous_rotation = np.zeros(3)

    def cost_slots(self):
        # Landmark: intrinsic feature (4), confidence/spread (1), count (1),
        # temporal metadata (2) = 8.
        landmark_cost = len(self.landmarks) * 8

        # Relation: relative motion (3), relative shift (3), count (1),
        # endpoints (2) = 9.
        relation_cost = len(self.relations) * 9
        return landmark_cost + relation_cost

    @staticmethod
    def raw_cost(observation_count):
        # relative position 3 + feature 4
        return observation_count * 7

    def pass_stats(self, pass_id):
        events = [e for e in self.events if e["pass"] == pass_id]
        return {
            "pass": pass_id,
            "observations": sum(e["visible"] for e in events),
            "new_landmarks": sum(e["created"] for e in events),
            "reused_landmarks": sum(e["reused"] for e in events),
            "relation_updates": sum(e["relation_updates"] for e in events),
            "final_landmarks": self.landmarks.__len__(),
            "final_relations": self.relations.__len__(),
        }


# ============================================================
# Trajectory
# ============================================================

def make_trajectory(steps=96):
    points = []
    for t in range(steps):
        a = 2 * math.pi * t / steps
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
    print("E36 v4b — LOCAL RELATIONAL MEMORY / DOUBLE TRAVERSAL")
    print("=" * 80)

    world = make_world()
    path = make_trajectory()
    memory = RelationalMemory()

    rows = []
    total_raw_observations = 0
    ever_seen = set()
    pass_summaries = {}

    # Two identical traversals, retaining memory between them.
    for pass_id in (1, 2):
        memory.begin_pass()
        pass_raw = 0
        pass_unique = set()
        previous_position = None
        previous_direction = None

        for t, position in enumerate(path):
            direction = direction_for_path(path, t)
            observer = Observer(position=position, direction=direction)
            visible = observe(observer, world)

            # Evaluator-only coverage accounting.
            for item in visible:
                pass_unique.add(item["world_index"])
                ever_seen.add(item["world_index"])

            local = memory_observations(visible)

            if previous_position is None:
                motion_delta = np.zeros(3)
            else:
                motion_delta = position - previous_position

            rot_delta = rotation_delta(previous_direction, direction)

            # Hard boundary assertion: these keys are the only things
            # memory is allowed to receive.
            allowed_keys = {"relative_position", "feature", "distance"}
            assert all(set(obs.keys()) == allowed_keys for obs in local)
            assert not any("world_index" in obs for obs in local)

            memory.update(
                local,
                timestep=t,
                pass_id=pass_id,
                motion_delta=motion_delta,
                rotation_delta=rot_delta,
            )

            pass_raw += len(local)
            total_raw_observations += len(local)
            previous_position = position.copy()
            previous_direction = direction.copy()

            rows.append({
                "pass": pass_id,
                "timestep": t,
                "visible": len(local),
                "unique_seen_this_pass": len(pass_unique),
                "unique_seen_total": len(ever_seen),
                "landmarks": len(memory.landmarks),
                "relations": len(memory.relations),
                "knowledge_cost_slots": memory.cost_slots(),
                "raw_observation_cost_slots": memory.raw_cost(total_raw_observations),
                "compression_ratio": (
                    memory.cost_slots() / memory.raw_cost(total_raw_observations)
                    if total_raw_observations else 0.0
                ),
                "created": memory.events[-1]["created"],
                "reused": memory.events[-1]["reused"],
                "expanded": memory.events[-1]["expanded"],
            })

            if t % 24 == 0 or t == len(path) - 1:
                event = memory.events[-1]
                print(
                    f"pass={pass_id} t={t:3d} | "
                    f"visible={len(local):3d} | "
                    f"new={event['created']:2d} | "
                    f"reuse={event['reused']:2d} | "
                    f"landmarks={len(memory.landmarks):3d} | "
                    f"relations={len(memory.relations):4d} | "
                    f"cost={memory.cost_slots():5d}"
                )

        pass_summaries[pass_id] = {
            "observations": pass_raw,
            "unique_world_points_seen_evaluator_only": len(pass_unique),
            "new_landmarks_created": sum(
                e["created"] for e in memory.events if e["pass"] == pass_id
            ),
            "landmarks_reused_from_prior_memory": sum(
                e["reused"] for e in memory.events if e["pass"] == pass_id
            ),
            "final_landmarks_after_pass": len(memory.landmarks),
            "final_relations_after_pass": len(memory.relations),
        }

    p1 = pass_summaries[1]
    p2 = pass_summaries[2]

    p1_new = p1["new_landmarks_created"]
    p2_new = p2["new_landmarks_created"]
    p2_reuse = p2["landmarks_reused_from_prior_memory"]

    # The second pass should not create another copy of the same learned
    # landmarks. This is the primary reuse signal.
    reuse_fraction = (
        p2_reuse / (p2_reuse + p2_new)
        if (p2_reuse + p2_new) else 0.0
    )

    results = {
        "experiment": "E36 v4b — local relational memory / double traversal",
        "world_points": len(world.voxels),
        "timesteps_per_pass": len(path),
        "passes": 2,
        "pass_summaries": pass_summaries,
        "total_unique_world_points_seen_evaluator_only": len(ever_seen),
        "final_landmarks": len(memory.landmarks),
        "final_relations": len(memory.relations),
        "total_raw_observations": total_raw_observations,
        "final_knowledge_cost_slots": memory.cost_slots(),
        "raw_observation_cost_slots": memory.raw_cost(total_raw_observations),
        "knowledge_to_raw_ratio": (
            memory.cost_slots() / memory.raw_cost(total_raw_observations)
        ),
        "pass2_landmark_reuse_fraction": reuse_fraction,
        "pass2_new_landmarks_vs_pass1": (
            p2_new / p1_new if p1_new else None
        ),
        "absolute_world_coordinates_given_to_memory": False,
        "world_index_given_to_memory": False,
        "position_derived_features": False,
        "hidden_geometry_given_to_memory": False,
        "movement_used_as_temporal_transition": True,
        "relational_memory": True,
        "double_pass_reuse": True,
        "trajectory": rows,
        "memory_events": memory.events,
    }

    out = Path("synthetic_latent_e36_local_relational_v4b")
    out.mkdir(exist_ok=True)
    (out / "e36_summary.json").write_text(json.dumps(results, indent=2))

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    t1 = [r["timestep"] for r in rows if r["pass"] == 1]
    t2 = [r["timestep"] for r in rows if r["pass"] == 2]

    new1 = [r["created"] for r in rows if r["pass"] == 1]
    new2 = [r["created"] for r in rows if r["pass"] == 2]
    reuse2 = [r["reused"] for r in rows if r["pass"] == 2]

    l1 = [r["landmarks"] for r in rows if r["pass"] == 1]
    l2 = [r["landmarks"] for r in rows if r["pass"] == 2]
    rel1 = [r["relations"] for r in rows if r["pass"] == 1]
    rel2 = [r["relations"] for r in rows if r["pass"] == 2]

    c1 = [r["knowledge_cost_slots"] for r in rows if r["pass"] == 1]
    c2 = [r["knowledge_cost_slots"] for r in rows if r["pass"] == 2]
    r1 = [r["compression_ratio"] for r in rows if r["pass"] == 1]
    r2 = [r["compression_ratio"] for r in rows if r["pass"] == 2]

    plt.figure(figsize=(9, 6))
    plt.plot(t1, new1, label="pass 1 new landmarks")
    plt.plot(t2, new2, label="pass 2 new landmarks")
    plt.plot(t2, reuse2, label="pass 2 reused landmarks")
    plt.xlabel("time within traversal")
    plt.ylabel("landmarks")
    plt.title("E36 v4b — New structure vs reused structure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_double_pass_reuse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(t1, l1, label="pass 1 landmarks")
    plt.plot(t2, l2, label="pass 2 landmarks")
    plt.plot(t1, rel1, label="pass 1 relations")
    plt.plot(t2, rel2, label="pass 2 relations")
    plt.xlabel("time within traversal")
    plt.ylabel("count")
    plt.title("E36 v4b — Relational topology growth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_relational_topology.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(t1, c1, label="pass 1 persistent cost")
    plt.plot(t2, c2, label="pass 2 persistent cost")
    plt.xlabel("time within traversal")
    plt.ylabel("scalar slots")
    plt.title("E36 v4b — Persistent relational memory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_relational_memory_cost.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(t1, r1, label="pass 1")
    plt.plot(t2, r2, label="pass 2")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("time within traversal")
    plt.ylabel("knowledge / raw observation cost")
    plt.title("E36 v4b — Compression across repeated traversal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "e36_double_pass_compression.png", dpi=160)
    plt.close()

    # Trajectory is shown for context only; memory never receives it.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    world_xyz = np.asarray([v.pos for v in world.voxels])
    path_xyz = np.asarray(path)
    ax.scatter(world_xyz[:, 0], world_xyz[:, 1], world_xyz[:, 2], s=8, alpha=0.18)
    ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], linewidth=2)
    ax.set_title("E36 v4b — World / observer trajectory (simulator only)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig(out / "e36_world_trajectory.png", dpi=160)
    plt.close()

    print()
    print("=" * 80)
    print("E36 v4b COMPLETE")
    print("=" * 80)
    print(f"World points:                 {len(world.voxels)}")
    print(f"Evaluator coverage:           {len(ever_seen)}/{len(world.voxels)}")
    print(f"Final landmarks:              {len(memory.landmarks)}")
    print(f"Final relations:              {len(memory.relations)}")
    print(f"Pass 1 new landmarks:         {p1_new}")
    print(f"Pass 2 new landmarks:         {p2_new}")
    print(f"Pass 2 reused landmarks:      {p2_reuse}")
    print(f"Pass 2 reuse fraction:         {reuse_fraction:.3f}")
    print(f"Final knowledge cost:          {memory.cost_slots()} slots")
    print(f"Raw observation cost:          {memory.raw_cost(total_raw_observations)} slots")
    print(f"Knowledge/raw ratio:            {memory.cost_slots() / memory.raw_cost(total_raw_observations):.3f}")
    print(f"Memory got world coordinates:   False")
    print(f"Memory got world IDs:           False")
    print(f"Results:                        {out.resolve()}")


if __name__ == "__main__":
    run()
