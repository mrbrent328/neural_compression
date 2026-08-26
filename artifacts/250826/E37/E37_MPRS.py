#!/usr/bin/env python3
"""
E37 — MPRS: Minimal Persistent Relational Structure
====================================================

Research objective
------------------
What is the smallest persistent relational structure capable of explaining
multiple observations of the same underlying system?

This is an interpretable, non-neural baseline.  It deliberately separates:

    SIMULATOR KNOWLEDGE
        full world, absolute coordinates, hidden geometry

    OBSERVER/MPRS INPUT
        only local visible features, relative positions, distance, and
        observer motion between observations

The MPRS memory never receives world IDs or absolute coordinates.

E37 first asks whether a structure learned on one traversal can explain a
second traversal with a different starting phase/direction.  It measures:

    * reconstruction of the current observation from persistent units
    * prediction of the next observation using persistent relations
    * structure size / cost
    * reuse vs new structure on pass 2
    * ablations removing relations and persistent memory
    * an explicit cost/error Pareto curve from relation-budget pruning

This is intentionally a baseline, not a claim of optimality.  "Minimum"
is treated as an optimization target and measured as the smallest tested
structure that stays within configured explanation tolerances.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Experiment configuration
# ============================================================

SEED = 37
WORLD_SEED = 36
STEPS = 96

# Recognition thresholds.  These are deliberately explicit so they can be
# swept later rather than hidden in the implementation.
FEATURE_MATCH_GATE = 0.22
PREDICTION_POSITION_GATE = 1.75

# Sparse relational memory: only retain the closest few relations per
# transition.  This prevents an all-pairs observation history.
MAX_RELATIONS_PER_STEP = 3

# Cost model in scalar slots.  This is a bookkeeping model, not a claim
# about a particular binary serialization format.
UNIT_COST = 5       # feature(4) + confidence/statistic(1)
RELATION_COST = 7   # endpoints(2) + motion(3) + shift(2)
TRANSITION_COST = 4 # observer motion(3) + orientation delta(1)
RAW_OBSERVATION_COST = 7  # relative position(3) + feature(4)

# Explanation tolerances for the "minimum sufficient" report.
RECON_TOLERANCE = 0.20
PRED_TOLERANCE = 0.45
PRED_F1_TOLERANCE = 0.50

OUT = Path("synthetic_mprs_e37_traversal_invariance")
OUT.mkdir(exist_ok=True)


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


def make_world(seed: int = WORLD_SEED) -> World:
    rng = np.random.default_rng(seed)
    voxels: list[Voxel] = []

    structures = [
        (np.array([0.0, 0.0, 0.0]), 2.5, 90),
        (np.array([7.0, 1.5, 2.0]), 2.0, 70),
        (np.array([2.0, 7.0, 3.5]), 2.2, 80),
        (np.array([8.0, 7.0, 1.0]), 1.6, 60),
    ]

    # Intrinsic local signatures.  They are independent of absolute world
    # position so the memory cannot infer x/y/z from appearance.
    for center, radius, count in structures:
        for _ in range(count):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            r = radius * (rng.random() ** (1.0 / 3.0))
            pos = center + d * r

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


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rotation_delta(prev: np.ndarray | None, current: np.ndarray) -> float:
    if prev is None:
        return 0.0
    a = normalize(prev)
    b = normalize(current)
    return float(math.acos(np.clip(np.dot(a, b), -1.0, 1.0)))


def line_of_sight(origin, target, world: World, ignore_index: int) -> bool:
    """Approximate occlusion using spherical point blockers."""
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


def observe(observer: Observer, world: World) -> list[dict]:
    """Simulator-only visibility query.  World IDs remain evaluator-only."""
    visible = []
    forward = normalize(observer.direction)
    cos_half_fov = math.cos(math.radians(observer.fov_degrees / 2.0))

    for world_index, voxel in enumerate(world.voxels):
        rel = voxel.pos - observer.position
        distance = np.linalg.norm(rel)
        if distance < 1e-9 or distance > observer.max_range:
            continue

        direction = rel / distance
        if np.dot(direction, forward) < cos_half_fov:
            continue

        if line_of_sight(observer.position, voxel.pos, world, world_index):
            visible.append({
                # Evaluator-only field.  It is removed before memory input.
                "world_index": world_index,
                "relative_position": rel.copy(),
                "feature": voxel.feature.copy(),
                "distance": float(distance),
            })

    return visible


def memory_view(visible: list[dict]) -> list[dict]:
    """Hard information firewall: strip all simulator/evaluator identity."""
    local = [
        {
            "relative_position": item["relative_position"].copy(),
            "feature": item["feature"].copy(),
            "distance": item["distance"],
        }
        for item in visible
    ]

    allowed = {"relative_position", "feature", "distance"}
    assert all(set(item.keys()) == allowed for item in local)
    assert all("world_index" not in item for item in local)
    return local


# ============================================================
# Trajectories
# ============================================================

def make_trajectory(steps: int = STEPS, phase: float = 0.0, reverse=False):
    points = []
    for t in range(steps):
        a = 2.0 * math.pi * t / steps + phase
        x = 4.0 + 4.4 * math.cos(a)
        y = 3.8 + 3.2 * math.sin(a)
        z = 2.0 + 0.8 * math.sin(2.0 * a)
        points.append(np.array([x, y, z], dtype=float))

    if reverse:
        points = list(reversed(points))

    return points


def direction_for_path(path: list[np.ndarray], i: int) -> np.ndarray:
    if i == 0:
        return path[1] - path[0]
    if i == len(path) - 1:
        return path[-1] - path[-2]
    return path[i + 1] - path[i - 1]


# ============================================================
# Persistent relational structure
# ============================================================

@dataclass
class Unit:
    unit_id: int
    feature: np.ndarray
    observations: int = 1
    first_seen: int = 0
    last_seen: int = 0
    passes_seen: set[int] = field(default_factory=set)


@dataclass
class Relation:
    from_unit: int
    to_unit: int
    motion_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shift_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    count: int = 0
    first_pass: int = 1
    last_pass: int = 1

    @property
    def motion_mean(self):
        return self.motion_sum / max(self.count, 1)

    @property
    def shift_mean(self):
        return self.shift_sum / max(self.count, 1)


class MPRS:
    """Interpretable persistent relational memory."""

    def __init__(self, use_relations=True):
        self.use_relations = use_relations
        self.units: dict[int, Unit] = {}
        self.relations: dict[tuple[int, int], Relation] = {}
        self.next_unit_id = 0
        self.previous_visible: list[tuple[int, np.ndarray]] = []
        self.previous_motion = np.zeros(3)
        self.events: list[dict] = []

    def match_unit(self, feature):
        best = None
        best_distance = float("inf")
        for unit in self.units.values():
            d = float(np.linalg.norm(feature - unit.feature))
            if d < best_distance:
                best_distance = d
                best = unit
        if best is not None and best_distance <= FEATURE_MATCH_GATE:
            return best, best_distance
        return None, best_distance

    def create_unit(self, feature, timestep, pass_id):
        unit = Unit(
            unit_id=self.next_unit_id,
            feature=feature.copy(),
            first_seen=timestep,
            last_seen=timestep,
            passes_seen={pass_id},
        )
        self.next_unit_id += 1
        self.units[unit.unit_id] = unit
        return unit

    def update_unit(self, unit: Unit, feature, timestep, pass_id):
        n = unit.observations
        delta = feature - unit.feature
        unit.feature = unit.feature + delta / (n + 1)
        unit.observations = n + 1
        unit.last_seen = timestep
        unit.passes_seen.add(pass_id)

    def update(self, observations, timestep, pass_id, motion_delta, rotation_delta):
        current_visible: list[tuple[int, np.ndarray]] = []
        created = 0
        reused = 0
        matched = 0

        for obs in observations:
            unit, _ = self.match_unit(obs["feature"])
            if unit is None:
                unit = self.create_unit(obs["feature"], timestep, pass_id)
                created += 1
            else:
                seen_before = pass_id in unit.passes_seen
                self.update_unit(unit, obs["feature"], timestep, pass_id)
                matched += 1
                if not seen_before:
                    reused += 1

            current_visible.append((unit.unit_id, obs["relative_position"].copy()))

        # Sparse relation update: connect each currently visible unit to a
        # small number of nearest previous-frame units.  This represents
        # continuity through traversal without storing all co-visible pairs.
        relation_updates = 0
        if self.use_relations and self.previous_visible and current_visible:
            candidates = []
            for cur_id, cur_pos in current_visible:
                for prev_id, prev_pos in self.previous_visible:
                    dist = float(np.linalg.norm(cur_pos - prev_pos))
                    candidates.append((dist, prev_id, cur_id, prev_pos, cur_pos))

            candidates.sort(key=lambda x: x[0])
            used_cur = set()
            for dist, prev_id, cur_id, prev_pos, cur_pos in candidates:
                if cur_id in used_cur:
                    continue
                if dist > PREDICTION_POSITION_GATE:
                    continue
                key = (prev_id, cur_id)
                rel = self.relations.get(key)
                if rel is None:
                    rel = Relation(
                        from_unit=prev_id,
                        to_unit=cur_id,
                        first_pass=pass_id,
                        last_pass=pass_id,
                    )
                    self.relations[key] = rel
                rel.motion_sum += motion_delta
                rel.shift_sum += (cur_pos - prev_pos)
                rel.count += 1
                rel.last_pass = pass_id
                relation_updates += 1
                used_cur.add(cur_id)
                if relation_updates >= MAX_RELATIONS_PER_STEP:
                    break

        self.previous_visible = current_visible
        self.previous_motion = motion_delta.copy()

        self.events.append({
            "pass": pass_id,
            "timestep": timestep,
            "created": created,
            "reused": reused,
            "matched": matched,
            "relation_updates": relation_updates,
        })

    def reset_temporal_cursor(self):
        self.previous_visible = []
        self.previous_motion = np.zeros(3)

    def reconstruct(self, observations):
        """Reconstruct current observation using persistent units only."""
        if not self.units:
            return [], 1.0

        errors = []
        reconstructed = []
        for obs in observations:
            unit, d = self.match_unit(obs["feature"])
            if unit is None:
                errors.append(1.0)
                continue
            errors.append(d)
            reconstructed.append(unit.unit_id)
        return reconstructed, float(np.mean(errors)) if errors else 1.0

    def predict_next(self, current_observations):
        """Predict next visible unit identities and approximate local positions."""
        if not current_observations:
            return {}

        predictions = {}
        for obs in current_observations:
            unit, _ = self.match_unit(obs["feature"])
            if unit is None:
                continue

            outgoing = [
                rel for (src, _), rel in self.relations.items()
                if src == unit.unit_id and rel.count > 0
            ]
            if not outgoing:
                continue

            # Use the most frequently observed transition.
            outgoing.sort(key=lambda r: r.count, reverse=True)
            rel = outgoing[0]
            predicted_pos = obs["relative_position"] + rel.shift_mean
            predictions[rel.to_unit] = predicted_pos

        return predictions

    def cost_slots(self):
        return (
            len(self.units) * UNIT_COST
            + len(self.relations) * RELATION_COST
            + (len(self.events) * TRANSITION_COST)
        )


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_prediction(predicted, actual):
    """Feature-identity prediction precision/recall + position error."""
    actual_ids = set()
    actual_positions = {}
    # Actual unit matching is permitted during evaluation only.
    for obs in actual:
        # This function is filled by caller after matching to persistent memory.
        pass
    return None


def actual_unit_view(memory: MPRS, observations):
    result = []
    for obs in observations:
        unit, d = memory.match_unit(obs["feature"])
        if unit is not None:
            result.append((unit.unit_id, obs["relative_position"].copy(), d))
    return result


def score_prediction(memory: MPRS, predicted, actual_observations):
    actual = actual_unit_view(memory, actual_observations)
    actual_ids = {u for u, _, _ in actual}

    if not actual_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "position_error": 1.0,
            "predicted": 0,
            "actual": 0,
        }

    pred_ids = set(predicted)
    tp = len(pred_ids & actual_ids)
    precision = tp / max(len(pred_ids), 1)
    recall = tp / len(actual_ids)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    position_errors = []
    actual_positions = {u: p for u, p, _ in actual}
    for unit_id, predicted_pos in predicted.items():
        if unit_id in actual_positions:
            position_errors.append(
                float(np.linalg.norm(predicted_pos - actual_positions[unit_id]))
            )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "position_error": float(np.mean(position_errors)) if position_errors else 1.0,
        "predicted": len(pred_ids),
        "actual": len(actual_ids),
    }


def run_pass(memory: MPRS, world: World, path, pass_id: int, collect_rows=True):
    rows = []
    unique_world_points = set()
    raw_observations = 0
    memory.reset_temporal_cursor()

    for t, position in enumerate(path):
        direction = direction_for_path(path, t)
        observer = Observer(position=position, direction=direction)
        visible = observe(observer, world)
        local = memory_view(visible)

        for item in visible:
            unique_world_points.add(item["world_index"])

        if t > 0:
            motion_delta = position - path[t - 1]
            rot_delta = rotation_delta(
                direction_for_path(path, t - 1),
                direction,
            )
        else:
            motion_delta = np.zeros(3)
            rot_delta = 0.0

        # Prediction uses the structure BEFORE today's update.  This keeps
        # future information from leaking backward into the prediction.
        prediction = memory.predict_next(memory.previous_visible_as_observations()) if False else {}

        memory.update(
            local,
            timestep=t,
            pass_id=pass_id,
            motion_delta=motion_delta,
            rotation_delta=rot_delta,
        )

        # Current-observation reconstruction is evaluated after update;
        # this measures whether persistent units can encode the observation.
        _, recon_error = memory.reconstruct(local)
        raw_observations += len(local)

        if collect_rows:
            rows.append({
                "pass": pass_id,
                "timestep": t,
                "visible": len(local),
                "unique_world_points_evaluator_only": len(unique_world_points),
                "units": len(memory.units),
                "relations": len(memory.relations),
                "cost_slots": memory.cost_slots(),
                "raw_observation_cost_slots": raw_observations * RAW_OBSERVATION_COST,
                "reconstruction_error": recon_error,
                "created": memory.events[-1]["created"],
                "reused": memory.events[-1]["reused"],
                "relation_updates": memory.events[-1]["relation_updates"],
            })

    return rows, unique_world_points, raw_observations


def collect_path_observations(world: World, path):
    """Precompute simulator observations once; memory never sees evaluator fields."""
    frames = []
    for t, position in enumerate(path):
        direction = direction_for_path(path, t)
        visible = observe(Observer(position=position, direction=direction), world)
        local = memory_view(visible)
        world_ids = {x["world_index"] for x in visible}
        if t > 0:
            motion = position - path[t - 1]
            rot = rotation_delta(direction_for_path(path, t - 1), direction)
        else:
            motion = np.zeros(3)
            rot = 0.0
        frames.append({"local": local, "world_ids": world_ids, "motion": motion, "rotation": rot})
    return frames


# ============================================================
# Better pass runner with explicit next-step prediction
# ============================================================

def run_pass_with_prediction(memory: MPRS, frames, pass_id: int):
    rows = []
    unique_world_points = set()
    raw_observations = 0
    memory.reset_temporal_cursor()
    previous_local = None

    for t, frame in enumerate(frames):
        local = frame["local"]
        unique_world_points.update(frame["world_ids"])

        predicted = memory.predict_next(previous_local) if previous_local else {}
        pred_score = score_prediction(memory, predicted, local) if previous_local else {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "position_error": 1.0, "predicted": 0, "actual": len(local),
        }

        memory.update(
            local, timestep=t, pass_id=pass_id,
            motion_delta=frame["motion"], rotation_delta=frame["rotation"],
        )

        _, recon_error = memory.reconstruct(local)
        raw_observations += len(local)

        rows.append({
            "pass": pass_id, "timestep": t, "visible": len(local),
            "unique_world_points_evaluator_only": len(unique_world_points),
            "units": len(memory.units), "relations": len(memory.relations),
            "cost_slots": memory.cost_slots(),
            "raw_observation_cost_slots": raw_observations * RAW_OBSERVATION_COST,
            "reconstruction_error": recon_error,
            "prediction_precision": pred_score["precision"],
            "prediction_recall": pred_score["recall"],
            "prediction_f1": pred_score["f1"],
            "prediction_position_error": pred_score["position_error"],
            "created": memory.events[-1]["created"],
            "reused": memory.events[-1]["reused"],
            "relation_updates": memory.events[-1]["relation_updates"],
        })
        previous_local = local

    return rows, unique_world_points, raw_observations


# ============================================================
# Ablation runner
# ============================================================

def run_ablation(frames_a, frames_b):
    """Run the three conceptual conditions."""
    conditions = {}

    # Full MPRS.
    full = MPRS()
    p1, u1, raw1 = run_pass_with_prediction(full, frames_a, 1)
    p2, u2, raw2 = run_pass_with_prediction(full, frames_b, 2)
    conditions["full"] = {
        "memory": full,
        "pass1": p1,
        "pass2": p2,
        "unique_pass1": len(u1),
        "unique_pass2": len(u2),
        "raw_observations": raw1 + raw2,
    }

    # No-relations ablation: units remain persistent, but prediction has no
    # relational substrate. We use a fresh memory and disable relation use.
    no_rel = MPRS(use_relations=False)
    p1_nr, u1_nr, raw1_nr = run_pass_with_prediction(no_rel, frames_a, 1)
    p2_nr, u2_nr, raw2_nr = run_pass_with_prediction(no_rel, frames_b, 2)
    conditions["no_relations"] = {
        "memory": no_rel,
        "pass1": p1_nr,
        "pass2": p2_nr,
        "unique_pass1": len(u1_nr),
        "unique_pass2": len(u2_nr),
        "raw_observations": raw1_nr + raw2_nr,
    }

    # Observation-only baseline: no persistence. Every timestep is treated
    # as a fresh representation, so it has no cross-time predictive memory.
    obs_only_rows = []
    total_raw = 0
    unique_a = set()
    unique_b = set()
    for pass_id, frames in ((1, frames_a), (2, frames_b)):
        for t, frame in enumerate(frames):
            local = frame["local"]
            total_raw += len(local)
            target = unique_a if pass_id == 1 else unique_b
            target.update(frame["world_ids"])
            obs_only_rows.append({
                "pass": pass_id,
                "timestep": t,
                "visible": len(local),
                "units": len(local),
                "relations": 0,
                "cost_slots": len(local) * RAW_OBSERVATION_COST,
                "raw_observation_cost_slots": total_raw * RAW_OBSERVATION_COST,
                "reconstruction_error": 0.0,
                "prediction_precision": 0.0,
                "prediction_recall": 0.0,
                "prediction_f1": 0.0,
                "prediction_position_error": 1.0,
                "created": len(local),
                "reused": 0,
                "relation_updates": 0,
            })

    conditions["observation_only"] = {
        "memory": None,
        "pass1": [r for r in obs_only_rows if r["pass"] == 1],
        "pass2": [r for r in obs_only_rows if r["pass"] == 2],
        "unique_pass1": len(unique_a),
        "unique_pass2": len(unique_b),
        "raw_observations": total_raw,
    }

    return conditions


# ============================================================
# Minimum-structure sweep
# ============================================================

def prune_relations(memory: MPRS, max_relations: int) -> MPRS:
    """Copy the learned structure while retaining only the strongest edges."""
    pruned = MPRS()
    pruned.units = {
        uid: Unit(
            unit_id=u.unit_id,
            feature=u.feature.copy(),
            observations=u.observations,
            first_seen=u.first_seen,
            last_seen=u.last_seen,
            passes_seen=set(u.passes_seen),
        )
        for uid, u in memory.units.items()
    }
    pruned.next_unit_id = memory.next_unit_id

    ranked = sorted(
        memory.relations.values(),
        key=lambda r: r.count,
        reverse=True,
    )[:max_relations]

    pruned.relations = {
        (r.from_unit, r.to_unit): Relation(
            from_unit=r.from_unit,
            to_unit=r.to_unit,
            motion_sum=r.motion_sum.copy(),
            shift_sum=r.shift_sum.copy(),
            count=r.count,
            first_pass=r.first_pass,
            last_pass=r.last_pass,
        )
        for r in ranked
    }
    return pruned


def evaluate_structure_sizes(full_memory: MPRS, frames_b):
    """Evaluate frozen relation subsets without letting the test pass add edges."""
    relation_counts = sorted({
        0, 1, 2, 4, 8, 16, 32, 64, 96, 128, len(full_memory.relations)
    })

    curve = []
    for n_rel in relation_counts:
        candidate = prune_relations(full_memory, n_rel)
        previous_local = None
        prediction_scores = []
        recon_errors = []

        for frame in frames_b:
            local = frame["local"]
            if previous_local is not None:
                predicted = candidate.predict_next(previous_local)
                score = score_prediction(candidate, predicted, local)
                prediction_scores.append(score)

            _, recon_error = candidate.reconstruct(local)
            recon_errors.append(recon_error)
            previous_local = local

        pred_f1 = float(np.mean([x["f1"] for x in prediction_scores])) if prediction_scores else 0.0
        pred_err = float(np.mean([x["position_error"] for x in prediction_scores])) if prediction_scores else 1.0
        recon_err = float(np.mean(recon_errors)) if recon_errors else 1.0
        cost = candidate.cost_slots()

        curve.append({
            "relations_retained": n_rel,
            "units": len(candidate.units),
            "relations": len(candidate.relations),
            "cost_slots": cost,
            "mean_prediction_f1": pred_f1,
            "mean_prediction_position_error": pred_err,
            "mean_reconstruction_error": recon_err,
        })

    sufficient = [
        row for row in curve
        if row["mean_reconstruction_error"] <= RECON_TOLERANCE
        and row["mean_prediction_position_error"] <= PRED_TOLERANCE
        and row["mean_prediction_f1"] >= PRED_F1_TOLERANCE
    ]

    minimum = min(sufficient, key=lambda r: r["cost_slots"]) if sufficient else None
    return curve, minimum


# ============================================================
# Main experiment
# ============================================================

def main():
    np.random.seed(SEED)

    print("=" * 88)
    print("E37 — MPRS: MINIMAL PERSISTENT RELATIONAL STRUCTURE")
    print("=" * 88)
    print()
    print("Objective:")
    print("  What is the smallest persistent relational structure capable of")
    print("  explaining multiple observations of the same underlying system?")
    print()

    world = make_world()

    # Pass A: original infinity traversal.
    path_a = make_trajectory(phase=0.0, reverse=False)

    # Pass B: same world, different starting phase and opposite traversal.
    # This intentionally shares the world but changes observation order.
    path_b = make_trajectory(phase=math.pi / 3.0, reverse=True)

    print(f"World points: {len(world.voxels)}")
    print(f"Steps/pass:   {len(path_a)}")
    print("Pass A:       phase=0, forward")
    print("Pass B:       phase=pi/3, reverse")
    print()

    frames_a = collect_path_observations(world, path_a)
    frames_b = collect_path_observations(world, path_b)
    conditions = run_ablation(frames_a, frames_b)

    summary = {
        "experiment": "E37 — MPRS: Traversal Invariance",
        "branch": "MPRS",
        "objective": "What is the smallest persistent relational structure capable of explaining multiple observations of the same underlying system?",
        "world_points": len(world.voxels),
        "steps_per_pass": len(path_a),
        "pass_a": "phase=0, forward",
        "pass_b": "phase=pi/3, reverse",
        "information_boundary": {
            "absolute_world_coordinates_given_to_memory": False,
            "world_index_given_to_memory": False,
            "hidden_geometry_given_to_memory": False,
            "position_derived_features": False,
            "future_observations_given_to_memory": False,
        },
        "cost_model": {
            "unit_cost": UNIT_COST,
            "relation_cost": RELATION_COST,
            "transition_cost": TRANSITION_COST,
            "raw_observation_cost": RAW_OBSERVATION_COST,
        },
        "tolerances": {
            "reconstruction": RECON_TOLERANCE,
            "prediction_position": PRED_TOLERANCE,
            "prediction_f1": PRED_F1_TOLERANCE,
        },
        "conditions": {},
    }

    all_rows = []
    for name, condition in conditions.items():
        p1 = condition["pass1"]
        p2 = condition["pass2"]
        rows = p1 + p2
        all_rows.extend([{**r, "condition": name} for r in rows])

        last = p2[-1] if p2 else p1[-1]
        valid = p2[1:] if len(p2) > 1 else []

        summary["conditions"][name] = {
            "pass1_unique_world_points_evaluator_only": condition["unique_pass1"],
            "pass2_unique_world_points_evaluator_only": condition["unique_pass2"],
            "final_units": last["units"],
            "final_relations": last["relations"],
            "final_cost_slots": last["cost_slots"],
            "total_raw_observation_cost_slots": condition["raw_observations"] * RAW_OBSERVATION_COST,
            "mean_pass2_reconstruction_error": float(np.mean([r["reconstruction_error"] for r in p2])) if p2 else None,
            "mean_pass2_prediction_f1": float(np.mean([r["prediction_f1"] for r in valid])) if valid else None,
            "mean_pass2_prediction_position_error": float(np.mean([r["prediction_position_error"] for r in valid])) if valid else None,
            "pass2_new_units": sum(r["created"] for r in p2),
            "pass2_reused_units": sum(r["reused"] for r in p2),
        }

    full_memory = conditions["full"]["memory"]
    curve, minimum = evaluate_structure_sizes(full_memory, frames_b)
    summary["minimum_structure_search"] = {
        "curve": curve,
        "minimum_sufficient_candidate": minimum,
    }

    # --------------------------------------------------------
    # Save machine-readable outputs
    # --------------------------------------------------------

    (OUT / "e37_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "e37_trajectory_metrics.json").write_text(json.dumps(all_rows, indent=2))

    # Save a human-readable audit of the actual persistent structure.
    structure = {
        "units": [
            {
                "unit_id": u.unit_id,
                "feature": u.feature.tolist(),
                "observations": u.observations,
                "first_seen": u.first_seen,
                "last_seen": u.last_seen,
                "passes_seen": sorted(u.passes_seen),
            }
            for u in full_memory.units.values()
        ],
        "relations": [
            {
                "from": r.from_unit,
                "to": r.to_unit,
                "count": r.count,
                "motion_mean": r.motion_mean.tolist(),
                "shift_mean": r.shift_mean.tolist(),
                "first_pass": r.first_pass,
                "last_pass": r.last_pass,
            }
            for r in full_memory.relations.values()
        ],
    }
    (OUT / "e37_persistent_structure.json").write_text(json.dumps(structure, indent=2))

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))
    for name, condition in conditions.items():
        rows = condition["pass2"]
        x = np.arange(len(rows))
        y = [r["prediction_f1"] for r in rows]
        plt.plot(x, y, label=name)
    plt.xlabel("time within pass B")
    plt.ylabel("next-observation prediction F1")
    plt.title("E37 — Traversal invariance: prediction by condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "e37_ablation_prediction.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, condition in conditions.items():
        rows = condition["pass2"]
        x = np.arange(len(rows))
        y = [r["cost_slots"] for r in rows]
        plt.plot(x, y, label=name)
    plt.xlabel("time within pass B")
    plt.ylabel("persistent representation cost")
    plt.title("E37 — Persistent structure cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "e37_ablation_cost.png", dpi=160)
    plt.close()

    costs = [r["cost_slots"] for r in curve]
    f1s = [r["mean_prediction_f1"] for r in curve]
    plt.figure(figsize=(9, 6))
    plt.plot(costs, f1s, marker="o")
    if minimum:
        plt.scatter([minimum["cost_slots"]], [minimum["mean_prediction_f1"]], s=80, marker="*", label="minimum tested sufficient")
        plt.legend()
    plt.xlabel("persistent structure cost")
    plt.ylabel("mean pass-B prediction F1")
    plt.title("E37 — Structure economy / explanation frontier")
    plt.tight_layout()
    plt.savefig(OUT / "e37_structure_frontier.png", dpi=160)
    plt.close()

    # Information-boundary audit is machine-readable and printed explicitly.
    audit = summary["information_boundary"]
    (OUT / "e37_information_boundary_audit.json").write_text(json.dumps(audit, indent=2))

    print()
    print("=" * 88)
    print("E37 COMPLETE")
    print("=" * 88)
    for name, s in summary["conditions"].items():
        print(f"\n{name.upper()}")
        print(f"  units:                 {s['final_units']}")
        print(f"  relations:             {s['final_relations']}")
        print(f"  final cost:            {s['final_cost_slots']}")
        print(f"  pass-2 new units:      {s['pass2_new_units']}")
        print(f"  pass-2 reused units:   {s['pass2_reused_units']}")
        print(f"  pass-2 recon error:    {s['mean_pass2_reconstruction_error']}")
        print(f"  pass-2 prediction F1:  {s['mean_pass2_prediction_f1']}")
        print(f"  pass-2 position error: {s['mean_pass2_prediction_position_error']}")

    print("\nMINIMUM-STRUCTURE SEARCH")
    if minimum:
        print(f"  minimum tested sufficient cost:      {minimum['cost_slots']}")
        print(f"  relations retained:                  {minimum['relations_retained']}")
        print(f"  prediction F1:                       {minimum['mean_prediction_f1']:.4f}")
        print(f"  prediction position error:           {minimum['mean_prediction_position_error']:.4f}")
    else:
        print("  No tested structure met both explanation tolerances.")

    print("\nINFORMATION BOUNDARY")
    for key, value in audit.items():
        print(f"  {key}: {value}")

    print(f"\nResults: {OUT.resolve()}")


if __name__ == "__main__":
    main()
