#!/usr/bin/env python3
"""Developmental Stream — continuously learning relational organism.

Experiment proposal: test whether a small relational system can develop
persistent internal structure from continuous experience without predefined
semantic labels, language, coordinates, or a supervised relation-prediction
task.

This first implementation keeps neural weights stable after initialization.
Learning occurs through mutable short-term state and finite-capacity long-term
relational memory. The environment is generated from constraints rather than
semantic labels. Memory consolidation is learned from prediction utility.

This is an exploratory engineering experiment, not a neuroscience or
consciousness claim.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

SEED = 314159
STEPS = 5000
NUM_ENTITIES = 16
INPUT_DIM = 8
STATE_DIM = 32
SHORT_TERM_CAPACITY = 32
LONG_TERM_CAPACITY = 12
MEMORY_CANDIDATES = 4
SNAPSHOT_EVERY = 250


@dataclass
class Experience:
    step: int
    entity: int
    observation: List[float]
    prediction_error: float
    consolidated: bool = False


class ConstrainedWorld:
    """A non-semantic world whose dynamics are defined only by constraints."""

    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)
        self.values = [self.rng.uniform(-1.0, 1.0) for _ in range(NUM_ENTITIES)]
        self.velocities = [self.rng.uniform(-0.05, 0.05) for _ in range(NUM_ENTITIES)]

    def step(self) -> Tuple[int, torch.Tensor]:
        entity = self.rng.randrange(NUM_ENTITIES)
        for i in range(NUM_ENTITIES):
            self.values[i] += self.velocities[i]
            if abs(self.values[i]) > 1.0:
                self.velocities[i] *= -1.0
                self.values[i] = max(-1.0, min(1.0, self.values[i]))

        neighborhood = [
            self.values[(entity + offset) % NUM_ENTITIES]
            for offset in (-2, -1, 0, 1, 2)
        ]
        observation = neighborhood + [
            self.velocities[entity],
            math.sin(self.values[entity] * math.pi),
            math.cos(self.values[entity] * math.pi),
        ]
        return entity, torch.tensor(observation, dtype=torch.float32)


class RelationalCore(nn.Module):
    """Persistent identities plus mutable relational/contextual state."""

    def __init__(self):
        super().__init__()
        self.identity = nn.Parameter(torch.randn(NUM_ENTITIES, STATE_DIM) * 0.05)
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, STATE_DIM),
            nn.Tanh(),
            nn.Linear(STATE_DIM, STATE_DIM),
            nn.Tanh(),
        )
        self.interaction = nn.Sequential(
            nn.Linear(STATE_DIM * 3, STATE_DIM * 2),
            nn.Tanh(),
            nn.Linear(STATE_DIM * 2, STATE_DIM),
            nn.Tanh(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(STATE_DIM * 2, STATE_DIM),
            nn.Tanh(),
            nn.Linear(STATE_DIM, INPUT_DIM),
        )
        self.consolidator = nn.Sequential(
            nn.Linear(STATE_DIM * 2, STATE_DIM),
            nn.Tanh(),
            nn.Linear(STATE_DIM, 1),
            nn.Sigmoid(),
        )
        self.register_buffer("context", torch.zeros(NUM_ENTITIES, STATE_DIM))

    def state(self, entity: int) -> torch.Tensor:
        return self.identity[entity] + self.context[entity]

    def predict(self, entity: int) -> torch.Tensor:
        return self.predictor(torch.cat([self.identity[entity], self.context[entity]], dim=-1))

    def surprise(self, entity: int, observation: torch.Tensor) -> float:
        prediction = self.predict(entity)
        return float(torch.mean((prediction - observation) ** 2).item())

    @torch.no_grad()
    def observe(self, entity: int, observation: torch.Tensor) -> float:
        encoded = self.encoder(observation)
        current = self.context[entity]
        # Use nearby persistent identities as the relational neighborhood.
        left = self.identity[(entity - 1) % NUM_ENTITIES]
        right = self.identity[(entity + 1) % NUM_ENTITIES]
        message = self.interaction(torch.cat([encoded, left, right], dim=-1))
        gate = torch.sigmoid(torch.mean(torch.abs(encoded - current)))
        self.context[entity].copy_((1.0 - gate) * current + gate * message)
        return float(gate.item())

    @torch.no_grad()
    def consolidation_score(self, entity: int, memory_state: torch.Tensor) -> float:
        features = torch.cat([self.state(entity), memory_state], dim=-1)
        return float(self.consolidator(features).item())


class DevelopmentalMemory:
    def __init__(self):
        self.short_term: List[Experience] = []
        self.long_term: List[Tuple[int, torch.Tensor, float, int]] = []

    def add_short_term(self, item: Experience) -> None:
        self.short_term.append(item)
        if len(self.short_term) > SHORT_TERM_CAPACITY:
            self.short_term.pop(0)

    def candidates(self) -> List[Experience]:
        return sorted(
            self.short_term,
            key=lambda x: x.prediction_error,
            reverse=True,
        )[:MEMORY_CANDIDATES]

    def consolidate(self, item: Experience, state: torch.Tensor, score: float) -> bool:
        if score <= 0.5:
            return False
        self.long_term.append((item.entity, state.detach().clone(), score, item.step))
        if len(self.long_term) > LONG_TERM_CAPACITY:
            # Learned scores determine which memory survives. Ties resolve by age.
            self.long_term.sort(key=lambda x: (x[2], x[3]), reverse=True)
            self.long_term.pop()
        item.consolidated = True
        return True

    def replay_for(self, entity: int) -> List[torch.Tensor]:
        return [state for e, state, _, _ in self.long_term if e == entity]


def set_seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)


def run() -> Dict:
    set_seed()
    world = ConstrainedWorld()
    core = RelationalCore()
    memory = DevelopmentalMemory()

    # The first version intentionally does not optimize neural weights online.
    # This isolates the effect of persistent state and learned consolidation.
    core.eval()

    events: List[Dict] = []
    total_surprise = 0.0
    consolidated = 0
    forgotten = 0

    for step in range(STEPS):
        entity, observation = world.step()
        error = core.surprise(entity, observation)
        total_surprise += error

        experience = Experience(
            step=step,
            entity=entity,
            observation=observation.tolist(),
            prediction_error=error,
        )
        memory.add_short_term(experience)

        gate = core.observe(entity, observation)

        # Consolidation is endogenous to the core: it scores candidate memories
        # using current internal state rather than a hand-coded novelty threshold.
        for candidate in memory.candidates():
            candidate_state = core.state(candidate.entity)
            score = core.consolidation_score(candidate.entity, candidate_state)
            if memory.consolidate(candidate, candidate_state, score):
                consolidated += 1
                events.append({
                    "step": step,
                    "event": "consolidate",
                    "source_step": candidate.step,
                    "entity": candidate.entity,
                    "score": score,
                })
                break

        if step % SNAPSHOT_EVERY == 0:
            events.append({
                "step": step,
                "event": "snapshot",
                "short_term": len(memory.short_term),
                "long_term": len(memory.long_term),
                "mean_context_norm": float(torch.linalg.vector_norm(core.context, dim=-1).mean().item()),
                "mean_identity_norm": float(torch.linalg.vector_norm(core.identity, dim=-1).mean().item()),
                "last_gate": gate,
                "last_surprise": error,
            })

    return {
        "experiment": "Developmental Stream",
        "seed": SEED,
        "steps": STEPS,
        "entities": NUM_ENTITIES,
        "state_dim": STATE_DIM,
        "short_term_capacity": SHORT_TERM_CAPACITY,
        "long_term_capacity": LONG_TERM_CAPACITY,
        "total_consolidations": consolidated,
        "long_term_final_size": len(memory.long_term),
        "mean_prediction_error": total_surprise / STEPS,
        "identity_weight_updates_during_run": False,
        "events": events,
        "long_term_memory": [
            {
                "entity": entity,
                "score": score,
                "source_step": source_step,
                "state_norm": float(torch.linalg.vector_norm(state).item()),
            }
            for entity, state, score, source_step in memory.long_term
        ],
        "interpretation_boundary": "This run measures continuous state evolution and finite-capacity memory behavior. It does not establish semantics, consciousness, or human-like cognition.",
    }


def main() -> None:
    result = run()
    path = Path("developmental_stream_result.json")
    path.write_text(json.dumps(result, indent=2))
    print("Developmental Stream")
    print(json.dumps({k: v for k, v in result.items() if k not in {"events", "long_term_memory"}}, indent=2))
    print(f"result: {path}")


if __name__ == "__main__":
    main()
