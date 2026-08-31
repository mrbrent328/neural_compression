#!/usr/bin/env python3
"""Developmental Stream — continuously learning relational organism.

This experiment tests whether a small relational system can maintain useful
internal state and finite-capacity memory while learning from a continuous
stream of experience. There are no semantic labels or supervised concepts.

Persistent identity vectors are fixed. Encoder, interaction, predictor, and
consolidation policy learn online. Context is mutable, and long-term memory is
retrieved by similarity and fed back into prediction.

This is an exploratory engineering experiment, not a neuroscience or
consciousness claim.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 314159
STEPS = 5000
NUM_ENTITIES = 16
INPUT_DIM = 8
STATE_DIM = 32
SHORT_TERM_CAPACITY = 32
LONG_TERM_CAPACITY = 12
MEMORY_CANDIDATES = 4
SNAPSHOT_EVERY = 250
LEARNING_RATE = 1e-3
RETRIEVAL_MIX = 0.5


@dataclass
class Experience:
    step: int
    entity: int
    observation: List[float]
    prediction_error: float
    consolidated: bool = False


class ConstrainedWorld:
    """A numerical world whose dynamics are defined only by constraints."""

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
    """Fixed identities plus learned dynamics over mutable contextual state."""

    def __init__(self):
        super().__init__()
        self.identity = nn.Parameter(torch.randn(NUM_ENTITIES, STATE_DIM) * 0.05,
                                     requires_grad=False)
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

    def predict_from_context(self, entity: int, context: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.identity[entity], context], dim=-1)
        return self.predictor(features)

    def predict(self, entity: int) -> torch.Tensor:
        return self.predict_from_context(entity, self.context[entity])

    def surprise(self, entity: int, observation: torch.Tensor) -> float:
        with torch.no_grad():
            prediction = self.predict(entity)
            return float(F.mse_loss(prediction, observation).item())

    @torch.no_grad()
    def observe(self, entity: int, observation: torch.Tensor) -> float:
        encoded = self.encoder(observation)
        current = self.context[entity]
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
        return [item for item in self.short_term if not item.consolidated][-MEMORY_CANDIDATES:]

    def retrieve(self, entity: int, current_state: torch.Tensor) -> torch.Tensor | None:
        memories = [entry for entry in self.long_term if entry[0] == entity]
        if not memories:
            return None
        states = torch.stack([entry[1] for entry in memories])
        query = current_state.detach().unsqueeze(0)
        similarities = F.cosine_similarity(states, query, dim=-1)
        weights = torch.softmax(similarities, dim=0)
        return torch.sum(states * weights.unsqueeze(-1), dim=0)

    def consolidate(self, item: Experience, state: torch.Tensor, score: float) -> bool:
        if item.consolidated:
            return False
        self.long_term.append((item.entity, state.detach().clone(), score, item.step))
        item.consolidated = True
        if len(self.long_term) > LONG_TERM_CAPACITY:
            self.long_term.sort(key=lambda x: (x[2], x[3]), reverse=True)
            self.long_term.pop()
        return True


def set_seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)


def train_step(core: RelationalCore, optimizer: torch.optim.Optimizer,
               entity: int, observation: torch.Tensor,
               memory: DevelopmentalMemory) -> Tuple[float, float]:
    """Learn prediction and consolidation usefulness from the current stream."""
    optimizer.zero_grad()

    current = core.context[entity].detach()
    prediction = core.predict_from_context(entity, current)
    prediction_loss = F.mse_loss(prediction, observation)

    retrieved = memory.retrieve(entity, core.state(entity))
    if retrieved is not None:
        memory_context = ((1.0 - RETRIEVAL_MIX) * current + RETRIEVAL_MIX * retrieved).detach()
        memory_prediction = core.predict_from_context(entity, memory_context)
        prediction_loss = prediction_loss + F.mse_loss(memory_prediction, observation)

    # The consolidation target is generated by consequence, not semantics:
    # a candidate is useful when using its stored state predicts the present
    # observation better than the current state does.
    baseline_error = F.mse_loss(core.predict_from_context(entity, current), observation).detach()
    candidates = memory.candidates()
    consolidation_loss = torch.zeros((), dtype=observation.dtype)
    if candidates:
        targets = []
        scores = []
        for candidate in candidates:
            candidate_state = core.state(candidate.entity).detach()
            candidate_prediction = core.predict_from_context(entity, candidate_state)
            candidate_error = F.mse_loss(candidate_prediction, observation).detach()
            target = torch.sigmoid((baseline_error - candidate_error) * 10.0)
            target = target.clamp(0.05, 0.95)
            features = torch.cat([core.state(entity).detach(), candidate_state], dim=-1)
            score = core.consolidator(features).squeeze()
            targets.append(target)
            scores.append(score)
        consolidation_loss = F.binary_cross_entropy(torch.stack(scores), torch.stack(targets))

    loss = prediction_loss + consolidation_loss
    loss.backward()
    optimizer.step()
    return float(prediction_loss.item()), float(consolidation_loss.item())


def run() -> Dict:
    set_seed()
    world = ConstrainedWorld()
    core = RelationalCore()
    memory = DevelopmentalMemory()

    trainable = [parameter for parameter in core.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)

    events: List[Dict] = []
    total_surprise = 0.0
    total_prediction_loss = 0.0
    total_consolidation_loss = 0.0
    consolidated = 0

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
        prediction_loss, consolidation_loss = train_step(
            core, optimizer, entity, observation, memory
        )
        total_prediction_loss += prediction_loss
        total_consolidation_loss += consolidation_loss

        candidates = memory.candidates()
        if candidates:
            scored = [
                (candidate, core.consolidation_score(
                    entity,
                    core.state(candidate.entity).detach(),
                ))
                for candidate in candidates
            ]
            candidate, score = max(scored, key=lambda pair: pair[1])
            if memory.consolidate(candidate, core.state(candidate.entity), score):
                consolidated += 1
                events.append({
                    "step": step,
                    "event": "consolidate",
                    "source_step": candidate.step,
                    "entity": candidate.entity,
                    "score": score,
                })

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
                "last_prediction_loss": prediction_loss,
                "last_consolidation_loss": consolidation_loss,
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
        "mean_prediction_loss": total_prediction_loss / STEPS,
        "mean_consolidation_loss": total_consolidation_loss / STEPS,
        "identity_weight_updates_during_run": False,
        "online_learning": True,
        "retrieval_affects_prediction": True,
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
        "interpretation_boundary": "This run measures continuous state evolution, online prediction learning, retrieval, and finite-capacity memory behavior. It does not establish semantics, consciousness, or human-like cognition.",
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
