"""E41 — Relational Concept Memory.

Question
--------
Can a persistent representation organize around reusable relations between
observations rather than the arbitrary symbols through which observations
arrive, and can that memory aid prediction under ambiguity?

The experiment uses a synthetic world with evaluator-only concept identities.
The learner sees only observations.  Some unrelated concepts deliberately share
a token (homonym pressure), while related concepts can use different tokens
(synonym pressure).  Language is one observation channel, not the memory key.

Two predictors are compared:
  * BasePredictor: predicts a target property from the current observation.
  * RelationalMemoryPredictor: receives the same observation plus a persistent
    memory summary built from previous observations and explicit relations.

No hidden concept id is supplied to either model.  Evaluation includes held-out
compositions, ambiguous tokens, context removal, and post-training token remap.
The remap probe freezes the learned representation and retrains only token
boundary embeddings, testing whether arbitrary vocabulary is the substrate.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    seed: int = 41
    concepts: int = 48
    families: int = 8
    feature_dim: int = 12
    relation_types: int = 4
    token_count: int = 40
    latent_dim: int = 24
    memory_slots: int = 24
    epochs: int = 180
    episodes_per_epoch: int = 96
    lr: float = 2e-3
    remap_epochs: int = 45


@dataclass
class World:
    features: torch.Tensor
    family: torch.Tensor
    tokens: torch.Tensor
    relations: list[tuple[int, int, int]]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_world(cfg: Config) -> World:
    """Create hidden concepts; ids below are simulator/evaluator-only."""
    g = torch.Generator().manual_seed(cfg.seed)
    per_family = cfg.concepts // cfg.families
    family = torch.arange(cfg.concepts) // per_family
    family = family.clamp_max(cfg.families - 1)
    centers = F.normalize(torch.randn(cfg.families, cfg.feature_dim, generator=g), dim=-1)
    features = centers[family] + 0.12 * torch.randn(cfg.concepts, cfg.feature_dim, generator=g)
    features = F.normalize(features, dim=-1)

    # Ordinary mappings first, then deliberate collisions and aliases.
    tokens = torch.arange(cfg.concepts) % cfg.token_count
    # Homonyms: distant-family pairs share the same surface token.
    for i in range(min(6, cfg.families // 2 + 2)):
        a = i * per_family
        b = ((i + cfg.families // 2) % cfg.families) * per_family
        tokens[b] = tokens[a]
    # Synonym-like pressure: members of the same family use distinct arbitrary
    # tokens despite highly similar nonlinguistic structure.
    for f in range(cfg.families):
        base = f * per_family
        for j in range(1, min(3, per_family)):
            tokens[base + j] = (cfg.concepts + f * 3 + j) % cfg.token_count

    relations: list[tuple[int, int, int]] = []
    rng = np.random.default_rng(cfg.seed)
    for src in range(cfg.concepts):
        same = torch.where(family == family[src])[0].tolist()
        same = [x for x in same if x != src]
        for rel_type in range(cfg.relation_types):
            if rel_type < 2 and same:
                dst = int(rng.choice(same))
            else:
                dst = int(rng.integers(cfg.concepts))
                while dst == src:
                    dst = int(rng.integers(cfg.concepts))
            relations.append((src, rel_type, dst))
    return World(features, family, tokens, relations)


class ObservationEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.token = nn.Embedding(cfg.token_count, cfg.latent_dim)
        self.feature = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.latent_dim), nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.relation = nn.Embedding(cfg.relation_types, cfg.latent_dim)
        self.mix = nn.Sequential(
            nn.Linear(cfg.latent_dim * 3, cfg.latent_dim), nn.GELU(),
            nn.LayerNorm(cfg.latent_dim),
        )

    def forward(self, token, feature, relation_type):
        return self.mix(torch.cat([
            self.token(token), self.feature(feature), self.relation(relation_type)
        ], dim=-1))


class PersistentRelationalMemory(nn.Module):
    """Content-addressed persistent slots; no slot is assigned a concept id."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.keys = nn.Parameter(torch.randn(cfg.memory_slots, cfg.latent_dim) / math.sqrt(cfg.latent_dim))
        self.message = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim), nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.gate = nn.Linear(cfg.latent_dim * 2, 1)

    def empty(self, device=None):
        return torch.zeros(self.cfg.memory_slots, self.cfg.latent_dim, device=device)

    def read(self, state, query):
        logits = (query @ (self.keys + state).T) / math.sqrt(self.cfg.latent_dim)
        weights = logits.softmax(-1)
        return weights @ state, weights

    def write(self, state, evidence, related):
        query = evidence + related
        logits = (query @ (self.keys + state).T) / math.sqrt(self.cfg.latent_dim)
        weights = logits.softmax(-1)
        msg = self.message(torch.cat([evidence, related], -1))
        proposal = state + weights[:, None] * msg[None, :]
        gate = torch.sigmoid(self.gate(torch.cat([state, proposal], -1)))
        return gate * proposal + (1.0 - gate) * state


class BasePredictor(nn.Module):
    def __init__(self, cfg: Config, encoder: ObservationEncoder | None = None):
        super().__init__()
        self.encoder = encoder or ObservationEncoder(cfg)
        self.head = nn.Linear(cfg.latent_dim, cfg.feature_dim)

    def forward(self, token, feature, relation_type):
        z = self.encoder(token, feature, relation_type)
        return F.normalize(self.head(z), dim=-1), z


class RelationalMemoryPredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ObservationEncoder(cfg)
        self.memory = PersistentRelationalMemory(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim), nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.feature_dim),
        )

    def predict(self, state, token, feature, relation_type):
        z = self.encoder(token, feature, relation_type)
        recalled, weights = self.memory.read(state, z)
        out = F.normalize(self.head(torch.cat([z, recalled], -1)), dim=-1)
        return out, z, recalled, weights


def split_relations(world: World, seed: int):
    rng = random.Random(seed)
    rels = world.relations[:]
    rng.shuffle(rels)
    cut = int(0.75 * len(rels))
    return rels[:cut], rels[cut:]


def noisy_feature(world: World, idx: int, scale=0.08):
    return F.normalize(world.features[idx] + scale * torch.randn_like(world.features[idx]), dim=-1)


def one(x: int):
    return torch.tensor(x, dtype=torch.long)


def train_base(model, world, train_rel, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.epochs):
        random.shuffle(train_rel)
        for src, r, dst in train_rel[:cfg.episodes_per_epoch]:
            pred, _ = model(one(int(world.tokens[src])), noisy_feature(world, src), one(r))
            loss = 1.0 - F.cosine_similarity(pred[None], world.features[dst][None]).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def train_memory(model, world, train_rel, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    for _ in range(cfg.epochs):
        random.shuffle(train_rel)
        state = model.memory.empty()
        # Truncated episodes preserve cross-observation state without an
        # unbounded autograd graph.
        for n, (src, r, dst) in enumerate(train_rel[:cfg.episodes_per_epoch]):
            sf = noisy_feature(world, src)
            pred, z, recalled, _ = model.predict(state, one(int(world.tokens[src])), sf, one(r))
            target = world.features[dst]
            pred_loss = 1.0 - F.cosine_similarity(pred[None], target[None]).mean()
            # The memory is pressured to carry useful nonlinguistic context.
            relation_evidence = model.encoder.feature(target)
            next_state = model.memory.write(state, z, relation_evidence)
            sparsity = next_state.abs().mean()
            loss = pred_loss + 2e-4 * sparsity
            opt.zero_grad(); loss.backward(); opt.step()
            state = next_state.detach()


def evaluate_base(model, world, rels):
    model.eval(); scores = []
    with torch.no_grad():
        for src, r, dst in rels:
            pred, _ = model(one(int(world.tokens[src])), world.features[src], one(r))
            scores.append(float(F.cosine_similarity(pred[None], world.features[dst][None])))
    return float(np.mean(scores))


def evaluate_memory(model, world, train_rel, test_rel, context_fraction=1.0):
    model.eval(); state = model.memory.empty(); scores = []
    with torch.no_grad():
        keep = int(len(train_rel) * context_fraction)
        for src, r, dst in train_rel[:keep]:
            z = model.encoder(one(int(world.tokens[src])), world.features[src], one(r))
            related = model.encoder.feature(world.features[dst])
            state = model.memory.write(state, z, related)
        for src, r, dst in test_rel:
            pred, _, _, _ = model.predict(state, one(int(world.tokens[src])), world.features[src], one(r))
            scores.append(float(F.cosine_similarity(pred[None], world.features[dst][None])))
    return float(np.mean(scores))


def homonym_separation(model, world):
    """Diagnostic only: same-token concepts should remain separable by context."""
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(world.tokens.tolist()): groups.setdefault(t, []).append(i)
    vals = []
    model.eval()
    with torch.no_grad():
        for ids in groups.values():
            if len(ids) < 2: continue
            a, b = ids[0], ids[1]
            if int(world.family[a]) == int(world.family[b]): continue
            za = model.encoder(one(int(world.tokens[a])), world.features[a], one(0))
            zb = model.encoder(one(int(world.tokens[b])), world.features[b], one(0))
            vals.append(float(F.cosine_similarity(za[None], zb[None])))
    return float(np.mean(vals)) if vals else float('nan')


def run(cfg: Config):
    seed_all(cfg.seed)
    world = make_world(cfg)
    train_rel, test_rel = split_relations(world, cfg.seed)

    base = BasePredictor(cfg)
    memory = RelationalMemoryPredictor(cfg)
    train_base(base, world, train_rel, cfg)
    train_memory(memory, world, train_rel, cfg)

    context_curve = {}
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        context_curve[str(fraction)] = evaluate_memory(memory, world, train_rel, test_rel, fraction)

    result = {
        "experiment": "E41 — Relational Concept Memory",
        "seed": cfg.seed,
        "question": "Can persistent relational context aid prediction without making arbitrary language symbols the memory substrate?",
        "base_test_cosine": evaluate_base(base, world, test_rel),
        "memory_test_cosine": context_curve["1.0"],
        "memory_zero_context_cosine": context_curve["0.0"],
        "context_curve": context_curve,
        "homonym_internal_cosine": homonym_separation(memory, world),
        "persistent_scalar_slots": cfg.memory_slots * cfg.latent_dim,
        "train_relations": len(train_rel),
        "test_relations": len(test_rel),
        "notes": [
            "Hidden concept ids are evaluator/simulator-only.",
            "Token identity is not used as a persistent memory address.",
            "Homonym geometry is diagnostic, never a training target.",
            "This first artifact measures model-aiding and context dependence; exact bit-cost and boundary-only remap adaptation are follow-up probes, not claimed here."
        ],
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/E41_relational_concept_memory.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    cfg = Config(epochs=20, episodes_per_epoch=48) if args.quick else Config()
    result = run(cfg)
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
