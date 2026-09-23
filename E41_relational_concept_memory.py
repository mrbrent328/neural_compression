"""E41 — Relational Concept Memory.

Question
--------
Can a persistent representation organize around reusable relations between
observations rather than the arbitrary symbols through which observations
arrive, and can that memory aid a separate predictor under genuine ambiguity?

Design
------
A synthetic world contains evaluator-only concept identities, continuous
nonlinguistic features, arbitrary surface tokens, aliases, and typed relations.

The decisive comparison is deliberately strict:

    frozen BasePredictor
        current observation -> prediction

    MemoryAidedPredictor
        the same frozen BasePredictor
        + persistent relational state built from prior observations
        -> residual correction -> prediction

The memory-aided model therefore cannot win merely because it has a different
base encoder/head. With zero memory its output is exactly the frozen base
prediction.

Two language pathologies are built into the world:

* homonym pressure: different underlying concepts share the same primary token;
* alias/synonym pressure: the same underlying concept may arrive through two
  different tokens.

For the homonym probe the current nonlinguistic feature is masked to zero.
Thus same-token concepts present exactly the same local token/feature evidence
for a given relation type. Any disambiguation must come from persistent
relational context accumulated from prior observations of that concept.

Hidden concept ids are used only by the simulator/evaluator to choose which
observations belong to an episode. They are never supplied to either network.

The post-training remap probe permutes the vocabulary and retrains only the
token embedding table. All other representation, memory, and prediction
parameters remain frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import copy
import json
import math
import random
from pathlib import Path

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
    token_count: int = 64
    latent_dim: int = 24
    memory_slots: int = 24
    epochs: int = 180
    episodes_per_epoch: int = 96
    lr: float = 2e-3
    remap_epochs: int = 45
    sparsity_weight: float = 2e-4


@dataclass
class World:
    # All indices into these tensors are simulator/evaluator-only concept ids.
    features: torch.Tensor
    family: torch.Tensor
    tokens: torch.Tensor
    aliases: torch.Tensor
    relations: list[tuple[int, int, int]]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_world(cfg: Config) -> World:
    """Create a latent world whose symbol mapping is intentionally misleading."""
    if cfg.concepts % cfg.families != 0:
        raise ValueError("concepts must be divisible by families")
    if cfg.token_count <= cfg.concepts:
        raise ValueError("token_count must exceed concepts so aliases can be distinct")

    g = torch.Generator().manual_seed(cfg.seed)
    per_family = cfg.concepts // cfg.families
    family = torch.arange(cfg.concepts) // per_family

    centers = F.normalize(
        torch.randn(cfg.families, cfg.feature_dim, generator=g), dim=-1
    )
    features = centers[family] + 0.12 * torch.randn(
        cfg.concepts, cfg.feature_dim, generator=g
    )
    features = F.normalize(features, dim=-1)

    # Start with one unique primary symbol per concept.
    tokens = torch.arange(cfg.concepts, dtype=torch.long)

    # Homonyms: concepts from distant families share a primary surface symbol.
    for i in range(min(6, cfg.families)):
        a = i * per_family
        b = ((i + cfg.families // 2) % cfg.families) * per_family
        if a != b:
            tokens[b] = tokens[a]

    # Aliases: the *same* underlying concept can arrive with a different token.
    # -1 means that concept has no alias.
    aliases = torch.full((cfg.concepts,), -1, dtype=torch.long)
    alias_pool = list(range(cfg.concepts, cfg.token_count))
    for n, concept in enumerate(range(min(len(alias_pool), cfg.concepts))):
        aliases[concept] = alias_pool[n]

    # Relations are generated from hidden structure. Two relation types are
    # mostly within-family, while the others are unconstrained.
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

    return World(features, family, tokens, aliases, relations)


def token_choices(world: World, concept: int) -> tuple[int, ...]:
    primary = int(world.tokens[concept])
    alias = int(world.aliases[concept])
    return (primary,) if alias < 0 else (primary, alias)


def sample_token(world: World, concept: int) -> int:
    return random.choice(token_choices(world, concept))


def ambiguous_primary_tokens(world: World) -> set[int]:
    groups: dict[int, set[int]] = {}
    for i, token in enumerate(world.tokens.tolist()):
        groups.setdefault(int(token), set()).add(int(world.family[i]))
    return {token for token, families in groups.items() if len(families) > 1}


def is_ambiguous_concept(world: World, concept: int) -> bool:
    return int(world.tokens[concept]) in ambiguous_primary_tokens(world)


def split_relations(world: World, seed: int):
    rng = random.Random(seed)
    rels = world.relations[:]
    rng.shuffle(rels)
    cut = int(0.75 * len(rels))
    return rels[:cut], rels[cut:]


def relations_by_source(rels: list[tuple[int, int, int]]):
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for rel in rels:
        grouped.setdefault(rel[0], []).append(rel)
    return grouped


def noisy_feature(world: World, idx: int, scale: float = 0.08):
    return F.normalize(
        world.features[idx] + scale * torch.randn_like(world.features[idx]), dim=-1
    )


def query_feature(world: World, idx: int, masked: bool):
    """Current local nonlinguistic evidence.

    For the decisive ambiguity probe this is exactly zero, so two concepts that
    share a token are locally indistinguishable for the same relation type.
    """
    if masked:
        return torch.zeros_like(world.features[idx])
    return noisy_feature(world, idx)


def one(x: int):
    return torch.tensor(x, dtype=torch.long)


class ObservationEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.token = nn.Embedding(cfg.token_count, cfg.latent_dim)
        self.feature = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.relation = nn.Embedding(cfg.relation_types, cfg.latent_dim)
        self.mix = nn.Sequential(
            nn.Linear(cfg.latent_dim * 3, cfg.latent_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.latent_dim),
        )

    def forward(self, token, feature, relation_type):
        return self.mix(
            torch.cat(
                [
                    self.token(token),
                    self.feature(feature),
                    self.relation(relation_type),
                ],
                dim=-1,
            )
        )


class BasePredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ObservationEncoder(cfg)
        self.head = nn.Linear(cfg.latent_dim, cfg.feature_dim)

    def forward(self, token, feature, relation_type):
        z = self.encoder(token, feature, relation_type)
        return F.normalize(self.head(z), dim=-1), z


class PersistentRelationalMemory(nn.Module):
    """Content-addressed slots with no concept-specific addresses."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.keys = nn.Parameter(
            torch.randn(cfg.memory_slots, cfg.latent_dim) / math.sqrt(cfg.latent_dim)
        )
        self.message = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.gate = nn.Linear(cfg.latent_dim * 2, 1)

    def empty(self, device=None):
        return torch.zeros(
            self.cfg.memory_slots, self.cfg.latent_dim, device=device
        )

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


class MemoryAidedPredictor(nn.Module):
    """Frozen base predictor plus a learned persistent-memory residual.

    Because the correction is multiplied by ||recalled||, an empty memory
    produces exactly the frozen base prediction even after memory training.
    """

    def __init__(self, cfg: Config, base: BasePredictor):
        super().__init__()
        self.cfg = cfg
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad = False

        self.memory = PersistentRelationalMemory(cfg)
        self.aid = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.feature_dim, bias=False),
        )

    def predict(self, state, token, feature, relation_type):
        base_pred, z = self.base(token, feature, relation_type)
        recalled, weights = self.memory.read(state, z)
        correction = self.aid(torch.cat([z, recalled], -1))
        # Exact base behavior when recalled == 0.
        scale = recalled.norm().clamp(max=1.0)
        out = F.normalize(base_pred + scale * correction, dim=-1)
        return out, base_pred, z, recalled, weights

    def write_observation(
        self,
        state,
        token: int,
        source_feature: torch.Tensor,
        relation_type: int,
        related_feature: torch.Tensor,
    ):
        # The encoder is frozen, so the memory has to learn how to organize
        # evidence instead of changing the underlying base representation.
        with torch.no_grad():
            z = self.base.encoder(one(token), source_feature, one(relation_type))
            related = self.base.encoder.feature(related_feature)
        return self.memory.write(state, z, related)


def train_base(model, world, train_rel, cfg):
    """Train the common base predictor before any memory is added."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model.train()
    ambiguous = ambiguous_primary_tokens(world)

    for _ in range(cfg.epochs):
        random.shuffle(train_rel)
        for src, r, dst in train_rel[: cfg.episodes_per_epoch]:
            # The base sees the same difficult masked-query distribution used
            # later by the memory system, not just easy full-feature examples.
            primary = int(world.tokens[src])
            masked = primary in ambiguous and random.random() < 0.5
            token = primary if masked else sample_token(world, src)
            pred, _ = model(
                one(token), query_feature(world, src, masked), one(r)
            )
            loss = 1.0 - F.cosine_similarity(
                pred[None], world.features[dst][None]
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


def build_context_state(
    model: MemoryAidedPredictor,
    world: World,
    context_relations: list[tuple[int, int, int]],
    context_fraction: float = 1.0,
    differentiable: bool = False,
):
    """Build state from prior observations without exposing concept ids to model.

    The evaluator chooses episode membership using hidden ids. The network only
    receives the resulting token/feature/relation observations.
    """
    state = model.memory.empty()
    keep = int(round(len(context_relations) * context_fraction))
    if context_fraction > 0.0 and context_relations:
        keep = max(1, keep)
    selected = context_relations[:keep]

    for src, r, dst in selected:
        token = sample_token(world, src)
        source_feature = noisy_feature(world, src)
        related_feature = noisy_feature(world, dst)
        state = model.write_observation(
            state, token, source_feature, r, related_feature
        )
        if not differentiable:
            state = state.detach()
    return state


def train_memory(model, world, train_rel, cfg):
    """Train memory on multi-observation episodes.

    The query loss backpropagates through all writes that built its context
    state. This fixes the earlier one-step detach bug: the writer is now
    directly rewarded when a stored relation helps a later prediction.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr)
    grouped = relations_by_source(train_rel)
    sources = [src for src, rels in grouped.items() if len(rels) >= 2]

    model.train()
    for _ in range(cfg.epochs):
        random.shuffle(sources)
        used = 0
        for src in sources:
            rels = grouped[src][:]
            random.shuffle(rels)
            query_rel = rels[0]
            context = rels[1:]
            if not context:
                continue

            # Do not detach while constructing the episode. The later query
            # loss must train the earlier writes.
            state = build_context_state(
                model, world, context, context_fraction=1.0, differentiable=True
            )

            q_src, q_r, q_dst = query_rel
            masked = is_ambiguous_concept(world, q_src)
            token = int(world.tokens[q_src]) if masked else sample_token(world, q_src)
            pred, _, _, _, _ = model.predict(
                state,
                one(token),
                query_feature(world, q_src, masked),
                one(q_r),
            )
            pred_loss = 1.0 - F.cosine_similarity(
                pred[None], world.features[q_dst][None]
            ).mean()
            loss = pred_loss + cfg.sparsity_weight * state.abs().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            used += 1
            if used >= cfg.episodes_per_epoch:
                break


def evaluate_base(model, world, rels, masked_ambiguity: bool = False):
    model.eval()
    scores = []
    ambiguous = ambiguous_primary_tokens(world)
    with torch.no_grad():
        for src, r, dst in rels:
            primary = int(world.tokens[src])
            masked = masked_ambiguity and primary in ambiguous
            pred, _ = model(
                one(primary),
                torch.zeros_like(world.features[src]) if masked else world.features[src],
                one(r),
            )
            scores.append(
                float(F.cosine_similarity(pred[None], world.features[dst][None]))
            )
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_memory(
    model,
    world,
    train_rel,
    test_rel,
    context_fraction: float = 1.0,
    masked_ambiguity: bool = False,
    ambiguous_only: bool = False,
):
    model.eval()
    grouped = relations_by_source(train_rel)
    ambiguous = ambiguous_primary_tokens(world)
    scores = []

    with torch.no_grad():
        for src, r, dst in test_rel:
            primary = int(world.tokens[src])
            is_amb = primary in ambiguous
            if ambiguous_only and not is_amb:
                continue

            state = build_context_state(
                model,
                world,
                grouped.get(src, []),
                context_fraction=context_fraction,
            )
            masked = masked_ambiguity and is_amb
            pred, _, _, _, _ = model.predict(
                state,
                one(primary),
                torch.zeros_like(world.features[src]) if masked else world.features[src],
                one(r),
            )
            scores.append(
                float(F.cosine_similarity(pred[None], world.features[dst][None]))
            )
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_base_ambiguous_only(model, world, rels):
    ambiguous = ambiguous_primary_tokens(world)
    subset = [rel for rel in rels if int(world.tokens[rel[0]]) in ambiguous]
    return evaluate_base(model, world, subset, masked_ambiguity=True)


def homonym_memory_separation(model, world, train_rel):
    """Compare persistent states for same-token/different-family concepts."""
    grouped = relations_by_source(train_rel)
    by_token: dict[int, list[int]] = {}
    for concept, token in enumerate(world.tokens.tolist()):
        by_token.setdefault(int(token), []).append(concept)

    vals = []
    model.eval()
    with torch.no_grad():
        for ids in by_token.values():
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    if int(world.family[a]) == int(world.family[b]):
                        continue
                    if not grouped.get(a) or not grouped.get(b):
                        continue
                    sa = build_context_state(model, world, grouped[a])
                    sb = build_context_state(model, world, grouped[b])
                    vals.append(
                        float(
                            F.cosine_similarity(
                                sa.flatten()[None], sb.flatten()[None]
                            )
                        )
                    )
    return float(np.mean(vals)) if vals else float("nan")


def alias_boundary_consistency(model, world, train_rel):
    """Same underlying concept, two different surface tokens."""
    grouped = relations_by_source(train_rel)
    pred_cos = []
    read_cos = []

    model.eval()
    with torch.no_grad():
        for src in range(len(world.tokens)):
            alias = int(world.aliases[src])
            if alias < 0 or not grouped.get(src):
                continue
            context = grouped[src]
            state = build_context_state(model, world, context)
            _, r, _ = context[0]

            p1, _, _, _, w1 = model.predict(
                state, one(int(world.tokens[src])), world.features[src], one(r)
            )
            p2, _, _, _, w2 = model.predict(
                state, one(alias), world.features[src], one(r)
            )
            pred_cos.append(float(F.cosine_similarity(p1[None], p2[None])))
            read_cos.append(float(F.cosine_similarity(w1[None], w2[None])))

    return {
        "prediction_cosine": float(np.mean(pred_cos)) if pred_cos else float("nan"),
        "read_weight_cosine": float(np.mean(read_cos)) if read_cos else float("nan"),
    }


def remap_world(world: World, permutation: torch.Tensor) -> World:
    aliases = world.aliases.clone()
    valid = aliases >= 0
    aliases[valid] = permutation[aliases[valid]]
    return World(
        features=world.features,
        family=world.family,
        tokens=permutation[world.tokens],
        aliases=aliases,
        relations=world.relations,
    )


def boundary_remap_probe(model, world, train_rel, test_rel, cfg):
    """Permute vocabulary and relearn only the token embedding boundary."""
    g = torch.Generator().manual_seed(cfg.seed + 1000)
    permutation = torch.randperm(cfg.token_count, generator=g)
    remapped_world = remap_world(world, permutation)

    probe = copy.deepcopy(model)
    for p in probe.parameters():
        p.requires_grad = False

    token_weight = probe.base.encoder.token.weight
    with torch.no_grad():
        token_weight.normal_(mean=0.0, std=1.0 / math.sqrt(cfg.latent_dim))
    token_weight.requires_grad = True

    before = evaluate_memory(
        probe,
        remapped_world,
        train_rel,
        test_rel,
        context_fraction=1.0,
        masked_ambiguity=True,
        ambiguous_only=True,
    )

    opt = torch.optim.AdamW([token_weight], lr=cfg.lr)
    grouped = relations_by_source(train_rel)
    sources = [src for src, rels in grouped.items() if len(rels) >= 2]

    probe.train()
    for _ in range(cfg.remap_epochs):
        random.shuffle(sources)
        used = 0
        for src in sources:
            rels = grouped[src][:]
            random.shuffle(rels)
            query_rel = rels[0]
            context = rels[1:]
            state = build_context_state(
                probe,
                remapped_world,
                context,
                context_fraction=1.0,
                differentiable=True,
            )
            q_src, q_r, q_dst = query_rel
            masked = is_ambiguous_concept(remapped_world, q_src)
            token = (
                int(remapped_world.tokens[q_src])
                if masked
                else sample_token(remapped_world, q_src)
            )
            pred, _, _, _, _ = probe.predict(
                state,
                one(token),
                query_feature(remapped_world, q_src, masked),
                one(q_r),
            )
            loss = 1.0 - F.cosine_similarity(
                pred[None], remapped_world.features[q_dst][None]
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            used += 1
            if used >= cfg.episodes_per_epoch:
                break

    after = evaluate_memory(
        probe,
        remapped_world,
        train_rel,
        test_rel,
        context_fraction=1.0,
        masked_ambiguity=True,
        ambiguous_only=True,
    )
    return {
        "before_boundary_relearn_cosine": before,
        "after_boundary_relearn_cosine": after,
        "trainable_parameters": int(token_weight.numel()),
        "frozen_parameters": int(
            sum(p.numel() for p in probe.parameters()) - token_weight.numel()
        ),
    }


def run(cfg: Config):
    seed_all(cfg.seed)
    world = make_world(cfg)
    train_rel, test_rel = split_relations(world, cfg.seed)

    base = BasePredictor(cfg)
    train_base(base, world, train_rel, cfg)

    memory = MemoryAidedPredictor(cfg, base)
    train_memory(memory, world, train_rel, cfg)

    # Context-removal curve on the decisive same-token + masked-feature cases.
    context_curve = {}
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        context_curve[str(fraction)] = evaluate_memory(
            memory,
            world,
            train_rel,
            test_rel,
            context_fraction=fraction,
            masked_ambiguity=True,
            ambiguous_only=True,
        )

    result = {
        "experiment": "E41 — Relational Concept Memory",
        "seed": cfg.seed,
        "question": (
            "Can persistent relational context aid a frozen predictor when "
            "surface symbols and current local evidence are insufficient?"
        ),
        "full_feature_control": {
            "base_test_cosine": evaluate_base(base, world, test_rel),
            "memory_test_cosine": evaluate_memory(
                memory, world, train_rel, test_rel, context_fraction=1.0
            ),
            "memory_zero_context_cosine": evaluate_memory(
                memory, world, train_rel, test_rel, context_fraction=0.0
            ),
        },
        "genuine_ambiguity_probe": {
            "base_masked_homonym_cosine": evaluate_base_ambiguous_only(
                base, world, test_rel
            ),
            "memory_masked_homonym_cosine": context_curve["1.0"],
            "context_curve": context_curve,
        },
        "persistent_state_homonym_cosine": homonym_memory_separation(
            memory, world, train_rel
        ),
        "same_concept_alias_probe": alias_boundary_consistency(
            memory, world, train_rel
        ),
        "token_remap_boundary_probe": boundary_remap_probe(
            memory, world, train_rel, test_rel, cfg
        ),
        "persistent_scalar_slots": cfg.memory_slots * cfg.latent_dim,
        "train_relations": len(train_rel),
        "test_relations": len(test_rel),
        "ambiguous_primary_tokens": len(ambiguous_primary_tokens(world)),
        "aliased_concepts": int((world.aliases >= 0).sum()),
        "notes": [
            "Hidden concept ids are simulator/evaluator-only episode selectors.",
            "The memory-aided model contains an exact frozen copy of the trained base predictor.",
            "With empty memory, the residual correction is mathematically zero.",
            "Ambiguous homonym queries use the same token and a zero feature vector; relational context is the only concept-specific evidence available.",
            "Memory query loss backpropagates through the prior writes that built the episode state.",
            "Aliases give the same underlying concept two different surface labels.",
            "The vocabulary-remap probe retrains only the token embedding table while freezing the rest of the learned representation and memory system.",
            "Persistent-state cost is still reported in scalar slots; exact bit/entropy cost remains a separate compression measurement.",
        ],
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", default="artifacts/E41_relational_concept_memory.json"
    )
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    cfg = (
        Config(epochs=20, episodes_per_epoch=48, remap_epochs=10)
        if args.quick
        else Config()
    )

    result = run(cfg)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
