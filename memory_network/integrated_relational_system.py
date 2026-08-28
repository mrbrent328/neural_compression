#!/usr/bin/env python3
"""S6E1 — Integrated Persistent Relational System.

First complete system combining the useful mechanisms from the earlier tests:

  persistent identity + mutable relational context + shared interaction +
  online observation + next-relation prediction.

No target embeddings or coordinates are supplied. The hidden world only
provides symbolic relational observations. The model must learn to predict
relations while keeping a persistent identity carrier for each entity and a
mutable context state whose value depends on its relational neighborhood.

The experiment deliberately separates identity from meaning:
  identity[e]  = persistent learned entity identity
  context[e]   = mutable state updated by observations
  state[e]     = identity[e] + context[e]

This is not intended as a neuroscience claim. It is an engineering test of
whether a small relational memory system can produce the proposed behavior.

Run:
    python3 integrated_relational_system.py

Output:
    integrated_relational_result.json
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
NUM_ENTITIES = 24
NUM_RELATIONS = 6
EMBED_DIM = 24
HIDDEN_DIM = 48
TRAIN_EPOCHS = 80
LR = 0.003
OBSERVATIONS_PER_EPOCH = 180


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)


def make_world():
    """Create a fixed symbolic world hidden from the model."""
    rng = random.Random(SEED + 7)
    edges = []

    # Each relation has a different deterministic neighborhood pattern.
    # The model receives only (source, relation, destination) observations.
    for src in range(NUM_ENTITIES):
        for rel in range(NUM_RELATIONS):
            step = rel + 1
            dst = (src + step) % NUM_ENTITIES
            edges.append((src, rel, dst))

            # A second neighbor makes the neighborhood nontrivial.
            dst2 = (src + step * 3 + (src % 3)) % NUM_ENTITIES
            if dst2 != src and dst2 != dst:
                edges.append((src, rel, dst2))

    rng.shuffle(edges)
    return edges


class RelationalSystem(nn.Module):
    """Persistent identities with mutable relational context."""

    def __init__(self):
        super().__init__()

        # These are the persistent identity carriers. They are never replaced
        # by observations; learning changes them only through optimization.
        self.identity = nn.Parameter(
            torch.randn(NUM_ENTITIES, EMBED_DIM) * 0.05
        )

        # Mutable relational neighborhood state. This is the part that changes
        # as observations arrive.
        self.context = nn.Parameter(
            torch.zeros(NUM_ENTITIES, EMBED_DIM)
        )

        self.relation = nn.Parameter(
            torch.randn(NUM_RELATIONS, EMBED_DIM) * 0.05
        )

        self.interaction = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
            nn.Tanh(),
        )

        self.gate = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, EMBED_DIM),
            nn.Sigmoid(),
        )

        self.score = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def state(self, entity):
        return self.identity[entity] + self.context[entity]

    def edge_features(self, src, rel, dst):
        return torch.cat([
            self.state(src),
            self.relation[rel],
            self.state(dst),
        ], dim=-1)

    def relation_score(self, src, rel, dst):
        return self.score(self.edge_features(src, rel, dst)).squeeze(-1)

    def observe(self, src, rel, dst):
        """Return context updates for the observed edge.

        Updates are returned rather than immediately written so the caller can
        decide whether the update is part of a differentiable training step.
        """
        features = self.edge_features(src, rel, dst)
        message = self.interaction(features)
        gate = self.gate(features)

        src_old = self.context[src]
        dst_old = self.context[dst]

        src_new = (1.0 - gate) * src_old + gate * message
        dst_new = (1.0 - gate) * dst_old + gate * message
        return src_new, dst_new

    def apply_observation(self, src, rel, dst):
        """Apply one observation as a state transition without changing weights."""
        with torch.no_grad():
            src_new, dst_new = self.observe(src, rel, dst)
            self.context[src].copy_(src_new)
            self.context[dst].copy_(dst_new)


def sample_negative(rng, src, rel, true_dst):
    while True:
        dst = rng.randrange(NUM_ENTITIES)
        if dst != src and dst != true_dst:
            return dst


def train_system(model, edges):
    """Train relation scoring while replaying observations online."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(SEED + 100)

    for epoch in range(TRAIN_EPOCHS):
        model.context.data.zero_()
        total_loss = 0.0

        for _ in range(OBSERVATIONS_PER_EPOCH):
            src, rel, dst = edges[rng.randrange(len(edges))]
            neg = sample_negative(rng, src, rel, dst)

            pos = model.relation_score(src, rel, dst)
            neg_score = model.relation_score(src, rel, neg)
            loss = F.softplus(-pos) + F.softplus(neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # The observation is an online state transition. It is deliberately
            # outside the optimizer graph so persistent context does not become
            # a hidden parameter update.
            model.apply_observation(src, rel, dst)
            total_loss += float(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1:3d} | mean loss {total_loss / OBSERVATIONS_PER_EPOCH:.4f}")


def relation_accuracy(model, edges):
    hits = 0
    total = 0

    for src, rel, true_dst in edges:
        candidates = [d for d in range(NUM_ENTITIES) if d != src]
        scores = [
            (model.relation_score(src, rel, dst).item(), dst)
            for dst in candidates
        ]
        pred = max(scores)[1]
        hits += int(pred == true_dst)
        total += 1

    return hits / total


def context_change_test(model, edges):
    """Show that relational observations can change context without replacing identity."""
    rng = random.Random(SEED + 200)

    identity_before = model.identity.detach().clone()
    context_before = model.context.detach().clone()

    for _ in range(120):
        src, rel, dst = edges[rng.randrange(len(edges))]
        model.apply_observation(src, rel, dst)

    identity_after = model.identity.detach().clone()
    context_after = model.context.detach().clone()

    identity_cos = F.cosine_similarity(
        identity_before,
        identity_after,
        dim=-1,
    )
    context_delta = torch.linalg.vector_norm(
        context_after - context_before,
        dim=-1,
    )

    return {
        "identity_mean_cosine": float(identity_cos.mean().item()),
        "identity_min_cosine": float(identity_cos.min().item()),
        "context_mean_delta": float(context_delta.mean().item()),
        "context_max_delta": float(context_delta.max().item()),
    }


def neighborhood_signature_test(model, edges):
    signatures = []

    for entity in range(NUM_ENTITIES):
        related = []
        for src, rel, dst in edges:
            if src == entity or dst == entity:
                related.append(model.relation[rel].detach())

        if related:
            signatures.append(torch.stack(related).mean(0))
        else:
            signatures.append(torch.zeros(EMBED_DIM))

    signatures = torch.stack(signatures)
    norms = torch.linalg.vector_norm(signatures, dim=-1)

    return {
        "mean_norm": float(norms.mean().item()),
        "min_norm": float(norms.min().item()),
        "max_norm": float(norms.max().item()),
    }


def evaluate(model, edges):
    model.eval()

    accuracy = relation_accuracy(model, edges)
    dynamics = context_change_test(model, edges)
    neighborhood = neighborhood_signature_test(model, edges)

    # Compare two entities with different learned neighborhoods. This is only
    # an observation of the learned representation, not a target geometry.
    pair_cosines = []
    for a in range(NUM_ENTITIES):
        for b in range(a + 1, NUM_ENTITIES):
            pair_cosines.append(
                F.cosine_similarity(
                    model.state(a).unsqueeze(0),
                    model.state(b).unsqueeze(0),
                ).item()
            )

    return {
        "experiment": "S6E1 integrated persistent relational system",
        "question": "Can persistent identities, changing relational context, shared interaction, and online learning coexist in one inspectable system?",
        "seed": SEED,
        "num_entities": NUM_ENTITIES,
        "num_relations": NUM_RELATIONS,
        "embed_dim": EMBED_DIM,
        "train_epochs": TRAIN_EPOCHS,
        "observations_per_epoch": OBSERVATIONS_PER_EPOCH,
        "relation_retrieval_accuracy": accuracy,
        "identity_context_dynamics": dynamics,
        "neighborhood_signature": neighborhood,
        "state_pairwise_cosine_mean": sum(pair_cosines) / len(pair_cosines),
        "state_pairwise_cosine_min": min(pair_cosines),
        "state_pairwise_cosine_max": max(pair_cosines),
        "interpretation": {
            "identity": "persistent identity parameters are separate from mutable context states",
            "meaning": "context is updated from observed relational edges through one shared interaction mechanism",
            "learning": "relation prediction supplies the task pressure; observations drive online state transitions",
            "geometry": "no coordinates or geometric targets are supplied; geometry remains an open follow-up question",
        },
    }


def main():
    set_seed()
    edges = make_world()
    model = RelationalSystem()

    print("S6E1 Integrated Persistent Relational System")
    print(f"entities={NUM_ENTITIES}, relations={NUM_RELATIONS}, embed_dim={EMBED_DIM}")
    print("Training...")

    train_system(model, edges)
    result = evaluate(model, edges)

    Path("integrated_relational_result.json").write_text(
        json.dumps(result, indent=2)
    )

    print("\nResult")
    print(json.dumps(result, indent=2))
    print("\nNo target embeddings or coordinates were provided.")


if __name__ == "__main__":
    main()
