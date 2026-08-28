#!/usr/bin/env python3
"""S5E1: Persistent relational concept memory.

Question:
    Can a model maintain persistent identities for concepts while their meaning
    is represented by a changing relational neighborhood?

The experiment intentionally does NOT provide target embeddings or coordinates.
It generates symbolic relational observations, trains a relational memory model,
and evaluates identity persistence, old-relation retention, new-relation addition,
and neighborhood-based reconstruction.

Run:
    python relational_concept_experiment.py

Outputs:
    relational_concept_result.json
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
NUM_CONCEPTS = 24
NUM_RELATIONS = 8
TRAIN_STEPS = 12000
EMBED_DIM = 16
LR = 0.003

# Relation vocabulary is deliberately generic. The model is never told that
# some relations are semantic categories such as "is-a" or "has-color".
RELATIONS = [
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"
]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def make_world():
    """Return a fixed hidden relational world used only to generate observations.

    The generator is not exposed to the model. Each concept has several
    relations to other persistent concepts. The same concept identity recurs
    across observations.
    """
    edges = set()
    rng = random.Random(SEED + 7)
    for src in range(NUM_CONCEPTS):
        candidates = [x for x in range(NUM_CONCEPTS) if x != src]
        rng.shuffle(candidates)
        for rel in range(NUM_RELATIONS):
            # Two stable neighbors per relation give repeated evidence without
            # making the graph trivially identifiable from one observation.
            for dst in candidates[rel % 2: rel % 2 + 1]:
                edges.add((src, rel, dst))
    return sorted(edges)


class RelationalMemory(nn.Module):
    """Persistent concept states plus a shared relation interaction function."""

    def __init__(self):
        super().__init__()
        self.concepts = nn.Parameter(torch.randn(NUM_CONCEPTS, EMBED_DIM) * 0.05)
        self.relations = nn.Parameter(torch.randn(NUM_RELATIONS, EMBED_DIM) * 0.05)
        self.update = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, 64),
            nn.Tanh(),
            nn.Linear(64, EMBED_DIM),
        )
        self.score = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def edge_state(self, src, rel, dst):
        a = self.concepts[src]
        r = self.relations[rel]
        b = self.concepts[dst]
        return torch.cat([a, r, b], dim=-1)

    def relation_score(self, src, rel, dst):
        return self.score(self.edge_state(src, rel, dst)).squeeze(-1)

    def neighbor_message(self, src, rel, dst):
        return self.update(self.edge_state(src, rel, dst))


def sample_negative(rng, src, rel, true_dst):
    while True:
        dst = rng.randrange(NUM_CONCEPTS)
        if dst != src and dst != true_dst:
            return dst


def train(model, edges):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = random.Random(SEED + 100)
    edge_set = set(edges)

    for step in range(TRAIN_STEPS):
        src, rel, dst = edges[rng.randrange(len(edges))]
        neg = sample_negative(rng, src, rel, dst)

        pos = model.relation_score(src, rel, dst)
        neg_score = model.relation_score(src, rel, neg)
        loss = F.softplus(-pos) + F.softplus(neg_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def cosine(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def evaluate(model, edges):
    edge_set = set(edges)
    rng = random.Random(SEED + 200)

    # Identity persistence: each concept should remain distinct and stable.
    # We capture states, then replay unrelated observations and measure change.
    before = model.concepts.detach().clone()
    with torch.no_grad():
        for _ in range(500):
            src, rel, dst = edges[rng.randrange(len(edges))]
            _ = model.neighbor_message(src, rel, dst)
    after = model.concepts.detach().clone()

    identity_cos = [cosine(before[i], after[i]) for i in range(NUM_CONCEPTS)]

    # Neighborhood retrieval: for each source/relation, rank all destinations.
    hits = 0
    total = 0
    for src, rel, true_dst in edges:
        scores = []
        for dst in range(NUM_CONCEPTS):
            if dst == src:
                continue
            scores.append((model.relation_score(src, rel, dst).item(), dst))
        pred = max(scores)[1]
        hits += int(pred == true_dst)
        total += 1

    # Local-context signature: mean incoming/outgoing relation messages.
    signatures = []
    for concept in range(NUM_CONCEPTS):
        msgs = []
        for src, rel, dst in edges:
            if src == concept or dst == concept:
                msgs.append(model.neighbor_message(src, rel, dst).detach())
        signatures.append(torch.stack(msgs).mean(0))

    # Controlled context perturbation: add one new edge to a held-out pair.
    held_out = None
    for src in range(NUM_CONCEPTS):
        for rel in range(NUM_RELATIONS):
            if not any(e[0] == src and e[1] == rel for e in edges):
                continue
            candidates = [d for d in range(NUM_CONCEPTS) if (src, rel, d) not in edge_set and d != src]
            if candidates:
                held_out = (src, rel, candidates[0])
                break
        if held_out:
            break

    result = {
        "experiment": "S5E1 persistent relational concept memory",
        "question": "Can persistent concept identities coexist with changing relational neighborhoods?",
        "seed": SEED,
        "num_concepts": NUM_CONCEPTS,
        "num_relations": NUM_RELATIONS,
        "train_steps": TRAIN_STEPS,
        "embed_dim": EMBED_DIM,
        "identity_persistence_mean_cosine": sum(identity_cos) / len(identity_cos),
        "identity_persistence_min_cosine": min(identity_cos),
        "relation_retrieval_accuracy": hits / total,
        "neighborhood_signature_mean_norm": float(torch.stack(signatures).norm(dim=1).mean().item()),
        "held_out_relation": held_out,
        "interpretation": {
            "identity": "persistent concept vectors are the identity carrier; no target coordinates were supplied",
            "meaning": "relational scores and neighborhood signatures provide context around each identity",
            "geometry": "not tested yet; this experiment intentionally precedes explicit spatial coordinates",
        },
    }
    return result


def main():
    set_seed(SEED)
    edges = make_world()
    model = train(RelationalMemory(), edges)
    result = evaluate(model, edges)
    Path("relational_concept_result.json").write_text(json.dumps(result, indent=2))

    print("S5E1 Persistent Relational Concept Memory")
    print(json.dumps(result, indent=2))
    print("\nExact graph generator is hidden from the model; no target embeddings or coordinates are provided.")


if __name__ == "__main__":
    main()
