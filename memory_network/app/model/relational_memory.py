from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalMemory(nn.Module):
    """Derived neural memory: persistent identity + mutable relational context."""
    def __init__(self, embed_dim=16, hidden_dim=32, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_dim = embed_dim
        self.identities = nn.ParameterDict()
        self.relations = nn.ParameterDict()
        self.context = {}
        self.interaction = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim), nn.Tanh())
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), nn.Sigmoid())
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))

    @staticmethod
    def _key(value):
        return "r_" + "_".join(f"{ord(c):x}" for c in value)

    def register_entity(self, entity_id: int):
        key = str(entity_id)
        if key not in self.identities:
            self.identities[key] = nn.Parameter(torch.randn(self.embed_dim) * 0.05)
            self.context[entity_id] = torch.zeros(self.embed_dim)

    def register_relation(self, relation: str):
        key = self._key(relation)
        if key not in self.relations:
            self.relations[key] = nn.Parameter(torch.randn(self.embed_dim) * 0.05)

    def identity(self, entity_id):
        self.register_entity(entity_id)
        return self.identities[str(entity_id)]

    def relation(self, relation):
        self.register_relation(relation)
        return self.relations[self._key(relation)]

    def state(self, entity_id):
        self.register_entity(entity_id)
        return self.identity(entity_id) + self.context[entity_id]

    def _features(self, source_id, relation, target_id):
        return torch.cat([self.state(source_id), self.relation(relation), self.state(target_id)])

    def score(self, source_id, relation, target_id):
        return self.predictor(self._features(source_id, relation, target_id)).squeeze(-1)

    def observe(self, source_id, relation, target_id):
        """Apply a directional transition to both participants; weights do not change."""
        features = self._features(source_id, relation, target_id)
        message = self.interaction(features)
        gate = self.gate(features)
        with torch.no_grad():
            self.context[source_id] = (1-gate) * self.context[source_id] + gate * message
            reverse = torch.cat([self.state(target_id), self.relation(relation), self.state(source_id)])
            reverse_message = self.interaction(reverse)
            reverse_gate = self.gate(reverse)
            self.context[target_id] = (1-reverse_gate) * self.context[target_id] + reverse_gate * reverse_message

    def learn_edge(self, source_id, relation, target_id, negative_id, optimizer):
        positive = self.score(source_id, relation, target_id)
        negative = self.score(source_id, relation, negative_id)
        loss = F.softplus(-positive) + F.softplus(negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.observe(source_id, relation, target_id)
        return float(loss.detach())
