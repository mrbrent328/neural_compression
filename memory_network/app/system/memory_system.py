from __future__ import annotations
import random
from pathlib import Path
import torch
from ..memory.database import MemoryDB
from ..model.relational_memory import RelationalMemory

class NeuralMemorySystem:
    def __init__(self, db_path="data/memory.db", embed_dim=16, seed=42, checkpoint_path=None):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = MemoryDB(db_path)
        self.model = RelationalMemory(embed_dim=embed_dim, seed=seed)
        self.checkpoint_path = checkpoint_path or str(Path(db_path).with_suffix(".pt"))
        self.rng = random.Random(seed)
        loaded = self._load_checkpoint()
        if not loaded:
            self._replay()

    def _load_checkpoint(self):
        path = Path(self.checkpoint_path)
        if not path.exists():
            return False
        for entity in self.db.all_entities():
            self.model.register_entity(entity.id)
        for obs in self.db.all_observations():
            self.model.register_relation(obs.relation)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.context = {int(k): v.clone() for k, v in checkpoint.get("context", {}).items()}
        return True

    def _save_checkpoint(self):
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(),
                    "context": {str(k): v.detach().clone() for k,v in self.model.context.items()}}, path)

    def _replay(self):
        for entity in self.db.all_entities():
            self.model.register_entity(entity.id)
        for obs in self.db.all_observations():
            self.model.register_relation(obs.relation)
            self.model.observe(obs.source_id, obs.relation, obs.target_id)

    def remember(self, source, relation, target, source_type="entity", target_type="entity"):
        obs = self.db.add_observation(source, relation, target, source_type, target_type)
        self.model.register_entity(obs.source_id)
        self.model.register_entity(obs.target_id)
        self.model.register_relation(relation)
        self.model.observe(obs.source_id, relation, obs.target_id)
        self._save_checkpoint()
        return obs

    def inspect(self, name):
        entity = self.db.entity_by_name(name)
        if not entity:
            return None
        edges = []
        for row in self.db.observations_for(entity.id):
            _, source, relation, target, _ = row
            other_id = target if source == entity.id else source
            other = self.db.entity_by_id(other_id)
            edges.append(("→" if source == entity.id else "←", relation, other.name if other else str(other_id)))
        return {"entity": entity, "edges": edges,
                "identity": self.model.identity(entity.id).detach().tolist(),
                "context": self.model.context[entity.id].detach().tolist()}

    def train(self, steps=500):
        entities = self.db.all_entities()
        observations = self.db.all_observations()
        if len(entities) < 2 or not observations:
            return {"steps": 0, "message": "Need at least two entities and one observation."}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        losses = []
        for _ in range(steps):
            obs = self.rng.choice(observations)
            candidates = [e for e in entities if e.id not in (obs.source_id, obs.target_id)]
            if not candidates:
                continue
            neg = self.rng.choice(candidates)
            losses.append(self.model.learn_edge(obs.source_id, obs.relation, obs.target_id, neg.id, optimizer))
        self._save_checkpoint()
        return {"steps": len(losses), "mean_loss": sum(losses)/len(losses) if losses else None}

    def query(self, name, relation=None, top_k=5):
        entity = self.db.entity_by_name(name)
        if not entity:
            return []
        relations = [relation] if relation else sorted({o.relation for o in self.db.all_observations()})
        candidates = [e for e in self.db.all_entities() if e.id != entity.id]
        results = []
        for rel in relations:
            for candidate in candidates:
                results.append((float(self.model.score(entity.id, rel, candidate.id).detach()), rel, candidate.name))
        return sorted(results, reverse=True)[:top_k]

    def close(self):
        self.db.close()
