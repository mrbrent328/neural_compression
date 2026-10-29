from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Concept:
    name: str
    kind: str = "concept"
    confidence: float = 1.0
    observations: int = 0
    aliases: List[str] = field(default_factory=list)


@dataclass
class Relation:
    src: str
    rel: str
    dst: str
    strength: float = 1.0
    evidence_count: int = 1
    contradiction_count: int = 0
    provenance: List[str] = field(default_factory=list)
    derived: bool = False

    @property
    def confidence(self) -> float:
        support = self.evidence_count
        oppose = self.contradiction_count
        return support / max(1, support + oppose)


@dataclass
class TraceStep:
    src: str
    rel: str
    dst: str
    activation: float
    confidence: float
    provenance: List[str]


@dataclass
class ThoughtState:
    seeds: Dict[str, float]
    activation: Dict[str, float]
    traces: List[TraceStep]
    conclusions: List[Relation]


class RelationalCognitiveEngine:
    """Small inspectable persistent relational world model.

    v0 intentionally uses symbolic identifiers and explicit graph operations.
    The point is to test the cognitive substrate independently from a large
    language model. Neural/text/vision adapters can later map into these nodes.
    """

    NEGATIONS = {
        "is": "is_not",
        "is_not": "is",
        "can": "cannot",
        "cannot": "can",
        "likes": "dislikes",
        "dislikes": "likes",
        "safe_with": "unsafe_with",
        "unsafe_with": "safe_with",
    }

    TRANSITIVE = {"is_a", "part_of"}

    def __init__(self) -> None:
        self.concepts: Dict[str, Concept] = {}
        self.relations: Dict[Tuple[str, str, str], Relation] = {}
        self.outgoing: Dict[str, set[Tuple[str, str, str]]] = defaultdict(set)
        self.incoming: Dict[str, set[Tuple[str, str, str]]] = defaultdict(set)
        self.event_counter = 0

    # ---------- persistent structural learning ----------

    def ensure_concept(self, name: str, kind: str = "concept") -> Concept:
        key = self._norm(name)
        if key not in self.concepts:
            self.concepts[key] = Concept(name=key, kind=kind)
        return self.concepts[key]

    def observe_relation(
        self,
        src: str,
        rel: str,
        dst: str,
        *,
        provenance: Optional[str] = None,
        strength: float = 1.0,
        derived: bool = False,
    ) -> Relation:
        src, rel, dst = self._norm(src), self._norm(rel), self._norm(dst)
        self.ensure_concept(src)
        self.ensure_concept(dst)
        self.event_counter += 1
        evidence = provenance or f"observation:{self.event_counter}"

        key = (src, rel, dst)
        opposite = self.NEGATIONS.get(rel)

        # Contradiction alters existing model state rather than appending an
        # unrelated memory entry.
        if opposite:
            opposite_key = (src, opposite, dst)
            if opposite_key in self.relations:
                old = self.relations[opposite_key]
                old.contradiction_count += 1
                old.strength *= 0.75
                old.provenance.append(f"contradicted_by:{evidence}")

        if key in self.relations:
            edge = self.relations[key]
            edge.evidence_count += 1
            edge.strength = min(8.0, edge.strength + 0.25 * strength)
            if evidence not in edge.provenance:
                edge.provenance.append(evidence)
        else:
            edge = Relation(
                src=src,
                rel=rel,
                dst=dst,
                strength=strength,
                provenance=[evidence],
                derived=derived,
            )
            self.relations[key] = edge
            self.outgoing[src].add(key)
            self.incoming[dst].add(key)

        self.concepts[src].observations += 1
        self.concepts[dst].observations += 1
        return edge

    def revise_relation(
        self, src: str, rel: str, dst: str, *, reason: str = "revision"
    ) -> None:
        key = (self._norm(src), self._norm(rel), self._norm(dst))
        if key in self.relations:
            edge = self.relations[key]
            edge.contradiction_count += 1
            edge.strength *= 0.5
            edge.provenance.append(reason)

    # ---------- active thought / local activation ----------

    def activate(
        self,
        seeds: Iterable[str] | Dict[str, float],
        *,
        depth: int = 3,
        decay: float = 0.62,
        threshold: float = 0.08,
    ) -> ThoughtState:
        if isinstance(seeds, dict):
            seed_map = {self._norm(k): float(v) for k, v in seeds.items()}
        else:
            seed_map = {self._norm(k): 1.0 for k in seeds}

        activation: Dict[str, float] = defaultdict(float)
        best_depth: Dict[str, int] = {}
        traces: List[TraceStep] = []
        q = deque()

        for node, value in seed_map.items():
            self.ensure_concept(node)
            activation[node] = max(activation[node], value)
            best_depth[node] = 0
            q.append((node, value, 0))

        while q:
            node, act, d = q.popleft()
            if d >= depth:
                continue

            for key in self.outgoing.get(node, ()):
                edge = self.relations[key]
                edge_gain = edge.strength * edge.confidence
                propagated = act * decay * min(1.0, edge_gain)
                if propagated < threshold:
                    continue

                # Multiple paths can converge on a concept. Saturating addition
                # makes overlap matter without allowing unbounded activation.
                prior = activation[edge.dst]
                combined = 1.0 - (1.0 - min(prior, 0.999)) * (
                    1.0 - min(propagated, 0.999)
                )
                activation[edge.dst] = combined
                traces.append(
                    TraceStep(
                        src=edge.src,
                        rel=edge.rel,
                        dst=edge.dst,
                        activation=propagated,
                        confidence=edge.confidence,
                        provenance=list(edge.provenance),
                    )
                )

                next_depth = d + 1
                if next_depth <= best_depth.get(edge.dst, 10**9):
                    best_depth[edge.dst] = next_depth
                    q.append((edge.dst, combined, next_depth))

        conclusions = self._derive_from_active_subgraph(activation, traces)
        return ThoughtState(
            seeds=seed_map,
            activation=dict(sorted(activation.items(), key=lambda x: -x[1])),
            traces=traces,
            conclusions=conclusions,
        )

    def _derive_from_active_subgraph(
        self, activation: Dict[str, float], traces: List[TraceStep]
    ) -> List[Relation]:
        conclusions: List[Relation] = []

        # Transitive relational closure only inside the active neighborhood.
        active = {k for k, v in activation.items() if v >= 0.1}
        for a in list(active):
            for key1 in self.outgoing.get(a, ()):
                e1 = self.relations[key1]
                if e1.rel not in self.TRANSITIVE or e1.dst not in active:
                    continue
                for key2 in self.outgoing.get(e1.dst, ()):
                    e2 = self.relations[key2]
                    if e2.rel != e1.rel or e2.dst not in active:
                        continue
                    k = (a, e1.rel, e2.dst)
                    if k in self.relations:
                        continue
                    conf = min(e1.confidence, e2.confidence)
                    strength = min(e1.strength, e2.strength) * 0.8
                    conclusions.append(
                        Relation(
                            src=a,
                            rel=e1.rel,
                            dst=e2.dst,
                            strength=strength,
                            evidence_count=max(1, round(conf * 4)),
                            provenance=[
                                f"derived:{e1.src}-{e1.rel}->{e1.dst}",
                                f"derived:{e2.src}-{e2.rel}->{e2.dst}",
                            ],
                            derived=True,
                        )
                    )

        # Contextual inheritance: if X is_a Y and Y has a property/relation,
        # infer X has that property. This gives useful non-token reasoning.
        for child in list(active):
            for k1 in self.outgoing.get(child, ()):
                isa = self.relations[k1]
                if isa.rel != "is_a" or isa.dst not in active:
                    continue
                parent = isa.dst
                for k2 in self.outgoing.get(parent, ()):
                    prop = self.relations[k2]
                    if prop.rel == "is_a":
                        continue
                    if prop.dst not in active:
                        # permit one-hop property target even if activation is low
                        pass
                    candidate = (child, prop.rel, prop.dst)
                    if candidate in self.relations:
                        continue
                    conclusions.append(
                        Relation(
                            src=child,
                            rel=prop.rel,
                            dst=prop.dst,
                            strength=min(isa.strength, prop.strength) * 0.75,
                            evidence_count=max(
                                1,
                                round(
                                    min(isa.confidence, prop.confidence) * 4
                                ),
                            ),
                            provenance=[
                                f"inheritance:{child}-is_a->{parent}",
                                f"inheritance:{parent}-{prop.rel}->{prop.dst}",
                            ],
                            derived=True,
                        )
                    )

        # Deduplicate derived conclusions by key, preserving strongest.
        best: Dict[Tuple[str, str, str], Relation] = {}
        for c in conclusions:
            k = (c.src, c.rel, c.dst)
            if k not in best or c.strength > best[k].strength:
                best[k] = c
        return list(best.values())

    def consolidate(self, thought: ThoughtState, min_confidence: float = 0.6) -> int:
        """Let thought permanently change the model.

        Only derived relations with enough support become persistent. This is a
        primitive consolidation step; later versions should include temporal
        evidence, exceptions, causal tests, and structural compression.
        """
        added = 0
        for c in thought.conclusions:
            if c.confidence < min_confidence:
                continue
            before = (c.src, c.rel, c.dst) in self.relations
            self.observe_relation(
                c.src,
                c.rel,
                c.dst,
                provenance="consolidated:" + "|".join(c.provenance),
                strength=c.strength,
                derived=True,
            )
            added += int(not before)
        return added

    # ---------- introspection ----------

    def explain(self, thought: ThoughtState, target: Optional[str] = None) -> str:
        lines = []
        if target:
            target = self._norm(target)
            lines.append(f"Target: {target}")
            lines.append(f"Activation: {thought.activation.get(target, 0.0):.3f}")
            relevant = [t for t in thought.traces if t.dst == target or t.src == target]
        else:
            relevant = thought.traces

        lines.append("Active concepts:")
        for name, act in list(thought.activation.items())[:12]:
            lines.append(f"  {name:20s} {act:.3f}")

        if relevant:
            lines.append("Reasoning/provenance:")
            for t in relevant[-12:]:
                src = ", ".join(t.provenance[-2:])
                lines.append(
                    f"  {t.src} --{t.rel}--> {t.dst}  "
                    f"activation={t.activation:.3f} confidence={t.confidence:.2f} "
                    f"[{src}]"
                )

        if thought.conclusions:
            lines.append("Candidate conclusions:")
            for c in thought.conclusions[:12]:
                lines.append(
                    f"  {c.src} --{c.rel}--> {c.dst} "
                    f"confidence={c.confidence:.2f} via {'; '.join(c.provenance)}"
                )
        return "\n".join(lines)

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        data = {
            "event_counter": self.event_counter,
            "concepts": {k: asdict(v) for k, v in self.concepts.items()},
            "relations": [asdict(v) for v in self.relations.values()],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RelationalCognitiveEngine":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.event_counter = int(data.get("event_counter", 0))
        obj.concepts = {k: Concept(**v) for k, v in data["concepts"].items()}
        for raw in data["relations"]:
            edge = Relation(**raw)
            key = (edge.src, edge.rel, edge.dst)
            obj.relations[key] = edge
            obj.outgoing[edge.src].add(key)
            obj.incoming[edge.dst].add(key)
        return obj

    @staticmethod
    def _norm(x: str) -> str:
        return x.strip().lower().replace(" ", "_")


# ---------- deliberately tiny modality adapter ----------

def teach_triplet(engine: RelationalCognitiveEngine, sentence: str) -> Relation:
    """Parse only `subject relation object`.

    This is intentionally not sold as language understanding. It is a transparent
    adapter used to exercise the world model before adding an LLM/vision frontend.
    Underscores may be used for multiword concepts.
    """
    parts = sentence.strip().rstrip(".").split()
    if len(parts) < 3:
        raise ValueError("Use: subject relation object")
    src = parts[0]
    rel = parts[1]
    dst = "_".join(parts[2:])
    return engine.observe_relation(src, rel, dst, provenance=f"text:{sentence}")


def demo() -> None:
    e = RelationalCognitiveEngine()
    facts = [
        "aster is_a glorp",
        "glorp vulnerable_to cold",
        "aster located_at outside",
        "tonight is_a cold",
        "outside exposed_to cold",
    ]
    for f in facts:
        teach_triplet(e, f)

    print("=== MODEL AFTER EXPERIENCE ===")
    print(f"concepts={len(e.concepts)} relations={len(e.relations)}")

    thought = e.activate(["aster", "tonight", "outside"], depth=3)
    print("\n=== ACTIVE THOUGHT ===")
    print(e.explain(thought))

    learned = e.consolidate(thought)
    print(f"\nConsolidated {learned} new derived relation(s).")

    # Repeating experience strengthens structure instead of storing duplicate
    # transcript rows.
    teach_triplet(e, "glorp vulnerable_to cold")
    edge = e.relations[("glorp", "vulnerable_to", "cold")]
    print(
        "Repeated observation changed existing edge: "
        f"evidence={edge.evidence_count}, strength={edge.strength:.2f}"
    )

    # Contradictory experience revises confidence in-place.
    teach_triplet(e, "glorp safe_with cold")
    teach_triplet(e, "glorp unsafe_with cold")
    safe = e.relations[("glorp", "safe_with", "cold")]
    print(
        "Contradiction revised prior belief: "
        f"safe_with confidence={safe.confidence:.2f}, strength={safe.strength:.2f}"
    )

    path = Path("rce_v0_state.json")
    e.save(path)
    restored = RelationalCognitiveEngine.load(path)
    thought2 = restored.activate(["aster", "cold"], depth=3)
    print("\n=== AFTER SAVE / RELOAD ===")
    print(
        f"concepts={len(restored.concepts)} relations={len(restored.relations)} "
        f"aster_activation={thought2.activation.get('aster', 0):.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
