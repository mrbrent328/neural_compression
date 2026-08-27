#!/usr/bin/env python3
"""R0: minimal relational reconstruction experiment.

No neural model and no object features. A hidden system is a tiny graph. The
observer receives separate relationships, retains them, and answers whether
two nodes are connected. The query requires combining both observations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

Node = str
Edge = Tuple[Node, Node]


@dataclass(frozen=True)
class HiddenSystem:
    edges: Set[Edge]

    def related(self, a: Node, b: Node) -> bool:
        return connected(build_graph(self.edges), a, b)


@dataclass
class RelationMemory:
    edges: Set[Edge] = field(default_factory=set)

    def observe(self, edge: Edge) -> None:
        self.edges.add(canonical_edge(*edge))

    def related(self, a: Node, b: Node) -> bool:
        return connected(build_graph(self.edges), a, b)


def canonical_edge(a: Node, b: Node) -> Edge:
    return (a, b) if a <= b else (b, a)


def build_graph(edges: Set[Edge]) -> Dict[Node, Set[Node]]:
    graph: Dict[Node, Set[Node]] = {}
    for left, right in edges:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def connected(graph: Dict[Node, Set[Node]], a: Node, b: Node) -> bool:
    if a == b:
        return True
    frontier = [a]
    visited = {a}
    while frontier:
        current = frontier.pop()
        for neighbor in graph.get(current, set()):
            if neighbor == b:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def run_minimal_case() -> None:
    system = HiddenSystem({canonical_edge("A", "B"), canonical_edge("B", "C")})
    observations: List[Edge] = [("A", "B"), ("B", "C")]
    query = ("A", "C")

    memory = RelationMemory()
    print("R0 — Minimal Relational Reconstruction")
    print("Hidden structure: A — B — C")
    print("Observer receives: A — B, then B — C")
    print(f"Query: {query[0]} — {query[1]}")

    for observation in observations:
        memory.observe(observation)
        print(f"memory after {observation}: {sorted(memory.edges)}")

    prediction = memory.related(*query)
    truth = system.related(*query)
    print(f"prediction: {prediction}")
    print(f"truth:      {truth}")
    print(f"correct:    {prediction == truth}")

    assert truth is True
    assert prediction is True


def run_direct_observation_control() -> None:
    system = HiddenSystem({canonical_edge("A", "B"), canonical_edge("B", "C")})
    query = ("A", "C")

    for observation in [("A", "B"), ("B", "C")]:
        memory = RelationMemory()
        memory.observe(observation)
        assert system.related(*query) is True
        assert memory.related(*query) is False


def main() -> None:
    run_minimal_case()
    run_direct_observation_control()
    print("R0 passed.")


if __name__ == "__main__":
    main()
