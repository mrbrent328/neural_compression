#!/usr/bin/env python3
"""R0: minimal relational reconstruction experiment.

The experiment intentionally contains no neural model and no object features.
A hidden system is a tiny graph. The observer receives separate relationships,
retains them in persistent memory, and answers whether two nodes are connected.

The key test is whether two observations can be combined to answer a query that
neither observation answers by itself.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

Node = str
Edge = Tuple[Node, Node]


@dataclass(frozen=True)
class HiddenSystem:
    """The ground-truth relational system, inaccessible to the observer."""

    edges: Set[Edge]

    def related(self, a: Node, b: Node) -> bool:
        """Return whether a and b are connected in the hidden system."""
        if a == b:
            return True

        graph: Dict[Node, Set[Node]] = {}
        for left, right in self.edges:
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)

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


@dataclass
class RelationMemory:
    """Persistent memory containing only observed relationships."""

    edges: Set[Edge] = field(default_factory=set)

    def observe(self, edge: Edge) -> None:
        """Store an observed undirected relationship."""
        a, b = edge
        self.edges.add(canonical_edge(a, b))

    def related(self, a: Node, b: Node) -> bool:
        """Infer connectivity using only relationships retained in memory."""
        if a == b:
            return True

        graph: Dict[Node, Set[Node]] = {}
        for left, right in self.edges:
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)

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


def canonical_edge(a: Node, b: Node) -> Edge:
    """Represent an undirected edge consistently."""
    return (a, b) if a <= b else (b, a)


def run_minimal_case() -> None:
    """Run the smallest non-trivial relational reconstruction case."""
    system = HiddenSystem(
        edges={
            canonical_edge("A", "B"),
            canonical_edge("B", "C"),
        }
    )

    observations: List[Edge] = [
        ("A", "B"),
        ("B", "C"),
    ]
    query = ("A", "C")

    memory = RelationMemory()
    print("R0 — Minimal Relational Reconstruction")
    print()
    print("Hidden structure: A — B — C")
    print("Observer receives: A — B, then B — C")
    print(f"Query: {query[0]} — {query[1]}")
    print()

    for observation in observations:
        memory.observe(observation)
        print(f"memory after {observation}: {sorted(memory.edges)}")

    prediction = memory.related(*query)
    truth = system.related(*query)
    print()
    print(f"prediction: {prediction}")
    print(f"truth:      {truth}")
    print(f"correct:    {prediction == truth}")

    assert truth is True
    assert prediction is True


def run_direct-observation_control() -> None:
    """Show that neither individual observation answers the query alone."""
    system = HiddenSystem(
        edges={canonical_edge("A", "B"), canonical_edge("B", "C")}
    )
    query = ("A", "C")

    for observation in [("A", "B"), ("B", "C")]:
        memory = RelationMemory()
        memory.observe(observation)
        assert system.related(*query) is True
        assert memory.related(*query) is False


def main() -> None:
    run_minimal_case()
    run_direct_observation_control()
    print("\nR0 passed.")


if __name__ == "__main__":
    main()
