"""E39 — Persistent Relational Sufficiency.

Build a deterministic relational representation from observations, hide the
observations, and test whether the graph alone can answer held-out questions.
No LLM is used: this isolates representational sufficiency.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    source_id: str
    span: str


@dataclass(frozen=True)
class Question:
    question_id: str
    kind: str
    subject: str
    relation: str
    answer: str
    chain: tuple[tuple[str, str, str], ...]


RELATIONS = {
    "works_at": "works_at",
    "located_in": "located_in",
    "uses": "uses",
    "depends_on": "depends_on",
    "owned_by": "owned_by",
}


class Memory:
    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute(
            "CREATE TABLE facts (id INTEGER PRIMARY KEY, subject TEXT, relation TEXT, object TEXT, source_id TEXT, span TEXT)"
        )

    def add(self, fact: Fact) -> None:
        self.db.execute(
            "INSERT INTO facts(subject, relation, object, source_id, span) VALUES (?, ?, ?, ?, ?)",
            (fact.subject, fact.relation, fact.object, fact.source_id, fact.span),
        )

    def commit(self) -> None:
        self.db.commit()

    def edges(self) -> list[tuple[str, str, str]]:
        return self.db.execute(
            "SELECT subject, relation, object FROM facts ORDER BY id"
        ).fetchall()

    def answer(self, q: Question) -> tuple[str | None, list[tuple[str, str, str]]]:
        # Direct query.
        row = self.db.execute(
            "SELECT object FROM facts WHERE subject=? AND relation=? ORDER BY id LIMIT 1",
            (q.subject, q.relation),
        ).fetchone()
        if row:
            edge = (q.subject, q.relation, row[0])
            return row[0], [edge]

        # Two-hop query: subject -- r1 --> intermediate -- r2 --> answer.
        first = self.db.execute(
            "SELECT object, relation FROM facts WHERE subject=? ORDER BY id",
            (q.subject,),
        ).fetchall()
        for intermediate, r1 in first:
            row = self.db.execute(
                "SELECT object FROM facts WHERE subject=? AND relation=? ORDER BY id LIMIT 1",
                (intermediate, q.relation),
            ).fetchone()
            if row:
                return row[0], [
                    (q.subject, r1, intermediate),
                    (intermediate, q.relation, row[0]),
                ]
        return None, []


def make_corpus(seed: int, entities: int) -> tuple[list[Fact], list[Question]]:
    rng = random.Random(seed)
    people = [f"person_{i:03d}" for i in range(entities)]
    companies = [f"company_{i:03d}" for i in range(max(entities // 3, 2))]
    cities = [f"city_{i:03d}" for i in range(max(entities // 4, 2))]
    tools = [f"tool_{i:03d}" for i in range(max(entities // 5, 2))]

    facts: list[Fact] = []
    for i, person in enumerate(people):
        company = companies[i % len(companies)]
        city = cities[(i * 3) % len(cities)]
        tool = tools[(i * 5 + 1) % len(tools)]
        facts.extend([
            Fact(person, RELATIONS["works_at"], company, f"obs_{len(facts):04d}", f"{person} works at {company}."),
            Fact(company, RELATIONS["located_in"], city, f"obs_{len(facts):04d}", f"{company} is located in {city}."),
            Fact(person, RELATIONS["uses"], tool, f"obs_{len(facts):04d}", f"{person} uses {tool}."),
        ])

    # Add a second relational layer so some held-out questions require two hops.
    for i, company in enumerate(companies):
        tool = tools[(i * 2 + 2) % len(tools)]
        facts.append(Fact(company, RELATIONS["uses"], tool, f"obs_{len(facts):04d}", f"{company} uses {tool}."))

    # Questions are generated from the complete source graph but are NOT fed to
    # graph construction. Direct and two-hop questions are kept separate.
    questions: list[Question] = []
    for i, person in enumerate(people):
        company = companies[i % len(companies)]
        city = cities[(i * 3) % len(cities)]
        tool = tools[(i * 5 + 1) % len(tools)]
        questions.append(Question(f"q_direct_{i:03d}", "direct", person, "works_at", company, ((person, "works_at", company),)))
        questions.append(Question(f"q_hop_{i:03d}", "two_hop", person, "located_in", city, ((person, "works_at", company), (company, "located_in", city))))
        questions.append(Question(f"q_direct_tool_{i:03d}", "direct", person, "uses", tool, ((person, "uses", tool),)))

    rng.shuffle(questions)
    return facts, questions


def reduce_graph(facts: list[Fact], budget_fraction: float, seed: int) -> list[Fact]:
    if budget_fraction >= 1.0:
        return list(facts)
    n = max(1, round(len(facts) * budget_fraction))
    # Deterministic selection, independent of the held-out questions.
    ranked = sorted(facts, key=lambda f: (f.subject, f.relation, f.object, f.source_id))
    rng = random.Random(seed + 3901)
    rng.shuffle(ranked)
    selected = sorted(ranked[:n], key=lambda f: f.source_id)
    return selected


def run(seed: int, entities: int, budgets: list[float], output: Path) -> dict:
    source_facts, questions = make_corpus(seed, entities)
    source_by_key = {(f.subject, f.relation): f.object for f in source_facts}

    # Ground truth is derived from the source corpus, not from the tested graph.
    source_truth = {q.question_id: q.answer for q in questions}

    results = []
    for fraction in budgets:
        graph_facts = reduce_graph(source_facts, fraction, seed)
        memory = Memory()
        for fact in graph_facts:
            memory.add(fact)
        memory.commit()

        started = time.perf_counter()
        correct = 0
        resolved = 0
        supported = 0
        question_results = []
        for q in questions:
            predicted, chain = memory.answer(q)
            is_correct = predicted == source_truth[q.question_id]
            has_support = bool(chain) and predicted is not None and all(
                edge in {(f.subject, f.relation, f.object) for f in graph_facts}
                for edge in chain
            )
            correct += int(is_correct)
            resolved += int(predicted is not None)
            supported += int(has_support)
            question_results.append({
                "question_id": q.question_id,
                "kind": q.kind,
                "answer": source_truth[q.question_id],
                "predicted": predicted,
                "correct": is_correct,
                "supported": has_support,
                "support_chain": [list(edge) for edge in chain],
            })
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total = len(questions)
        results.append({
            "budget_fraction": fraction,
            "graph_facts": len(graph_facts),
            "graph_fraction_actual": len(graph_facts) / len(source_facts),
            "accuracy": correct / total,
            "resolution_rate": resolved / total,
            "evidence_coverage": supported / total,
            "query_time_ms": elapsed_ms,
            "questions": question_results,
        })

    # Baseline is deliberately source-grounded: all source facts are available.
    baseline_memory = Memory()
    for fact in source_facts:
        baseline_memory.add(fact)
    baseline_memory.commit()
    baseline_correct = sum(
        baseline_memory.answer(q)[0] == source_truth[q.question_id] for q in questions
    )

    payload = {
        "experiment": "E39",
        "question": "How much information can be removed from an observation stream while a persistent relational representation retains the ability to answer previously unseen questions?",
        "seed": seed,
        "entities": entities,
        "source_facts": len(source_facts),
        "held_out_questions": len(questions),
        "question_types": {
            "direct": sum(q.kind == "direct" for q in questions),
            "two_hop": sum(q.kind == "two_hop" for q in questions),
        },
        "baseline": {
            "available_source_facts": len(source_facts),
            "accuracy": baseline_correct / len(questions),
            "exact_source_truth": True,
        },
        "budgets": results,
        "invariants": {
            "original_observations_hidden_during_graph_queries": True,
            "questions_not_used_for_graph_selection": True,
            "ground_truth_external_to_tested_graph": True,
            "support_requires_explicit_graph_edges": True,
            "no_llm": True,
        },
        "source_observations": [f.__dict__ for f in source_facts],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--entities", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("e39_result.json"))
    args = parser.parse_args()
    budgets = [1.0, 0.75, 0.5, 0.25, 0.125]
    payload = run(args.seed, args.entities, budgets, args.output)
    print("E39 — Persistent Relational Sufficiency")
    print(f"source facts: {payload['source_facts']}")
    print(f"held-out questions: {payload['held_out_questions']}")
    print(f"baseline accuracy: {payload['baseline']['accuracy']:.3f}")
    for r in payload["budgets"]:
        print(
            f"budget={r['graph_fraction_actual']:.3f} "
            f"facts={r['graph_facts']} "
            f"accuracy={r['accuracy']:.3f} "
            f"resolved={r['resolution_rate']:.3f} "
            f"evidence={r['evidence_coverage']:.3f}"
        )
    print(f"result: {args.output}")


if __name__ == "__main__":
    main()
