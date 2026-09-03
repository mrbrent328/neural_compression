#!/usr/bin/env python3
"""Controlled facts/questions for E38.

The corpus is intentionally deterministic and small. It lets us distinguish
memory/retrieval failures from model failures before moving to natural text.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    kind: str
    text: str
    answer: str | None = None


ENTITIES = ["Alice", "Bob", "Cara", "Dan", "Eve", "Fox", "Gina", "Hank"]
OBJECTS = ["bicycle", "camera", "lantern", "map", "key", "backpack", "compass", "radio"]
PLACES = ["red barn", "old bridge", "north cabin", "blue shed", "stone tower", "green gate", "river dock", "hill house"]


def build(seed: int = 3801, facts: int = 96, questions: int = 48) -> tuple[list[Item], list[Item]]:
    rng = random.Random(seed)
    stream: list[Item] = []
    graph: dict[tuple[str, str], str] = {}

    for _ in range(facts):
        subject = rng.choice(ENTITIES)
        obj = rng.choice(OBJECTS)
        place = rng.choice(PLACES)
        graph[(subject, "owns")] = obj
        graph[(obj, "at")] = place
        stream.append(Item("fact", f"{subject} owns the {obj}."))
        stream.append(Item("fact", f"The {obj} is at the {place}."))

    # Deduplicate the final question set while retaining one-hop and two-hop queries.
    candidates: list[Item] = []
    for (subject, pred), obj in graph.items():
        candidates.append(Item("question", f"What does {subject} own?", obj))
        place = graph.get((obj, "at"))
        if place:
            candidates.append(Item("question", f"Where is the object owned by {subject}?", place))
    rng.shuffle(candidates)
    seen = set()
    for item in candidates:
        if item.text not in seen:
            seen.add(item.text)
            if len(questions) <= len(seen):
                break
    questions_out = [x for x in candidates if x.text in seen][:questions]
    return stream, questions_out
