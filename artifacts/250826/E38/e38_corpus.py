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


def build(seed: int = 3801, facts: int = 96, questions: int = 24) -> tuple[list[Item], list[Item]]:
    """Build a deterministic fact stream and a bounded question set.

    ``facts`` counts observations, and each observation contributes two fact
    items: an ownership edge and an object-location edge. Questions are built
    from the final graph state, so every expected answer is unambiguous.
    """
    if facts < 0 or questions < 0:
        raise ValueError("facts and questions must be non-negative")

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

    candidates: list[Item] = []
    for (subject, predicate), obj in graph.items():
        if predicate != "owns":
            continue
        candidates.append(Item("question", f"What does {subject} own?", obj))
        place = graph.get((obj, "at"))
        if place is not None:
            candidates.append(Item("question", f"Where is the object owned by {subject}?", place))

    # There are only two semantic question types per entity. Keep one question
    # per exact wording and fail clearly if a caller asks for more than exists.
    rng.shuffle(candidates)
    unique: list[Item] = []
    seen: set[str] = set()
    for item in candidates:
        if item.text in seen:
            continue
        seen.add(item.text)
        unique.append(item)

    if questions > len(unique):
        raise ValueError(
            f"requested {questions} questions, but this corpus can produce only "
            f"{len(unique)} unique questions; reduce --questions or expand the corpus"
        )

    return stream, unique[:questions]
