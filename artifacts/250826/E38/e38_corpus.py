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
    """Build a deterministic fact stream and bounded question set.

    ``facts`` counts observations. Each observation emits two fact items:
    ownership and object-location. The first eight observations deliberately
    cover every entity so the default 24-question space is always available;
    remaining observations are randomized.
    """
    if facts < len(ENTITIES) or questions < 0:
        raise ValueError(f"facts must be at least {len(ENTITIES)} and questions non-negative")

    rng = random.Random(seed)
    stream: list[Item] = []
    graph: dict[tuple[str, str], str] = {}

    # Guarantee coverage of every entity. This makes the default question
    # count deterministic instead of depending on random sampling luck.
    for subject, obj, place in zip(ENTITIES, OBJECTS, PLACES):
        graph[(subject, "owns")] = obj
        graph[(obj, "at")] = place
        stream.append(Item("fact", f"{subject} owns the {obj}."))
        stream.append(Item("fact", f"The {obj} is at the {place}."))

    for _ in range(facts - len(ENTITIES)):
        subject = rng.choice(ENTITIES)
        obj = rng.choice(OBJECTS)
        place = rng.choice(PLACES)
        graph[(subject, "owns")] = obj
        graph[(obj, "at")] = place
        stream.append(Item("fact", f"{subject} owns the {obj}."))
        stream.append(Item("fact", f"The {obj} is at the {place}."))

    candidates: list[Item] = []
    for subject in ENTITIES:
        obj = graph[(subject, "owns")]
        place = graph.get((obj, "at"))
        candidates.append(Item("question", f"What does {subject} own?", obj))
        if place is not None:
            candidates.append(Item("question", f"Where is the object owned by {subject}?", place))
            # Keep the subject in the question so repeated objects do not
            # collapse distinct questions into duplicate text.
            candidates.append(Item("question", f"Where is {subject}'s object?", place))

    rng.shuffle(candidates)
    unique: list[Item] = []
    seen: set[str] = set()
    for item in candidates:
        if item.text not in seen:
            seen.add(item.text)
            unique.append(item)

    if questions > len(unique):
        raise ValueError(f"requested {questions} questions, but this corpus can produce only {len(unique)} unique questions")

    return stream, unique[:questions]
