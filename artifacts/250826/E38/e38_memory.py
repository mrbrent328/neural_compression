#!/usr/bin/env python3
"""E38 persistent relational memory.

Small, inspectable memory layer derived from the E36/E37 persistent-unit and
relation idea. It intentionally stores structure rather than raw history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
from pathlib import Path
from typing import Iterable


@dataclass
class Entity:
    name: str
    attributes: dict[str, str] = field(default_factory=dict)
    observations: int = 0


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    observations: int = 1


class RelationalMemory:
    """SQLite-backed persistent graph with a deliberately tiny API."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                attributes TEXT NOT NULL DEFAULT '{}',
                observations INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS relations (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                observations INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(subject, predicate, object)
            );
            CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject);
            CREATE INDEX IF NOT EXISTS idx_rel_object ON relations(object);
            """
        )
        self.db.commit()

    def observe_entity(self, name: str, attributes: dict[str, str] | None = None) -> None:
        attributes = attributes or {}
        row = self.db.execute("SELECT attributes, observations FROM entities WHERE name=?", (name,)).fetchone()
        if row is None:
            merged = attributes
            count = 1
            self.db.execute("INSERT INTO entities VALUES (?, ?, ?)", (name, json.dumps(merged), count))
        else:
            merged = json.loads(row["attributes"])
            merged.update(attributes)
            count = row["observations"] + 1
            self.db.execute("UPDATE entities SET attributes=?, observations=? WHERE name=?", (json.dumps(merged), count, name))
        self.db.commit()

    def observe_relation(self, subject: str, predicate: str, object_: str, confidence: float = 1.0) -> None:
        self.observe_entity(subject)
        self.observe_entity(object_)
        row = self.db.execute(
            "SELECT observations, confidence FROM relations WHERE subject=? AND predicate=? AND object=?",
            (subject, predicate, object_),
        ).fetchone()
        if row is None:
            self.db.execute("INSERT INTO relations VALUES (?, ?, ?, ?, ?)", (subject, predicate, object_, confidence, 1))
        else:
            n = row["observations"]
            mean_conf = (row["confidence"] * n + confidence) / (n + 1)
            self.db.execute(
                "UPDATE relations SET observations=?, confidence=? WHERE subject=? AND predicate=? AND object=?",
                (n + 1, mean_conf, subject, predicate, object_),
            )
        self.db.commit()

    def related(self, entity: str, depth: int = 2, limit: int = 32) -> list[Relation]:
        """Return a compact neighborhood around an entity."""
        seen = {entity}
        frontier = [entity]
        found: list[Relation] = []
        for _ in range(max(depth, 1)):
            if not frontier or len(found) >= limit:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = self.db.execute(
                f"SELECT * FROM relations WHERE subject IN ({placeholders}) OR object IN ({placeholders}) ORDER BY observations DESC LIMIT ?",
                frontier + frontier + [limit - len(found)],
            ).fetchall()
            next_frontier = []
            for row in rows:
                rel = Relation(row["subject"], row["predicate"], row["object"], row["confidence"], row["observations"])
                found.append(rel)
                other = rel.object if rel.subject in seen else rel.subject
                if other not in seen:
                    seen.add(other)
                    next_frontier.append(other)
                if len(found) >= limit:
                    break
            frontier = next_frontier
        return found

    def serialize_context(self, entities: Iterable[str], depth: int = 2, limit: int = 32) -> str:
        lines = []
        emitted: set[tuple[str, str, str]] = set()
        for entity in entities:
            for rel in self.related(entity, depth=depth, limit=limit):
                key = (rel.subject, rel.predicate, rel.object)
                if key in emitted:
                    continue
                emitted.add(key)
                lines.append(f"{rel.subject} --{rel.predicate}--> {rel.object}")
        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        return {
            "entities": self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
            "relations": self.db.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
        }

    def close(self) -> None:
        self.db.close()
