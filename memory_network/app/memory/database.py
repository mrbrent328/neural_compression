from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass(frozen=True)
class Entity:
    id: int
    name: str
    entity_type: str

@dataclass(frozen=True)
class Observation:
    id: int
    source_id: int
    relation: str
    target_id: int
    created_at: str

class MemoryDB:
    def __init__(self, path: str = "data/memory.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL DEFAULT 'entity'
        );
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL REFERENCES entities(id),
            relation TEXT NOT NULL,
            target_id INTEGER NOT NULL REFERENCES entities(id),
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_obs_source ON observations(source_id);
        CREATE INDEX IF NOT EXISTS idx_obs_target ON observations(target_id);
        CREATE INDEX IF NOT EXISTS idx_obs_relation ON observations(relation);
        ''')
        self.conn.commit()

    def entity(self, name: str, entity_type: str = "entity") -> Entity:
        row = self.conn.execute("SELECT id,name,entity_type FROM entities WHERE name=?", (name,)).fetchone()
        if row:
            return Entity(*row)
        cur = self.conn.execute("INSERT INTO entities(name,entity_type) VALUES(?,?)", (name, entity_type))
        self.conn.commit()
        return Entity(cur.lastrowid, name, entity_type)

    def add_observation(self, source: str, relation: str, target: str,
                        source_type: str = "entity", target_type: str = "entity") -> Observation:
        s = self.entity(source, source_type)
        t = self.entity(target, target_type)
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO observations(source_id,relation,target_id,created_at) VALUES(?,?,?,?)",
            (s.id, relation, t.id, now),
        )
        self.conn.commit()
        return Observation(cur.lastrowid, s.id, relation, t.id, now)

    def observations_for(self, entity_id: int):
        return self.conn.execute('''
            SELECT o.id,o.source_id,o.relation,o.target_id,o.created_at
            FROM observations o
            WHERE o.source_id=? OR o.target_id=?
            ORDER BY o.id
        ''', (entity_id, entity_id)).fetchall()

    def entity_by_id(self, entity_id: int) -> Entity | None:
        row = self.conn.execute("SELECT id,name,entity_type FROM entities WHERE id=?", (entity_id,)).fetchone()
        return Entity(*row) if row else None

    def entity_by_name(self, name: str) -> Entity | None:
        row = self.conn.execute("SELECT id,name,entity_type FROM entities WHERE name=?", (name,)).fetchone()
        return Entity(*row) if row else None

    def all_entities(self):
        return [Entity(*r) for r in self.conn.execute("SELECT id,name,entity_type FROM entities ORDER BY id")]

    def all_observations(self):
        return [Observation(*r) for r in self.conn.execute("SELECT id,source_id,relation,target_id,created_at FROM observations ORDER BY id")]

    def close(self):
        self.conn.close()
