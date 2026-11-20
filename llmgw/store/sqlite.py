"""Local state in one SQLite file. Same contract as the DynamoDB store, used for development and tests."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ..teams import TeamPolicy

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams    (team_id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS clients  (client_id TEXT PRIMARY KEY, team_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS spend    (team_id TEXT, month TEXT, value INTEGER NOT NULL, PRIMARY KEY (team_id, month));
CREATE TABLE IF NOT EXISTS cache    (key TEXT PRIMARY KEY, body TEXT NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS audit    (team_id TEXT, ts TEXT, request_id TEXT, body TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS audit_team_ts ON audit (team_id, ts);
"""


class SqliteStore:
    def __init__(self, path: Path | str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")   # durable at checkpoints; fine for local state
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    # --- teams -------------------------------------------------------------------------
    def put_team(self, policy: TeamPolicy) -> None:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute("INSERT OR REPLACE INTO teams VALUES (?, ?)", (policy.team_id, json.dumps(policy.to_dict())))
            self._db.execute("DELETE FROM clients WHERE team_id = ?", (policy.team_id,))
            for cid in policy.client_ids:
                self._db.execute("INSERT OR REPLACE INTO clients VALUES (?, ?)", (cid, policy.team_id))
            self._db.execute("COMMIT")

    def _one(self, sql: str, args=()):
        rows = self._q(sql, args)
        return rows[0] if rows else None

    def _q(self, sql: str, args=()) -> list:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def get_team(self, team_id: str) -> TeamPolicy | None:
        row = self._one("SELECT body FROM teams WHERE team_id = ?", (team_id,))
        return TeamPolicy.from_dict(json.loads(row[0])) if row else None

    def team_for_client(self, client_id: str) -> TeamPolicy | None:
        row = self._one("SELECT team_id FROM clients WHERE client_id = ?", (client_id,))
        return self.get_team(row[0]) if row else None

    def list_teams(self) -> list[TeamPolicy]:
        return [TeamPolicy.from_dict(json.loads(r[0])) for r in self._q("SELECT body FROM teams ORDER BY team_id")]

    # --- limits ------------------------------------------------------------------------
    def hit_window(self, key: str, amount: int, limit: int, ttl_s: int) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute("SELECT value, expires FROM counters WHERE key = ?", (key,)).fetchone()
            value = row[0] if row and row[1] > now else 0
            if value + amount > limit:
                self._db.execute("COMMIT")
                return False, value
            self._db.execute("INSERT OR REPLACE INTO counters VALUES (?, ?, ?)", (key, value + amount, now + ttl_s))
            self._db.execute("COMMIT")
            return True, value + amount

    def reserve_spend(self, team_id: str, month: str, amount: int, budget: int) -> tuple[bool, int]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            spent = self._spend(team_id, month)
            if spent + amount > budget:
                self._db.execute("COMMIT")
                return False, spent
            self._db.execute("INSERT OR REPLACE INTO spend VALUES (?, ?, ?)", (team_id, month, spent + amount))
            self._db.execute("COMMIT")
            return True, spent + amount

    def adjust_spend(self, team_id: str, month: str, delta: int) -> int:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            value = self._spend(team_id, month) + delta
            self._db.execute("INSERT OR REPLACE INTO spend VALUES (?, ?, ?)", (team_id, month, value))
            self._db.execute("COMMIT")
            return value

    def get_spend(self, team_id: str, month: str) -> int:
        with self._lock:
            return self._spend(team_id, month)

    def _spend(self, team_id: str, month: str) -> int:
        row = self._db.execute("SELECT value FROM spend WHERE team_id = ? AND month = ?", (team_id, month)).fetchone()
        return row[0] if row else 0

    # --- cache -------------------------------------------------------------------------
    def cache_get(self, key: str) -> dict | None:
        row = self._one("SELECT body, expires FROM cache WHERE key = ?", (key,))
        return json.loads(row[0]) if row and row[1] > time.time() else None

    def cache_put(self, key: str, value: dict, ttl_s: int) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value), time.time() + ttl_s))

    # --- audit -------------------------------------------------------------------------
    def write_audit(self, record: dict) -> None:
        with self._lock:
            self._db.execute("INSERT INTO audit VALUES (?, ?, ?, ?)",
                             (record.get("team_id"), record["ts"], record["request_id"], json.dumps(record)))

    def audit_records(self, team_id: str | None = None, since: str | None = None) -> list[dict]:
        sql, args = "SELECT body FROM audit WHERE 1=1", []
        if team_id:
            sql += " AND team_id = ?"; args.append(team_id)
        if since:
            sql += " AND ts >= ?"; args.append(since)
        return [json.loads(r[0]) for r in self._q(sql + " ORDER BY ts", args)]
