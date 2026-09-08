"""
audit.py -- an append-only record of what the assistant director did and why.

Six weeks after a build, "why is panel 11 that one?" needs an answer. Every tool
call, every choice between variants, every consent decision lands here with its
reasoning. SQLite because it survives concurrent writers, unlike the registry's
workbook; per SUITE.md the creative *result* still lands in the registry row's
notes -- this is the operational log beside it, not a competing source of truth.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

_LOCAL = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    name        TEXT,
    entry_id    TEXT,
    summary     TEXT,
    detail      TEXT,
    ok          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_agent_events_run ON agent_events(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_entry ON agent_events(entry_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _connect(db_path: str) -> sqlite3.Connection:
    """One connection per thread. SQLite objects aren't shareable across threads,
    and the runner deliberately works on a background thread."""
    key = f"conn::{os.path.abspath(db_path)}"
    conn = getattr(_LOCAL, key, None)
    if conn is None:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        conn.commit()
        setattr(_LOCAL, key, conn)
    return conn


class AuditLog:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def record(self, run_id: str, kind: str, name: str = "", entry_id: str = "",
               summary: str = "", detail=None, ok: bool = True) -> None:
        conn = _connect(self.db_path)
        payload = detail
        if payload is not None and not isinstance(payload, str):
            try:
                payload = json.dumps(payload, default=str)[:20000]
            except (TypeError, ValueError):
                payload = str(payload)[:20000]
        conn.execute(
            "INSERT INTO agent_events (run_id, at, kind, name, entry_id, summary, detail, ok)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (run_id, _now(), kind, name, entry_id, (summary or "")[:2000], payload, 1 if ok else 0),
        )
        conn.commit()

    def events(self, run_id: str = "", limit: int = 200) -> list:
        conn = _connect(self.db_path)
        if run_id:
            rows = conn.execute(
                "SELECT run_id, at, kind, name, entry_id, summary, detail, ok FROM agent_events"
                " WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT run_id, at, kind, name, entry_id, summary, detail, ok FROM agent_events"
                " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys = ("run_id", "at", "kind", "name", "entry_id", "summary", "detail", "ok")
        return [dict(zip(keys, row)) for row in rows]

    def history_for(self, entry_id: str, limit: int = 50) -> list:
        conn = _connect(self.db_path)
        rows = conn.execute(
            "SELECT at, kind, name, summary, ok FROM agent_events WHERE entry_id=?"
            " ORDER BY id DESC LIMIT ?", (entry_id, limit)).fetchall()
        return [dict(zip(("at", "kind", "name", "summary", "ok"), row)) for row in rows]
