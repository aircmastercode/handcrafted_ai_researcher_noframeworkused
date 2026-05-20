"""SQLite-backed session store.

One file, four tables, atomic writes. Survives restarts and concurrent turns.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from deep_research.models import (
    Message,
    Plan,
    Session,
    Snippet,
    Turn,
    now_iso,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    rolling_summary TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    ts         TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    query          TEXT NOT NULL,
    plan_json      TEXT DEFAULT '',
    search_queries TEXT DEFAULT '[]',
    urls_opened    TEXT DEFAULT '[]',
    final_answer   TEXT DEFAULT '',
    ts             TEXT NOT NULL,
    latency_ms     INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS snippets (
    snippet_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id      INTEGER NOT NULL,
    sid          TEXT NOT NULL,
    url          TEXT NOT NULL,
    title        TEXT DEFAULT '',
    domain       TEXT DEFAULT '',
    text         TEXT NOT NULL,
    score        REAL DEFAULT 0.0,
    retrieved_at TEXT NOT NULL,
    FOREIGN KEY (turn_id) REFERENCES turns(turn_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_session    ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_snippets_turn    ON snippets(turn_id);
"""


class SessionStore:
    """Thread-safe SQLite store for sessions, messages, turns, and snippets."""

    def __init__(self, db_path: str | Path = "session.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._init_schema()

    # ----- low-level helpers -----

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with foreign keys on and dict-style rows."""
        with self._lock:
            conn = sqlite3.connect(self.db_path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                yield conn
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # ----- sessions -----

    def create_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at, rolling_summary) "
                "VALUES (?, ?, '')",
                (sid, now_iso()),
            )
        return sid

    def list_sessions(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT s.session_id, s.created_at, s.rolling_summary, "
                "(SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS n_turns "
                "FROM sessions s ORDER BY s.created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_rolling_summary(self, session_id: str) -> str:
        with self._conn() as c:
            row = c.execute(
                "SELECT rolling_summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["rolling_summary"] if row else ""

    def set_rolling_summary(self, session_id: str, summary: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE sessions SET rolling_summary = ? WHERE session_id = ?",
                (summary, session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    # ----- messages -----

    def append_message(self, session_id: str, msg: Message) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, msg.role, msg.content, msg.ts),
            )

    def get_messages(self, session_id: str) -> list[Message]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, ts FROM messages WHERE session_id = ? "
                "ORDER BY message_id ASC",
                (session_id,),
            ).fetchall()
        return [Message(**dict(r)) for r in rows]

    # ----- turns + snippets (single atomic write) -----

    def append_turn(self, session_id: str, turn: Turn) -> int:
        """Insert a turn plus all its snippets atomically. Returns the turn_id."""
        plan_json = turn.plan.model_dump_json() if turn.plan else ""
        with self._conn() as c:
            c.execute("BEGIN")
            try:
                cur = c.execute(
                    "INSERT INTO turns("
                    "session_id, query, plan_json, search_queries, urls_opened, "
                    "final_answer, ts, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        turn.query,
                        plan_json,
                        json.dumps(turn.search_queries),
                        json.dumps(turn.urls_opened),
                        turn.final_answer,
                        turn.ts,
                        turn.latency_ms,
                    ),
                )
                turn_id = int(cur.lastrowid or 0)
                for s in turn.snippets:
                    c.execute(
                        "INSERT INTO snippets("
                        "turn_id, sid, url, title, domain, text, score, retrieved_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            turn_id,
                            s.sid,
                            s.url,
                            s.title,
                            s.domain,
                            s.text,
                            s.score,
                            s.retrieved_at,
                        ),
                    )
                c.execute("COMMIT")
                return turn_id
            except Exception:
                c.execute("ROLLBACK")
                raise

    def get_turns(self, session_id: str) -> list[Turn]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT turn_id, query, plan_json, search_queries, urls_opened, "
                "final_answer, ts, latency_ms "
                "FROM turns WHERE session_id = ? ORDER BY turn_id ASC",
                (session_id,),
            ).fetchall()
            turns: list[Turn] = []
            for r in rows:
                snip_rows = c.execute(
                    "SELECT sid, url, title, domain, text, score, retrieved_at "
                    "FROM snippets WHERE turn_id = ? ORDER BY snippet_id ASC",
                    (r["turn_id"],),
                ).fetchall()
                snippets = [Snippet(**dict(s)) for s in snip_rows]
                plan = Plan.model_validate_json(r["plan_json"]) if r["plan_json"] else None
                turns.append(
                    Turn(
                        query=r["query"],
                        plan=plan,
                        search_queries=json.loads(r["search_queries"] or "[]"),
                        urls_opened=json.loads(r["urls_opened"] or "[]"),
                        snippets=snippets,
                        final_answer=r["final_answer"] or "",
                        ts=r["ts"],
                        latency_ms=r["latency_ms"] or 0,
                    )
                )
        return turns

    # ----- aggregate views -----

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._conn() as c:
            row = c.execute(
                "SELECT session_id, created_at, rolling_summary FROM sessions "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return Session(
            session_id=row["session_id"],
            created_at=row["created_at"],
            rolling_summary=row["rolling_summary"] or "",
            messages=self.get_messages(session_id),
            turns=self.get_turns(session_id),
        )
