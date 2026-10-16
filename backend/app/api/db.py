"""SQLite storage layer for sessions, messages, and feedback using aiosqlite in WAL mode.

Includes an explicit shared write lock to serialize write transactions across concurrent streams,
preventing 'database is locked' errors while reads run lock-free in WAL mode.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.app.api.rag_bridge import HistoryMessage

try:
    import aiosqlite  # type: ignore
except ImportError:
    aiosqlite = None


DEFAULT_DB_PATH = os.getenv("SQLITE_PATH", "runtime/aitunator.db")
SESSION_RETENTION_DAYS = int(os.getenv("SESSION_RETENTION_DAYS", "30"))
HISTORY_WINDOW = 10


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources_json TEXT,
    program_code TEXT,
    finish_reason TEXT,
    diagnostics_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_sessions_last_active
    ON sessions(last_active_at);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_message
    ON feedback(message_id);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _StdlibAsyncCursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    async def fetchone(self) -> sqlite3.Row | None:
        return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self) -> list[sqlite3.Row]:
        return await asyncio.to_thread(self._cursor.fetchall)

    async def close(self) -> None:
        await asyncio.to_thread(self._cursor.close)


class _StdlibAsyncConnection:
    """Async wrapper over sqlite3.Connection matching aiosqlite.Connection semantics."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> _StdlibAsyncConnection:
        self._conn = await asyncio.to_thread(
            sqlite3.connect, self._db_path, timeout=10.0, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def execute(self, sql: str, parameters: tuple | list = ()) -> _StdlibAsyncCursor:
        assert self._conn is not None
        async with self._lock:
            cur = await asyncio.to_thread(self._conn.execute, sql, parameters)
            return _StdlibAsyncCursor(cur)

    async def executescript(self, sql_script: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await asyncio.to_thread(self._conn.executescript, sql_script)

    async def commit(self) -> None:
        assert self._conn is not None
        async with self._lock:
            await asyncio.to_thread(self._conn.commit)

    async def rollback(self) -> None:
        assert self._conn is not None
        async with self._lock:
            await asyncio.to_thread(self._conn.rollback)


class _AiosqliteConnectionWrapper:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: Any = None

    async def __aenter__(self) -> Any:
        self._conn = await aiosqlite.connect(self._db_path, timeout=10.0, isolation_level=None)
        self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


def _connect(db_path: str | Path):
    if aiosqlite is not None:
        return _AiosqliteConnectionWrapper(db_path)
    return _StdlibAsyncConnection(db_path)


class Database:
    """Async SQLite repository with WAL mode and shared write serialization."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or os.getenv("SQLITE_PATH", DEFAULT_DB_PATH))
        self._write_lock = asyncio.Lock()

    async def init(self, retention_days: int = SESSION_RETENTION_DAYS) -> int:
        """Enable WAL, create schema, and delete expired sessions."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._write_lock, _connect(self.db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA synchronous=NORMAL;")
            await conn.execute("PRAGMA foreign_keys=ON;")
            await conn.execute("PRAGMA busy_timeout=10000;")
            await conn.executescript(SCHEMA_SQL)
            await conn.execute("PRAGMA foreign_keys=ON;")
        return await self.cleanup_expired_sessions(days=retention_days)

    async def _prepare(self, conn: Any) -> None:
        await conn.execute("PRAGMA foreign_keys=ON;")
        await conn.execute("PRAGMA busy_timeout=10000;")

    async def cleanup_expired_sessions(self, days: int = SESSION_RETENTION_DAYS) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        async with self._write_lock, _connect(self.db_path) as conn:
            await self._prepare(conn)
            await conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = await conn.execute(
                    "DELETE FROM sessions WHERE last_active_at < ?;",
                    (cutoff,),
                )
                deleted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
                await conn.execute("COMMIT;")
                return deleted
            except Exception:
                await conn.execute("ROLLBACK;")
                raise

    async def get_recent_history(
        self, session_id: str, limit: int = HISTORY_WINDOW
    ) -> list[HistoryMessage]:
        """Read-only: runs concurrently in WAL mode without acquiring write lock."""
        async with _connect(self.db_path) as conn:
            await self._prepare(conn)
            cur = await conn.execute(
                """
                SELECT role, content
                FROM (
                    SELECT role, content, created_at, rowid
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, rowid ASC;
                """,
                (session_id, limit),
            )
            rows = await cur.fetchall()
            return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def save_completed_turn(
        self,
        *,
        session_id: str,
        question: str,
        answer_text: str,
        program_code: str | None = None,
        finish_reason: str = "stop",
        diagnostics: dict | None = None,
    ) -> str:
        """Atomically persist session, user question, and assistant answer under write lock."""
        now_user = _utc_now_iso()
        now_assistant = _utc_now_iso()
        user_msg_id = str(uuid.uuid4())
        assistant_msg_id = str(uuid.uuid4())

        diagnostics_dict = diagnostics if isinstance(diagnostics, dict) else {}
        sources_list = diagnostics_dict.get("sources") if isinstance(diagnostics_dict.get("sources"), list) else []
        sources_json = json.dumps(sources_list, ensure_ascii=False)
        diagnostics_json = json.dumps(diagnostics_dict, ensure_ascii=False) if diagnostics_dict else None

        async with self._write_lock, _connect(self.db_path) as conn:
            await self._prepare(conn)
            await conn.execute("BEGIN IMMEDIATE;")
            try:
                await conn.execute(
                    """
                    INSERT INTO sessions (id, created_at, last_active_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET last_active_at = excluded.last_active_at;
                    """,
                    (session_id, now_user, now_assistant),
                )
                await conn.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content, sources_json, program_code, finish_reason, diagnostics_json, created_at)
                    VALUES (?, ?, 'user', ?, NULL, ?, NULL, NULL, ?);
                    """,
                    (user_msg_id, session_id, question, program_code, now_user),
                )
                await conn.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content, sources_json, program_code, finish_reason, diagnostics_json, created_at)
                    VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        assistant_msg_id,
                        session_id,
                        answer_text,
                        sources_json,
                        program_code,
                        finish_reason,
                        diagnostics_json,
                        now_assistant,
                    ),
                )
                await conn.execute("COMMIT;")
                return assistant_msg_id
            except Exception:
                await conn.execute("ROLLBACK;")
                raise

    async def get_session_history(self, session_id: str) -> dict[str, Any] | None:
        """Read-only: runs concurrently in WAL mode without acquiring write lock."""
        async with _connect(self.db_path) as conn:
            await self._prepare(conn)
            cur = await conn.execute(
                "SELECT id, created_at, last_active_at FROM sessions WHERE id = ?;",
                (session_id,),
            )
            session_row = await cur.fetchone()
            if session_row is None:
                return None

            msg_cur = await conn.execute(
                """
                SELECT
                    m.id,
                    m.role,
                    m.content,
                    m.program_code,
                    m.finish_reason,
                    m.created_at,
                    f.rating AS feedback_rating,
                    f.comment AS feedback_comment
                FROM messages m
                LEFT JOIN feedback f ON f.message_id = m.id
                WHERE m.session_id = ?
                ORDER BY m.created_at ASC, m.rowid ASC;
                """,
                (session_id,),
            )
            rows = await msg_cur.fetchall()
            messages = [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "program_code": row["program_code"],
                    "finish_reason": row["finish_reason"],
                    "created_at": row["created_at"],
                    "feedback": (
                        {
                            "rating": row["feedback_rating"],
                            "comment": row["feedback_comment"],
                        }
                        if row["feedback_rating"] is not None
                        else None
                    ),
                }
                for row in rows
            ]
            return {
                "session_id": session_row["id"],
                "created_at": session_row["created_at"],
                "last_active_at": session_row["last_active_at"],
                "messages": messages,
            }

    async def save_feedback(
        self,
        *,
        message_id: str,
        rating: int,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Save feedback under shared write lock."""
        now = _utc_now_iso()
        async with self._write_lock, _connect(self.db_path) as conn:
            await self._prepare(conn)
            await conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = await conn.execute(
                    "SELECT id, session_id, role FROM messages WHERE id = ?;",
                    (message_id,),
                )
                msg = await cur.fetchone()
                if msg is None:
                    raise KeyError(f"Message '{message_id}' not found")
                if msg["role"] != "assistant":
                    raise ValueError("Feedback can only be attached to assistant messages")

                session_id = msg["session_id"]
                await conn.execute(
                    """
                    INSERT INTO feedback (message_id, session_id, rating, comment, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(message_id) DO UPDATE SET
                        rating = excluded.rating,
                        comment = excluded.comment,
                        created_at = excluded.created_at;
                    """,
                    (message_id, session_id, rating, comment, now),
                )
                await conn.execute(
                    "UPDATE sessions SET last_active_at = ? WHERE id = ?;",
                    (now, session_id),
                )
                await conn.execute("COMMIT;")
                return {
                    "status": "ok",
                    "message_id": message_id,
                    "session_id": session_id,
                    "rating": rating,
                    "comment": comment,
                    "created_at": now,
                }
            except Exception:
                await conn.execute("ROLLBACK;")
                raise

    async def check_health(self) -> tuple[bool, str]:
        try:
            async with _connect(self.db_path) as conn:
                cur = await conn.execute("PRAGMA journal_mode;")
                row = await cur.fetchone()
                mode = str(row[0]).lower() if row else "unknown"
                return True, mode
        except Exception as exc:
            return False, str(exc)
