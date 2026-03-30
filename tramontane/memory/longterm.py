"""SQLite + FTS5 long-term memory for Tramontane.

Auto-creates the database on first write — no ``tramontane init``
required.  Uses WAL mode for concurrent read performance.
"""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class MemoryEntry(BaseModel):
    """A single long-term memory record."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str | None = None
    pipeline_name: str | None = None
    entity_key: str | None = None
    memory_type: str = "fact"
    content: str = ""
    importance: float = 0.5
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    last_accessed: datetime.datetime | None = None
    access_count: int = 0
    expires_at: datetime.datetime | None = None
    erased_at: datetime.datetime | None = None


class LongTermMemory:
    """SQLite-backed long-term memory with FTS5 full-text search.

    Auto-creates the database and schema on first access.
    Uses WAL mode for performance and foreign keys for integrity.
    """

    def __init__(self, db_path: str = "tramontane.db") -> None:
        self._db_path = db_path
        self._db: sqlite3.Connection | None = None

    # -- Lazy DB with auto-schema ------------------------------------------

    def _get_db(self) -> sqlite3.Connection:
        """Return (and cache) the SQLite connection, creating schema if needed."""
        if self._db is not None:
            return self._db

        self._db = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")

        # Auto-apply schema if tables are missing
        cursor = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='tramontane_memory'"
        )
        if cursor.fetchone() is None:
            schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
            self._db.executescript(schema_sql)
            logger.info("Tramontane schema created in %s", self._db_path)

        return self._db

    # -- Store -------------------------------------------------------------

    async def store(
        self,
        content: str,
        entity_key: str,
        memory_type: str,
        user_id: str | None = None,
        pipeline_name: str | None = None,
        importance: float = 0.5,
        ttl_days: int | None = None,
    ) -> MemoryEntry:
        """Store a new memory entry. Auto-creates DB on first call."""
        db = self._get_db()
        entry_id = uuid.uuid4().hex
        now = datetime.datetime.now(datetime.timezone.utc)
        expires_at: datetime.datetime | None = None
        if ttl_days is not None:
            expires_at = now + datetime.timedelta(days=ttl_days)

        db.execute(
            "INSERT INTO tramontane_memory "
            "(id, user_id, pipeline_name, entity_key, memory_type, "
            "content, importance, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry_id, user_id, pipeline_name, entity_key,
                memory_type, content, importance,
                now.isoformat(), expires_at.isoformat() if expires_at else None,
            ),
        )

        # Insert into FTS index
        db.execute(
            "INSERT INTO tramontane_memory_fts "
            "(rowid, content, entity_key, pipeline_name) "
            "VALUES (last_insert_rowid(), ?, ?, ?)",
            (content, entity_key, pipeline_name or ""),
        )
        db.commit()

        return MemoryEntry(
            id=entry_id,
            user_id=user_id,
            pipeline_name=pipeline_name,
            entity_key=entity_key,
            memory_type=memory_type,
            content=content,
            importance=importance,
            created_at=now,
            expires_at=expires_at,
        )

    # -- Search ------------------------------------------------------------

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Full-text search via FTS5, filtering expired and erased entries."""
        db = self._get_db()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        sql = (
            "SELECT m.* FROM tramontane_memory m "
            "JOIN tramontane_memory_fts fts ON m.rowid = fts.rowid "
            "WHERE tramontane_memory_fts MATCH ? "
            "AND m.erased_at IS NULL "
            "AND (m.expires_at IS NULL OR m.expires_at > ?) "
        )
        params: list[Any] = [query, now_iso]

        if user_id is not None:
            sql += "AND m.user_id = ? "
            params.append(user_id)
        if memory_type is not None:
            sql += "AND m.memory_type = ? "
            params.append(memory_type)

        sql += "ORDER BY m.importance DESC, m.access_count DESC LIMIT ?"
        params.append(limit)

        cursor = db.execute(sql, params)
        rows = cursor.fetchall()

        entries: list[MemoryEntry] = []
        for row in rows:
            # Update access stats
            db.execute(
                "UPDATE tramontane_memory SET last_accessed = ?, "
                "access_count = access_count + 1 WHERE id = ?",
                (now_iso, row["id"]),
            )
            entries.append(self._row_to_entry(row))

        db.commit()
        return entries

    # -- Get by entity -----------------------------------------------------

    async def get_by_entity(
        self,
        entity_key: str,
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve all memories for a given entity key."""
        db = self._get_db()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        sql = (
            "SELECT * FROM tramontane_memory "
            "WHERE entity_key = ? AND erased_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > ?) "
        )
        params: list[Any] = [entity_key, now_iso]

        if user_id is not None:
            sql += "AND user_id = ? "
            params.append(user_id)

        sql += "ORDER BY importance DESC"
        cursor = db.execute(sql, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    # -- GDPR Article 17: Right to Erasure ---------------------------------

    async def erase_user(
        self,
        user_id: str,
        requested_by: str = "user",
    ) -> int:
        """Erase all data for a user (GDPR Article 17).

        Sets erased_at timestamp, removes from FTS index, and logs
        the erasure event. Returns count of erased entries.
        """
        db = self._get_db()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Count entries to erase
        cursor = db.execute(
            "SELECT COUNT(*) FROM tramontane_memory "
            "WHERE user_id = ? AND erased_at IS NULL",
            (user_id,),
        )
        count: int = cursor.fetchone()[0]

        if count == 0:
            return 0

        # Get rowids for FTS cleanup
        cursor = db.execute(
            "SELECT rowid FROM tramontane_memory "
            "WHERE user_id = ? AND erased_at IS NULL",
            (user_id,),
        )
        rowids = [row[0] for row in cursor.fetchall()]

        # Mark as erased (soft delete — audit trail preserved)
        db.execute(
            "UPDATE tramontane_memory SET erased_at = ?, content = '[ERASED]' "
            "WHERE user_id = ? AND erased_at IS NULL",
            (now_iso, user_id),
        )

        # Remove from FTS index
        for rowid in rowids:
            db.execute(
                "DELETE FROM tramontane_memory_fts WHERE rowid = ?",
                (rowid,),
            )

        # Log erasure
        db.execute(
            "INSERT INTO tramontane_erasure_log "
            "(id, user_id, erased_at, erased_count, requested_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, user_id, now_iso, count, requested_by),
        )
        db.commit()

        logger.info(
            "GDPR erasure: %d entries erased for user %s", count, user_id
        )
        return count

    # -- Expiry cleanup ----------------------------------------------------

    async def expire_old(self) -> int:
        """Delete entries past their expiry date. Returns count deleted."""
        db = self._get_db()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        cursor = db.execute(
            "SELECT COUNT(*) FROM tramontane_memory "
            "WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now_iso,),
        )
        count: int = cursor.fetchone()[0]

        if count > 0:
            db.execute(
                "DELETE FROM tramontane_memory "
                "WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now_iso,),
            )
            db.commit()

        return count

    # -- Stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return database statistics."""
        db = self._get_db()

        total = db.execute(
            "SELECT COUNT(*) FROM tramontane_memory WHERE erased_at IS NULL"
        ).fetchone()[0]

        users = db.execute(
            "SELECT COUNT(DISTINCT user_id) FROM tramontane_memory "
            "WHERE erased_at IS NULL AND user_id IS NOT NULL"
        ).fetchone()[0]

        oldest = db.execute(
            "SELECT MIN(created_at) FROM tramontane_memory "
            "WHERE erased_at IS NULL"
        ).fetchone()[0]

        db_size_bytes = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0

        return {
            "total_entries": total,
            "total_users": users,
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
            "oldest_entry": oldest,
        }

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        """Convert a sqlite3.Row to a MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            user_id=row["user_id"],
            pipeline_name=row["pipeline_name"],
            entity_key=row["entity_key"],
            memory_type=row["memory_type"],
            content=row["content"],
            importance=row["importance"],
            access_count=row["access_count"],
        )
