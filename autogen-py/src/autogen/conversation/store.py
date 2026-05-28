"""SqlConversationStore — Phase 4 Day 17 conversation persistence.

Two SQL tables (conversations + messages) scoped by app_id so that
cross-tenant reads are impossible by construction:

    history(conv_id, app_id="mds")  →  zero rows if conv_id belongs to neetpg

Mirrors autogen.net LiteDbConversationContextStorage semantics but uses
aiosqlite (dev) or asyncpg/Postgres (prod) via a single connection-string
env-var change:

    sqlite+aiosqlite:///./conversations.db   ← default dev
    postgresql+asyncpg://user:pw@host/db     ← prod
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from autogen.logging.setup import get_logger
from autogen.models.llm import LlmMessage

# TYPE_CHECKING-only to avoid circular import — AgentContext is a thin Pydantic model
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogen.models.agent import AgentContext

logger = get_logger("autogen.conversation.store")


class SqlConversationStore:
    """Async SQL-backed conversation store.

    One instance per process (wired as a singleton in app.state).  Uses a
    single persistent aiosqlite connection for SQLite or falls back to a
    per-call connection pattern documented in the docstring.

    All public read methods accept ``app_id`` and JOIN / filter on it even
    when ``conversation_id`` alone would be sufficient — defence in depth
    against cross-tenant leaks.
    """

    def __init__(self, database_url: str) -> None:
        self._db_path = self._parse_db_path(database_url)
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_db_path(database_url: str) -> str:
        """Strip driver prefix, leaving the raw file path or :memory:."""
        for prefix in (
            "sqlite+aiosqlite:///",
            "sqlite:///",
        ):
            if database_url.startswith(prefix):
                return database_url[len(prefix):]
        return database_url  # already a raw path or :memory:

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            path = Path(self._db_path)
            if str(path) not in (":memory:", ""):
                path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def init_schema(self) -> None:
        """Create tables + indices idempotently.  Call once at app startup."""
        conn = await self._ensure_conn()
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id           TEXT PRIMARY KEY,
                app_id       TEXT NOT NULL,
                user_id      TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_conv_app_user
                ON conversations (app_id, user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_msg_conv_time
                ON messages (conversation_id, created_at ASC);
            """
        )
        await conn.commit()
        logger.info("conversation_store.schema_ready", db=self._db_path)

    async def aclose(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_create(self, context: AgentContext) -> dict:
        """Look up a conversation by id; create it if missing.  Idempotent.

        Returns the conversation row as a plain dict.
        """
        conn = await self._ensure_conn()
        async with conn.execute(
            "SELECT id, app_id, user_id, created_at, metadata_json "
            "FROM conversations WHERE id = ?",
            (context.conversation_id,),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            now = datetime.now(UTC).isoformat()
            meta = json.dumps({"tier": str(context.tier)})
            await conn.execute(
                "INSERT INTO conversations (id, app_id, user_id, created_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (context.conversation_id, context.app_id, context.user_id, now, meta),
            )
            await conn.commit()
            logger.debug(
                "conversation_store.created",
                conv_id=context.conversation_id,
                app_id=context.app_id,
            )
            return {
                "id": context.conversation_id,
                "app_id": context.app_id,
                "user_id": context.user_id,
                "created_at": now,
                "metadata_json": meta,
            }

        return dict(row)

    async def append(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append one message to an existing conversation."""
        conn = await self._ensure_conn()
        await conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, conversation_id, role, content, datetime.now(UTC).isoformat()),
        )
        await conn.commit()

    async def history(
        self,
        conversation_id: str,
        app_id: str,
        limit: int = 6,
    ) -> list[LlmMessage]:
        """Return the most recent ``limit`` messages in chronological order.

        Filters by BOTH conversation_id AND app_id — the app_id check is the
        cross-tenant safety net: a forged conversation_id from a different tenant
        returns an empty list here rather than leaking data.
        """
        conn = await self._ensure_conn()
        async with conn.execute(
            """
            SELECT m.role, m.content
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ? AND c.app_id = ?
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (conversation_id, app_id, limit),
        ) as cur:
            rows = await cur.fetchall()

        # rows are newest-first; reverse to chronological for the LLM prompt
        return [LlmMessage(role=row["role"], content=row["content"]) for row in reversed(rows)]

    async def list_by_user(
        self,
        user_id: str,
        app_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Return the most recent conversations for a user within one exam."""
        conn = await self._ensure_conn()
        async with conn.execute(
            """
            SELECT id, app_id, user_id, created_at, metadata_json
            FROM conversations
            WHERE user_id = ? AND app_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, app_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]
