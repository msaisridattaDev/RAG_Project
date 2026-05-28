"""Embedded (aiosqlite-backed) key-value store — mirrors autogen.net's LiteDB KvStore.

Daily-rotated cache file pattern: CachingClient_{yyyyMMdd}.db

Uses WAL mode for concurrent reads, transactional writes. File rotation
provides automatic TTL — yesterday's entries naturally expire when that
day's file is cleaned up by the rotation cron.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from structlog import get_logger

logger = get_logger(__name__)


class EmbeddedKvCache:
    """File-backed KV store using aiosqlite.

    One database file per day. The calling code provides the directory;
    this class derives the daily file path from today's UTC date.

    Thread/async safety: writes serialize via aiosqlite (single-writer
    by design); readers can overlap with WAL mode enabled.
    """

    SQL_SCHEMA = """
    CREATE TABLE IF NOT EXISTS kv (
        key   TEXT PRIMARY KEY NOT NULL,
        value TEXT NOT NULL
    );
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=NORMAL;
    """

    def __init__(self, directory: Path) -> None:
        """Args:
            directory: The directory containing daily cache files.
                e.g. Path("./cache/OpenLM")
        """
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _today_file_name() -> str:
        """Return today's cache file name: CachingClient_YYYYMMDD.db."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"CachingClient_{today}.db"

    def _today_path(self) -> Path:
        return self._directory / self._today_file_name()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str) -> str | None:
        """Read a value from today's cache. Returns None on miss."""
        path = self._today_path()
        try:
            async with aiosqlite.connect(str(path)) as db:
                await db.execute("PRAGMA journal_mode=WAL;")
                cursor = await db.execute(
                    "SELECT value FROM kv WHERE key = ?", (key,)
                )
                row = await cursor.fetchone()
                return row[0] if row else None
        except Exception as exc:
            logger.debug("kv_cache.get_miss", key=key[:16], error=str(exc))
            return None

    async def set(self, key: str, value: str) -> None:
        """Insert or replace a key under transactional lock."""
        path = self._today_path()
        async with self._lock:
            try:
                async with aiosqlite.connect(str(path)) as db:
                    await db.execute("PRAGMA journal_mode=WAL;")
                    await db.execute("PRAGMA synchronous=NORMAL;")
                    await db.execute(
                        "CREATE TABLE IF NOT EXISTS kv ("
                        "key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)"
                    )
                    await db.execute(
                        "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                        (key, value),
                    )
                    await db.commit()
            except Exception as exc:
                logger.warning("kv_cache.set_failed", key=key[:16], error=str(exc))
                raise

    async def delete_old_files(self, max_age_days: int = 7) -> list[str]:
        """Remove cache files older than *max_age_days*.

        Returns the list of deleted file names (for logging).
        Called by the daily cleanup cron.
        """
        import time as time_mod

        cutoff = time_mod.time() - (max_age_days * 86400)
        deleted: list[str] = []
        for entry in self._directory.iterdir():
            if entry.name.startswith("CachingClient_") and entry.suffix == ".db":
                try:
                    mtime = entry.stat().st_mtime
                    if mtime < cutoff:
                        entry.unlink()
                        deleted.append(entry.name)
                except OSError:
                    pass
        if deleted:
            logger.info("kv_cache.cleanup", deleted_count=len(deleted))
        return deleted