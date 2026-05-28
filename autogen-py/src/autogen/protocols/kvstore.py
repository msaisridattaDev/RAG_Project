"""Key-value storage protocol — mirrors autogen.net IKeyValueStorage<T>.

Generic per-app KV store used by Phase 3 for the FullDoc / chunk-by-id /
entity-by-id lookup paths. A concrete LiteDB-style implementation will
be added in Phase 2 Day 8.

The namespace is bound at construction time (``{label}_{app_id}``) so
every operation below stays within the per-tenant partition automatically.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class KeyValueStorage(Protocol[T]):
    """Per-app KV store — mirrors autogen.net ``IKeyValueStorage<T>``."""

    @property
    def namespace(self) -> str:
        """The per-(label, app_id) namespace this store is bound to."""
        ...

    async def get(self, key: str) -> T | None:
        """Return the stored value or None if missing."""
        ...

    async def upsert(self, key: str, value: T) -> None:
        """Insert or overwrite ``key`` → ``value``."""
        ...

    async def filter_keys(self, keys: list[str]) -> list[str]:
        """Return the subset of ``keys`` that are NOT present in the store.

        Mirrors the .NET source's ``FilterKeys`` helper used by the chunk
        ingestion pipeline (Phase 3 Day 11) to skip already-indexed chunks.
        """
        ...
