"""Cache store protocol — mirrors autogen.net ICacheStore<T>.

Defines the interface for a lightweight embedded cache (analogous to
LiteDB in the .NET codebase). Used for session state, intermediate
results, and frequently accessed data.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", covariant=False)


@runtime_checkable
class CacheStore(Protocol[T]):
    """Embedded cache interface — mirrors autogen.net ICacheStore<T>.

    Implementations: LiteDB-style local file store, Redis, in-memory dict.
    Every key is namespaced by appId for multi-tenancy.
    """

    async def get(self, key: str, app_id: str) -> T | None:
        """Retrieve a value from the cache.

        Args:
            key: The cache key.
            app_id: Tenant scope — keys are namespaced by app.

        Returns:
            The cached value if found and not expired, else None.
        """
        ...

    async def set(self, key: str, value: T, app_id: str, ttl_seconds: int = 300) -> None:
        """Store a value in the cache with a TTL.

        Args:
            key: The cache key.
            value: The value to cache.
            app_id: Tenant scope.
            ttl_seconds: Time-to-live in seconds (default 5 minutes).
        """
        ...

    async def delete(self, key: str, app_id: str) -> bool:
        """Remove a value from the cache.

        Args:
            key: The cache key.
            app_id: Tenant scope.

        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        ...
