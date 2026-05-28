"""Lazy service container / registry — mirrors autogen.net DI container.

Provides a simple service locator for registering and resolving
singleton and factory-scoped services. Used as a fallback when
FastAPI's Depends is insufficient (e.g., for background tasks).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class ServiceContainer:
    """Lazy service registry — mirrors autogen.net DI container pattern.

    Supports:
    - Singleton registration (one instance, created on first resolve)
    - Factory registration (new instance per resolve)
    - Generic type-safe resolution via resolve[T]()
    """

    def __init__(self) -> None:
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Callable[[], Any]] = {}
        self._singleton_instances: dict[str, Any] = {}

    def register_singleton(self, key: type[T] | str, instance: T) -> None:
        """Register a singleton instance.

        Args:
            key: The type or string key to register under.
            instance: The singleton instance.
        """
        k = self._key(key)
        self._singletons[k] = instance

    def register_factory(self, key: type[T] | str, factory: Callable[[], T]) -> None:
        """Register a factory function.

        Args:
            key: The type or string key to register under.
            factory: A callable that creates a new instance.
        """
        k = self._key(key)
        self._factories[k] = factory

    def resolve(self, key: type[T] | str) -> T:
        """Resolve a service by type or string key.

        Args:
            key: The type or string key to resolve.

        Returns:
            The resolved service instance.

        Raises:
            KeyError: If the service is not registered.
        """
        k = self._key(key)

        # Check singletons first
        if k in self._singletons:
            return self._singletons[k]  # type: ignore[return-value]

        # Check factories
        if k in self._factories:
            return self._factories[k]()  # type: ignore[return-value]

        msg = f"Service not registered: {k}"
        raise KeyError(msg)

    def _key(self, key: type[T] | str) -> str:
        return key if isinstance(key, str) else f"{key.__module__}.{key.__qualname__}"
