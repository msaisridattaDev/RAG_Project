"""Tests for the DI container and providers."""

from __future__ import annotations

import pytest

from autogen.di.container import ServiceContainer


class TestServiceContainer:
    """Verify ServiceContainer registration and resolution."""

    def test_register_and_resolve_singleton(self) -> None:
        """A registered singleton should return the same instance."""
        container = ServiceContainer()
        instance = {"key": "value"}
        container.register_singleton(dict, instance)
        resolved = container.resolve(dict)
        assert resolved is instance

    def test_register_and_resolve_factory(self) -> None:
        """A registered factory should create a new instance each time."""
        container = ServiceContainer()
        call_count = 0

        def factory() -> dict:
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        container.register_factory(dict, factory)
        first = container.resolve(dict)
        second = container.resolve(dict)
        assert first["count"] == 1
        assert second["count"] == 2
        assert first is not second

    def test_resolve_unregistered_raises(self) -> None:
        """Resolving an unregistered key should raise KeyError."""
        container = ServiceContainer()
        with pytest.raises(KeyError, match="not registered"):
            container.resolve("nonexistent")

    def test_string_key_registration(self) -> None:
        """Services can be registered and resolved by string key."""
        container = ServiceContainer()
        container.register_singleton("my.service", 42)
        assert container.resolve("my.service") == 42
