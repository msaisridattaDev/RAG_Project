"""Tests for Settings — verify env loading and defaults.

Tests the nested settings structure that mirrors autogen.net's appsettings.json.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autogen.config.settings import Settings


class TestSettings:
    """Verify that Settings loads correctly from .env and defaults."""

    def test_defaults_used_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no env vars are set, defaults should be used."""
        # Clear any env vars that might interfere
        for key in [
            "ENV",
            "APP_IDENTITY__DEFAULT_APP_ID",
            "LLM_QUERY_AUTH__ALLOWED_TOKEN",
        ]:
            monkeypatch.delenv(key, raising=False)

        # Temporarily move .env out of the way
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            env_path.rename(env_path.with_suffix(".env.bak"))

        try:
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            assert settings.app_identity.default_app_id == "neetpg"
            assert settings.app_identity.allowed_app_ids == ["neetpg", "neetug", "mds", "ems"]
            assert settings.llm_query_auth.header_name == "X-LlmQuery-Token"
            assert settings.llm_query_auth.allowed_token is None
            assert settings.env == "dev"
        finally:
            if env_path.with_suffix(".env.bak").exists():
                env_path.with_suffix(".env.bak").rename(env_path)

    def test_env_vars_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables should override .env and defaults."""
        monkeypatch.setenv("APP_IDENTITY__DEFAULT_APP_ID", "neetug")
        monkeypatch.setenv("ENV", "staging")
        monkeypatch.setenv("LLM_QUERY_AUTH__ALLOWED_TOKEN", "sk-test-key")

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.app_identity.default_app_id == "neetug"
        assert settings.env == "staging"
        assert settings.llm_query_auth.allowed_token == "sk-test-key"

    def test_storage_defaults(self) -> None:
        """Storage substrate defaults should be sensible."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.elasticsearch.url == "http://localhost:9200"
        assert settings.lightrag.neo4j_uri == "bolt://localhost:7687"
        assert settings.cache.base_path == "./cache/OpenLM"

    def test_embedding_defaults(self) -> None:
        """Embedding provider defaults should match autogen.net."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.embedding_options.provider == "jina"
        assert settings.embedding_options.default_model == "jina-embeddings-v3"

    def test_reranking_defaults(self) -> None:
        """Reranking provider defaults should match autogen.net."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.reranking_options.provider == "qwen"
        assert settings.reranking_options.default_model == "Qwen/Qwen3-Reranker-4B"

    def test_qna_defaults(self) -> None:
        """QnA settings should have sensible defaults."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.qna.models_catalog_path == "Configuration/models.json"
        assert settings.qna.tier_configurations == {}
