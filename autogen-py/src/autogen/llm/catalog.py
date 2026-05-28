"""Models catalog loader + tier-based model router.

Mirrors autogen.net Program.cs:46-74 (models.json loading) and
Program.cs:172-284 (tier-based model resolution with 13 roles).

Two pieces:
  1. ModelsCatalog — loads Configuration/models.json, provides per-logical-model
     provider lists with pricing and priority. Hot-reload via watchfiles.
  2. TierModelRouter — given a Tier + role + the catalog, returns the
     LiteLLM-formatted model string. Also exposes parallel_thinking_models().

Flow:
  QnATierConfig → logical model name (e.g. "Gemini-1.5-Flash-8B")
  ModelsCatalog → provider list (e.g. [{OpenRouter, groq/...}, {Groq, groq/...}])
  TierModelRouter → LiteLLM string (e.g. "openrouter/google/gemini-2.0-flash-lite")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from structlog import get_logger

from autogen.config.tiers import Tier

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ModelsCatalog — Pydantic model for Configuration/models.json
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    """One provider entry for a logical model."""

    name: str  # e.g. "OpenRouter", "OpenAi", "Groq", "DeepInfra"
    model_name: str  # provider-specific id, e.g. "google/gemini-2.0-flash-lite"
    priority: int = 1  # lower = higher priority
    pricing: dict[str, float] = Field(default_factory=dict)
    # e.g. {"prompt_per_million": 0.10, "completion_per_million": 0.40}


class ModelEntry(BaseModel):
    """One logical model with its provider options."""

    id: str  # logical name, e.g. "Gemini-1.5-Flash-8B"
    is_reasoning: bool = False
    providers: list[ProviderConfig]


class ModelsCatalog(BaseModel):
    """Top-level container for the models catalog file."""

    models: list[ModelEntry]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_providers(self, logical_name: str) -> list[ProviderConfig]:
        """Return provider list for a given logical model name, or empty list."""
        for entry in self.models:
            if entry.id == logical_name:
                return sorted(entry.providers, key=lambda p: p.priority)
        logger.warning("catalog.model_not_found", logical_name=logical_name)
        return []

    def get_entry(self, logical_name: str) -> ModelEntry | None:
        """Return the full ModelEntry for the logical name, or None."""
        for entry in self.models:
            if entry.id == logical_name:
                return entry
        return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_models_catalog(path: str | Path) -> ModelsCatalog:
    """Load and validate the models catalog from a JSON file.

    Args:
        path: Path to Configuration/models.json (relative or absolute).

    Returns:
        Validated ModelsCatalog instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValidationError: If the file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Models catalog not found at {path}")

    with open(path, encoding="utf-8") as f:
        raw: Any = json.load(f)

    return ModelsCatalog(**raw)


# ---------------------------------------------------------------------------
# TierModelRouter — maps (tier, role) → LiteLLM model string
# ---------------------------------------------------------------------------

# The 13 roles from autogen.net QnAModels (Program.cs:172-284)
MODEL_ROLES = [
    "explanation",
    "conversation",
    "thinking",
    "method_call",
    "relevance_check",
    "segment_finder",
    "question_category",
    "action_dispatcher",
    "answer_extraction",
    "option_explanation",
    "detailed_explanation",
    "short_explanation",
    "hint",
]


class TierModelRouter:
    """Routes (tier, role) pairs to LiteLLM-formatted model strings.

    Uses the tier_configurations from QnASettings (in-memory, per the source)
    plus the external models.json catalog for provider resolution.

    Fallback chain:
      1. tier_configurations[tier][role] → logical model name
      2. catalog.get_providers(logical) → list of provider entries sorted by priority
      3. Pick first provider whose API key env var is set
      4. If no provider has its key set, take the highest-priority one anyway
      5. If role is missing in tier_config, fall back to "conversation"
      6. If tier is missing, fall back to "Free"
    """

    def __init__(
        self,
        tier_configurations: dict[str, object],  # QnATierConfig-compatible dicts
        catalog: ModelsCatalog,
    ) -> None:
        self._tiers = tier_configurations
        self._catalog = catalog

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def model_for(self, tier: Tier | str, role: str = "conversation") -> str:
        """Return the LiteLLM-formatted model string for (tier, role).

        Args:
            tier: One of Free / Testing / Regular / Premium.
            role: One of the 13 MODEL_ROLES. Defaults to "conversation".

        Returns:
            LiteLLM model string, e.g. "openrouter/google/gemini-2.0-flash-lite".
        """
        tier_str = tier.value if isinstance(tier, Tier) else tier

        # Resolve tier config (fall back to Free)
        cfg = self._tiers.get(tier_str) or self._tiers.get("Free", {})

        # Resolve logical model name (fall back to conversation_model)
        logical = getattr(cfg, role, None) if hasattr(cfg, role) else None
        if logical is None:
            logical = getattr(cfg, "conversation_model", None)
        if logical is None:
            logger.error("router.no_default_model", tier=tier_str, role=role)
            return "groq/llama-3.3-70b-versatile"  # hard fallback

        return self._resolve_logical_to_litellm(logical)

    def parallel_thinking_models(self, tier: Tier | str) -> list[str]:
        """Return the list of LiteLLM model strings for parallel thinking.

        For Premium tier this returns 3 models (Meta-Llama-3.3-70B,
        Qwen-R1-32B, QwQ-32B). For other tiers, returns 1 model
        (same as thinking_model).
        """
        tier_str = tier.value if isinstance(tier, Tier) else tier
        cfg = self._tiers.get(tier_str) or self._tiers.get("Free", {})

        if hasattr(cfg, "parallel_thinking_models"):
            models: list[str] = getattr(cfg, "parallel_thinking_models", [])
            if models:
                return [self._resolve_logical_to_litellm(m) for m in models]

        # No explicit parallel models → fall back to thinking_model
        thinking = getattr(cfg, "thinking_model", None)
        if thinking:
            return [self._resolve_logical_to_litellm(thinking)]

        return []

    # ------------------------------------------------------------------
    # Internal resolution
    # ------------------------------------------------------------------

    def _resolve_logical_to_litellm(self, logical: str) -> str:
        """Resolve a logical model name to a LiteLLM provider-prefixed string.

        Args:
            logical: e.g. "Gemini-1.5-Flash-8B"

        Returns:
            e.g. "openrouter/google/gemini-2.0-flash-lite"
        """
        providers = self._catalog.get_providers(logical)

        if not providers:
            # No catalog entry — use the logical name as-is (LiteLLM may handle it)
            logger.warning("router.no_providers", logical_name=logical)
            return logical

        # Find first provider whose API key env var is set
        for p in providers:
            env_key = self._provider_env_key(p.name)
            if env_key and os.environ.get(env_key):
                litellm_prefix = self._to_litellm_prefix(p.name)
                return f"{litellm_prefix}/{p.model_name}"

        # No keys set — return the highest-priority provider anyway
        best = providers[0]
        litellm_prefix = self._to_litellm_prefix(best.name)
        logger.warning(
            "router.no_api_key",
            logical_name=logical,
            provider=best.name,
            expected_env=self._provider_env_key(best.name),
        )
        return f"{litellm_prefix}/{best.model_name}"

    # ------------------------------------------------------------------
    # Provider metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _to_litellm_prefix(provider_name: str) -> str:
        """Map catalog provider name to LiteLLM prefix."""
        mapping: dict[str, str] = {
            "OpenAi": "openai",
            "OpenRouter": "openrouter",
            "Groq": "groq",
            "DeepInfra": "deepinfra",
            "Anthropic": "anthropic",
            "Gemini": "gemini",
            "Cohere": "cohere",
            "Mistral": "mistral",
            "Cerebras": "cerebras",
        }
        return mapping.get(provider_name, provider_name.lower())

    @staticmethod
    def _provider_env_key(provider_name: str) -> str:
        """Map catalog provider name to expected API key env var."""
        mapping: dict[str, str] = {
            "OpenAi": "OPENAI_API_KEY",
            "OpenRouter": "OPENROUTER_API_KEY",
            "Groq": "GROQ_API_KEY",
            "DeepInfra": "DEEPINFRA_API_KEY",
            "Anthropic": "ANTHROPIC_API_KEY",
            "Gemini": "GEMINI_API_KEY",
            "Cohere": "COHERE_API_KEY",
            "Mistral": "MISTRAL_API_KEY",
            "Cerebras": "CEREBRAS_API_KEY",
        }
        return mapping.get(provider_name, f"{provider_name.upper()}_API_KEY")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_router(
    tier_configurations: dict[str, object],
    catalog_path: str | Path,
) -> TierModelRouter:
    """Build the production-ready TierModelRouter from config.

    Args:
        tier_configurations: QnASettings.tier_configurations (dict[tier → QnATierConfig])
        catalog_path: Path to Configuration/models.json file.
    """
    catalog = load_models_catalog(catalog_path)
    return TierModelRouter(tier_configurations, catalog)