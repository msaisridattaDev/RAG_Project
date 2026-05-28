"""EntityTypeResolver — canonical type normalization via LLM (Phase 3 Day 12).

Mirrors EntityTypeResolver.cs: takes raw type strings from LLM extraction
("DRUG", "Pharmaceutical", "medication") and maps them to a canonical type
from the MEDICAL_ENTITY_TYPES list.  Raw types are preserved in each entity's
``historical_entity_types[]`` audit trail.

Cache: successful resolutions are cached in an in-memory dict so the LLM is
only called once per distinct (raw_type, context_prefix).

Fallback: if the LLM doesn't return a canonical type, or the call fails,
return "CONCEPT".
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from autogen.extraction.prompts import ENTITY_TYPE_RESOLVER_PROMPT, MEDICAL_ENTITY_TYPES

if TYPE_CHECKING:
    from autogen.protocols.llm import LlmClient

logger = logging.getLogger(__name__)


class EntityTypeResolver:
    """LLM-backed resolver that maps raw entity types to canonical types.

    Stateful (in-memory cache) — designed to be instantiated once per pipeline
    run and shared across all extraction tasks.

    Usage::

        resolver = EntityTypeResolver(llm, model="gpt-4o-mini")
        canonical = await resolver.resolve_canonical_type("Pharmaceutical", "Aspirin...")
        # → "DRUG"
        canonical2 = await resolver.resolve_canonical_type("Pharmaceutical")
        # → "DRUG" (cached, no LLM call)
    """

    # ------------------------------------------------------------------
    # Cache key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(raw_type: str, context: str = "") -> str:
        """Build a cache key from the raw type and a context prefix."""
        prefix = context[:80].strip().lower() if context else ""
        return f"{raw_type.lower().strip()}|{prefix}"

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        llm: LlmClient,
        *,
        model: str = "gpt-4o-mini",
        valid_types: list[str] | None = None,
    ) -> None:
        """
        Args:
            llm: LLM client for the resolution calls.
            model: Model to use for type resolution (cheap/fast is fine).
            valid_types: Canonical type list. Defaults to MEDICAL_ENTITY_TYPES.
        """
        self._llm = llm
        self._model = model
        self._valid_types = valid_types or list(MEDICAL_ENTITY_TYPES)
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def resolve_canonical_type(
        self, raw_type: str, context: str = ""
    ) -> str:
        """Return the canonical type for *raw_type*, with optional context.

        Args:
            raw_type: The raw type string from LLM extraction (e.g. "Pharmaceutical").
            context: Optional entity description or surrounding text for disambiguation.

        Returns:
            A canonical type from the valid list, or "CONCEPT" on failure.
        """
        if not raw_type or not raw_type.strip():
            return "CONCEPT"

        # Fast path: already canonical
        cleaned = raw_type.strip()
        if cleaned.upper() in (t.upper() for t in self._valid_types):
            return cleaned.upper()

        # Cache check
        key = self._cache_key(cleaned, context)
        if key in self._cache:
            logger.debug("Type resolution cache hit: %s → %s", key, self._cache[key])
            return self._cache[key]

        # LLM resolution
        try:
            canonical = await self._resolve_via_llm(cleaned, context)
        except Exception:
            logger.warning(
                "LLM type resolution failed for raw_type=%r; falling back to CONCEPT",
                cleaned,
                exc_info=True,
            )
            canonical = "CONCEPT"

        self._cache[key] = canonical
        return canonical

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _resolve_via_llm(self, raw_type: str, context: str) -> str:
        """Call the LLM with the EntityTypeResolver prompt."""
        valid_types_str = ", ".join(self._valid_types)
        prompt = ENTITY_TYPE_RESOLVER_PROMPT.format(
            valid_types=valid_types_str,
            raw_type=raw_type,
            context=context if context else "(none)",
        )

        messages = [{"role": "user", "content": prompt}]
        response = await self._llm.complete(
            [{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            model=self._model,
        )
        resolved = response.strip().strip('"').strip("'").strip()

        # Validate against canonical list
        if resolved.upper() in (t.upper() for t in self._valid_types):
            return resolved.upper()

        logger.debug(
            "LLM returned non-canonical type %r for raw_type=%r; "
            "falling back to CONCEPT",
            resolved,
            raw_type,
        )
        return "CONCEPT"