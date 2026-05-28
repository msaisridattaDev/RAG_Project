"""Entity-type resolver protocol — mirrors autogen.net EntityTypeResolver.

When the LLM extractor returns a raw entity type, this resolver maps it to
the canonical taxonomy term used in the index. Multi-pass extraction can
yield different raw types for the same entity (DRUG vs MEDICATION); the
resolver picks a canonical one and the originals are preserved on
``EntityNode.historical_entity_types`` for audit (Phase 3 Day 12).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EntityTypeResolver(Protocol):
    """Maps raw entity types to canonical types.

    Implementations may be table-driven (a static dictionary), LLM-driven
    (a small classifier prompt), or hybrid.
    """

    async def resolve_canonical_type(
        self,
        raw_type: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Resolve ``raw_type`` to its canonical form.

        Args:
            raw_type: The type the extractor produced (e.g., ``"MEDICATION"``).
            context: Optional disambiguation hints (entity name, surrounding
                text, app_id, etc.).

        Returns:
            The canonical type (e.g., ``"DRUG"``).
        """
        ...
