"""EntityRelationProcessor — LLM summarization of bloated entity/relation descriptions.

Phase 3 Day 15 Stage 5.  Mirrors EntityRelationProcessor.cs.

When an entity has been merged across many chunks, its concatenated
descriptions can exceed EntitySummaryToMaxTokens (default 500).
Embedding raw concatenated text drifts toward "everything-ever-said-about-X"
noise rather than a clean semantic centroid.

This processor:
    1. Tokenizes the current description.
    2. If tokens > threshold, calls a small LLM to produce a concise summary.
    3. Replaces entity.description (or relation.description) with the summary
       BUT preserves the descriptions[] list as audit trail.

Usage::

    proc = EntityRelationProcessor(llm, model="gpt-4o-mini", max_tokens=500)
    await proc.summarize_entity(entity_node)   # mutates entity_node.description
    await proc.summarize_relation(relation)    # mutates relation.description
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from autogen.extraction.prompts import MEDICAL_ENTITY_TYPES

if TYPE_CHECKING:
    from autogen.models.storage import EntityNode, EntityRelation
    from autogen.protocols.llm import LlmClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt for summarization
# ---------------------------------------------------------------------------
_SUMMARIZE_ENTITY_PROMPT = """\
You are a medical knowledge editor. Below are multiple descriptions of the same
medical entity collected from different sources. Produce a SINGLE concise,
comprehensive description that captures all the important information.

Entity name: {entity_name}
Entity type: {entity_type}

Descriptions:
{descriptions}

Concise summary (2-4 sentences, preserving all key facts):
"""

_SUMMARIZE_RELATION_PROMPT = """\
You are a medical knowledge editor. Below are multiple descriptions of the same
relationship between two medical entities collected from different sources.
Produce a SINGLE concise description that captures the essence of the relationship.

Source entity: {source_name}
Target entity: {target_name}
Keywords: {keywords}

Descriptions:
{descriptions}

Concise summary (1-3 sentences, preserving key mechanism/effect):
"""


class EntityRelationProcessor:
    """Summarize bloated entity/relation descriptions via LLM.

    Token counting relies on tiktoken (cl100k_base) to match the .NET source.
    Installed as an optional dependency — if unavailable, falls back to a
    simple word-count heuristic.
    """

    def __init__(
        self,
        llm: LlmClient,
        *,
        model: str = "gpt-4o-mini",
        max_tokens: int = 500,  # EntitySummaryToMaxTokens default
    ) -> None:
        """
        Args:
            llm: LLM client for summarization calls.
            model: Small/fast model for summarization (default SmallLMModelName).
            max_tokens: If description token count exceeds this, summarize.
        """
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens

        # Lazy-load tokenizer
        self._tokenizer = self._try_load_tokenizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def summarize_entity(self, entity: EntityNode) -> None:
        """If entity.description is too long, replace with concise summary.

        Mutates entity.description in place.  Does NOT touch entity.descriptions[].
        """
        if not entity.descriptions or len(entity.descriptions) <= 1:
            return  # Single description — nothing to summarize.

        combined = entity.description
        if not combined or not combined.strip():
            return

        token_count = self._count_tokens(combined)
        if token_count <= self._max_tokens:
            return

        logger.info(
            "Summarizing entity %s: %d tokens → target ≤%d",
            entity.entity_name,
            token_count,
            self._max_tokens,
        )

        try:
            summary = await self._summarize_entity(
                entity.entity_name,
                entity.entity_type,
                entity.descriptions,
            )
            entity.description = summary.strip()
            logger.debug(
                "Entity %s summary: %d tokens → %d tokens",
                entity.entity_name,
                token_count,
                self._count_tokens(summary),
            )
        except Exception:
            logger.warning(
                "Entity summarization failed for %s; keeping original",
                entity.entity_name,
                exc_info=True,
            )

    async def summarize_relation(self, relation: EntityRelation) -> None:
        """If relation.description is too long, replace with concise summary.

        Mutates relation.description in place.
        """
        if not relation.descriptions or len(relation.descriptions) <= 1:
            return

        combined = relation.description
        if not combined or not combined.strip():
            return

        token_count = self._count_tokens(combined)
        if token_count <= self._max_tokens:
            return

        logger.info(
            "Summarizing relation %s -> %s: %d tokens → target ≤%d",
            relation.source_name,
            relation.target_name,
            token_count,
            self._max_tokens,
        )

        try:
            summary = await self._summarize_relation(
                relation.source_name,
                relation.target_name,
                relation.keywords,
                relation.descriptions,
            )
            relation.description = summary.strip()
            logger.debug(
                "Relation %s->%s summary: %d tokens → %d tokens",
                relation.source_name,
                relation.target_name,
                token_count,
                self._count_tokens(summary),
            )
        except Exception:
            logger.warning(
                "Relation summarization failed for %s->%s; keeping original",
                relation.source_name,
                relation.target_name,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internal — LLM calls
    # ------------------------------------------------------------------
    async def _summarize_entity(
        self,
        name: str,
        entity_type: str,
        descriptions: list[str],
    ) -> str:
        descs_text = "\n".join(f"- {d}" for d in descriptions)
        prompt = _SUMMARIZE_ENTITY_PROMPT.format(
            entity_name=name,
            entity_type=entity_type,
            descriptions=descs_text,
        )
        return await self._llm.complete(
            [{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            model=self._model,
        )

    async def _summarize_relation(
        self,
        source_name: str,
        target_name: str,
        keywords: list[str],
        descriptions: list[str],
    ) -> str:
        descs_text = "\n".join(f"- {d}" for d in descriptions)
        kw_str = ", ".join(keywords) if keywords else "(none)"
        prompt = _SUMMARIZE_RELATION_PROMPT.format(
            source_name=source_name,
            target_name=target_name,
            keywords=kw_str,
            descriptions=descs_text,
        )
        return await self._llm.complete(
            [{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            model=self._model,
        )

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------
    @staticmethod
    def _try_load_tokenizer():
        """Try to load tiktoken's cl100k_base; fall back to None."""
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning(
                "tiktoken not available; using word-count heuristic for "
                "EntityRelationProcessor token threshold."
            )
            return None

    def _count_tokens(self, text: str) -> int:
        """Count tokens with tiktoken or fall back to word-count × 1.3."""
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        # Rough: ~1.3 tokens per word for English medical text
        return int(len(text.split()) * 1.3)