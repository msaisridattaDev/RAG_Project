"""EntityExtractor — LLM-driven entity/relationship extraction from text chunks.

Phase 3 Days 12-13.  Mirrors EntityExtractor.cs.

For each TextChunk the extractor:
    1. Calls the LLM with the EntityExtraction prompt chain (Day 12).
    2. Runs EntityTypeResolver on each extracted entity's raw type (Day 12).
    3. Runs up to EntityExtractMaxGleaning gleaning passes (Day 13).
    4. Collects missing entity references and recovers them via a targeted
       second LLM pass (Day 13).

Every extracted EntityNode carries:
    - ``app_id``          — the tenant scope.
    - ``historical_entity_types`` — the raw type(s) the LLM assigned,
      even after EntityTypeResolver canonicalises to a standard type.

Public API
----------
    ExtractionResult  — dataclass: (nodes, relations, content_keywords).
    EntityExtractor   — class with extract_from_chunk(chunk) -> ExtractionResult.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from autogen.extraction.prompts import (
    COMPLETION_DELIMITER,
    ENTITY_CONTINUE_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_WITH_TYPES_SECTION,
    EXTRACT_MISSING_ENTITIES_PROMPT,
    MEDICAL_ENTITY_TYPES,
    RESPONSE_DELIMITER,
    TUPLE_DELIMITER,
)
from autogen.extraction.type_resolver import EntityTypeResolver
from autogen.models.storage import EntityNode, EntityRelation, TextChunk

if TYPE_CHECKING:
    from autogen.protocols.llm import LlmClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from the .NET source
# ---------------------------------------------------------------------------
DEFAULT_MAX_GLEANING = 1  # EntityExtractMaxGleaning — LightRagConfig.cs


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """The structured output of extract_from_chunk().

    nodes                — every EntityNode discovered (across all passes).
    relations            — every EntityRelation discovered.
    content_keywords     — keyword list from the primary extraction pass.
    """

    nodes: list[EntityNode] = field(default_factory=list)
    relations: list[EntityRelation] = field(default_factory=list)
    content_keywords: list[str] = field(default_factory=list)


#: Alias for ExtractionResult — used by tests and external callers.
ExtractionResponse = ExtractionResult


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------
class EntityExtractor:
    """LLM-driven entity and relationship extractor for text chunks.

    Stateful: holds a reference to the LLM client and model configuration,
    plus an EntityTypeResolver shared across all chunk extractions.

    Typical use::

        extractor = EntityExtractor(
            llm=llm,
            model="gpt-4o",
            max_gleaning=1,
            type_resolver=EntityTypeResolver(llm),
        )
        result = await extractor.extract_from_chunk(chunk)
    """

    # ------------------------------------------------------------------
    # Static helpers — entity / relation ID generators
    # ------------------------------------------------------------------
    @staticmethod
    def _make_entity_id(name: str) -> str:
        """Canonical entity ID: ent-(lowercased, underscores→spaces)."""
        cleaned = name.lower().strip().replace("_", " ").replace("-", " ")
        # Collapse multiple spaces
        cleaned = " ".join(cleaned.split())
        return f"ent-({cleaned})"

    @staticmethod
    def _make_relation_id(src_name: str, tgt_name: str) -> str:
        """Canonical relation ID via EntityRelation.id_from_names (order-invariant)."""
        return EntityRelation.id_from_names(src_name, tgt_name)

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        llm: LlmClient,
        *,
        model: str = "gpt-4o",
        max_gleaning: int = DEFAULT_MAX_GLEANING,
        type_resolver: EntityTypeResolver | None = None,
        concurrency: int = 4,
        entity_types: list[str] | None = None,
    ) -> None:
        """
        Args:
            llm: LLM client (LiteLlm, etc.).
            model: Model name for extraction calls.
            max_gleaning: How many gleaning passes to run (0 = none).
            type_resolver: Pre-built EntityTypeResolver. Created fresh if None.
            concurrency: Max concurrent chunk extractions (used by pipeline).
            entity_types: Canonical entity types to pass to the LLM.
                          Defaults to MEDICAL_ENTITY_TYPES.
        """
        self._llm = llm
        self._model = model
        self._max_gleaning = max_gleaning
        self._type_resolver = type_resolver or EntityTypeResolver(llm)
        self._concurrency = concurrency
        self._entity_types = entity_types or list(MEDICAL_ENTITY_TYPES)
        self._semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def extract_from_chunk(self, chunk: TextChunk) -> ExtractionResult:
        """Run the full extraction pipeline on one TextChunk.

        Returns ExtractionResult — never raises.
        """
        logger.info("Extracting from chunk %s (app_id=%s)", chunk.id, chunk.app_id)

        # 1. Primary extraction pass
        raw_text = await self._call_llm_for_extraction(chunk.content)
        nodes, relations, keywords = self._parse_primary(raw_text, chunk)

        # 2. Gleaning loop (Day 13)
        for glean_idx in range(self._max_gleaning):
            prev_names = [n.entity_name for n in nodes]
            glean_text = await self._call_llm_for_gleaning(
                chunk.content, prev_names
            )
            if not glean_text or glean_text.strip() == COMPLETION_DELIMITER:
                break
            glean_nodes, glean_relations, _ = self._parse_primary(
                glean_text, chunk
            )
            nodes.extend(glean_nodes)
            relations.extend(glean_relations)
            logger.debug(
                "Gleaning pass %d: +%d entities, +%d relations",
                glean_idx + 1,
                len(glean_nodes),
                len(glean_relations),
            )

        # 3. Missing-entity recovery (Day 13)
        missing = self._find_missing_entities(nodes, relations)
        if missing:
            logger.info("Recovering %d missing entities for chunk %s", len(missing), chunk.id)
            recovery_text = await self._call_llm_for_missing(
                chunk.content, missing
            )
            if recovery_text and recovery_text.strip() != COMPLETION_DELIMITER:
                rec_nodes, rec_relations, _ = self._parse_primary(
                    recovery_text, chunk
                )
                nodes.extend(rec_nodes)
                relations.extend(rec_relations)

        return ExtractionResult(
            nodes=nodes,
            relations=relations,
            content_keywords=keywords,
        )

    async def process_chunk(self, chunk: TextChunk) -> ExtractionResult:
        """Concurrency-safe wrapper around extract_from_chunk."""
        async with self._semaphore:
            return await self.extract_from_chunk(chunk)

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------
    async def _call_llm_for_extraction(self, content: str) -> str:
        """Primary extraction LLM call."""
        entity_types_section = self._render_entity_types_section()
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types_section=entity_types_section,
            response_delimiter=RESPONSE_DELIMITER,
            tuple_delimiter=TUPLE_DELIMITER,
            completion_delimiter=COMPLETION_DELIMITER,
            input_text=content,
        )
        return await self._llm.complete(
            [{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            model=self._model,
        )

    async def _call_llm_for_gleaning(
        self, content: str, previous_entities: list[str]
    ) -> str:
        """Gleaning pass — ask the LLM what it missed."""
        if not previous_entities:
            return COMPLETION_DELIMITER

        previous_str = ", ".join(previous_entities)
        entity_types_section = self._render_entity_types_section()
        prompt = ENTITY_CONTINUE_EXTRACTION_PROMPT.format(
            previous_entities=previous_str,
            entity_types_section=entity_types_section,
            response_delimiter=RESPONSE_DELIMITER,
            tuple_delimiter=TUPLE_DELIMITER,
            completion_delimiter=COMPLETION_DELIMITER,
            input_text=content,
        )
        return await self._llm.complete(
            [{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            model=self._model,
        )

    async def _call_llm_for_missing(
        self, content: str, missing_names: list[str]
    ) -> str:
        """Targeted recovery for entities the LLM referenced but didn't extract."""
        missing_str = ", ".join(missing_names)
        prompt = EXTRACT_MISSING_ENTITIES_PROMPT.format(
            missing_names=missing_str,
            response_delimiter=RESPONSE_DELIMITER,
            tuple_delimiter=TUPLE_DELIMITER,
            completion_delimiter=COMPLETION_DELIMITER,
            input_text=content,
        )
        return await self._llm.complete(
            [{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            model=self._model,
        )

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    def _render_entity_types_section(self) -> str:
        """Render the entity-types section for prompt templates."""
        if not self._entity_types:
            return ""
        types_str = ", ".join(self._entity_types)
        return ENTITY_EXTRACTION_WITH_TYPES_SECTION.format(entity_types=types_str)

    # ------------------------------------------------------------------
    # Parsing (mirrors EntityExtractor.cs parsing logic)
    # ------------------------------------------------------------------
    def _parse_primary(
        self, raw_text: str, chunk: TextChunk
    ) -> tuple[list[EntityNode], list[EntityRelation], list[str]]:
        """Parse the LLM's delimiter-based output into structured data.

        Format expected (mirrors .NET source):
            entity1##type##desc<|>entity2##type##desc<|>
            source##target##desc##kw1,kw2##0.8<|>
            kw1,kw2,kw3<|COMPLETE|>
        """
        entities: list[EntityNode] = []
        relations: list[EntityRelation] = []
        keywords: list[str] = []

        # Strip completion marker
        text = raw_text.replace(COMPLETION_DELIMITER, "").strip()
        if not text:
            return entities, relations, keywords

        sections = text.split(RESPONSE_DELIMITER)
        if len(sections) == 0:
            return entities, relations, keywords

        # ---- Entities section (first block) ----
        entity_lines = [s.strip() for s in sections[0].split("\n") if s.strip()]
        for line in entity_lines:
            parts = [p.strip() for p in line.split(TUPLE_DELIMITER)]
            if len(parts) >= 3:
                name, raw_type, desc = parts[0], parts[1], parts[2]
                if name:
                    entity = EntityNode(
                        id=self._make_entity_id(name),
                        entity_name=name.strip(),
                        entity_type=raw_type.strip().upper(),
                        description=desc.strip(),
                        descriptions=[desc.strip()],
                        historical_entity_types=[raw_type.strip()],
                        source_ids=[chunk.id],
                        app_id=chunk.app_id,
                    )
                    entities.append(entity)

        # ---- Content keywords (last section if present) ----
        if len(sections) >= 2:
            # The middle sections are relations, the last may be keywords
            keyword_section = sections[-1].strip()
            # Heuristic: if the section looks like comma-separated single words/short phrases
            # and not containing TUPLE_DELIMITER, treat as keywords
            if TUPLE_DELIMITER not in keyword_section and "," in keyword_section:
                keywords = [k.strip() for k in keyword_section.split(",") if k.strip()]
            else:
                keywords = []

        # ---- Relations section(s) — between entities and keywords ----
        relation_sections = sections[1:-1] if keywords else sections[1:]
        for rel_section in relation_sections:
            rel_lines = [
                s.strip() for s in rel_section.split("\n") if s.strip()
            ]
            for line in rel_lines:
                parts = [p.strip() for p in line.split(TUPLE_DELIMITER)]
                if len(parts) >= 5:
                    src, tgt, desc, kw_str, strength_str = parts[:5]
                    if src and tgt:
                        try:
                            strength = float(strength_str)
                        except (ValueError, TypeError):
                            strength = 0.5
                        rel_keywords = (
                            [k.strip() for k in kw_str.split(",") if k.strip()]
                            if kw_str
                            else []
                        )
                        relation = EntityRelation(
                            id=self._make_relation_id(src, tgt),
                            source_name=src.strip(),
                            target_name=tgt.strip(),
                            source_id=self._make_entity_id(src),
                            target_id=self._make_entity_id(tgt),
                            description=desc.strip() if desc else "",
                            descriptions=[desc.strip()] if desc else [],
                            keywords=rel_keywords,
                            strength=strength,
                            source_ids=[chunk.id],
                            app_id=chunk.app_id,
                        )
                        relations.append(relation)

        return entities, relations, keywords

    # ------------------------------------------------------------------
    # Missing-entity detection
    # ------------------------------------------------------------------
    @staticmethod
    def _find_missing_entities(
        nodes: list[EntityNode], relations: list[EntityRelation]
    ) -> list[str]:
        """Return entity names referenced in relations but missing from nodes."""
        node_names = {n.entity_name.lower().strip() for n in nodes}
        missing: set[str] = set()
        for rel in relations:
            for name in (rel.source_name, rel.target_name):
                if name.lower().strip() not in node_names:
                    missing.add(name)
        return sorted(missing)