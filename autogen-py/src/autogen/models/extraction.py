"""Pydantic models for LLM-structured entity/relation extraction output.

These models define the JSON schema passed to the LLM via
``response_format={"type": "json_schema", "strict": True}``.

The LLM must return a JSON object that validates against ``LlmExtractionOutput``.
The extractor then converts these raw extracted records into the richer
``EntityNode`` / ``EntityRelation`` storage models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    """One entity as the LLM extracts it — raw, before type resolution."""

    name: str = Field(description="Entity name as extracted from text")
    type: str = Field(description="Raw entity type assigned by the LLM")
    description: str = Field(description="Short description of the entity")


class ExtractedRelation(BaseModel):
    """One directed relationship between two entities."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    description: str = Field(default="", description="Description of the relationship")
    keywords: list[str] = Field(default_factory=list, description="Keywords for the relationship")
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Relationship strength 0–1")


class LlmExtractionOutput(BaseModel):
    """Full JSON output the LLM returns for one extraction call.

    Used as the ``json_schema`` in ``response_format`` so the LLM is
    constrained to emit exactly this shape.
    """

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    content_keywords: list[str] = Field(
        default_factory=list,
        description="High-level keywords summarising this chunk's content",
    )
