"""Storage models — Pydantic models for vector-indexed content types.

Mirrors the eight collection types used across the system:
    - TextChunk, EntityNode, EntityRelation (Phase 3 Graph RAG)
    - BookSegment, PdfSegment, WebSegment, ImageSegment, QuestionSegment (multi-modal)

Each model carries an ``embedding`` field (populated by VectorIndexer before upsert)
and an ``app_id`` for belt-and-suspenders tenancy.

Field naming preserves the .NET source's vocabulary, snake_cased:
    EntityName        → entity_name
    SourceIds         → source_ids
    HistoricalEntityTypes → historical_entity_types
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from autogen.models.base import AppId


class HasEmbedding(BaseModel):
    """Mixin for models that carry a vector embedding.

    Embedding lifecycle (three states):
        1. At extraction time — None / [] (not embedded yet).
        2. At storage time   — populated by VectorIndexer before upsert.
        3. At retrieval time — None / [] (Elasticsearch strips it via
                                _source_excludes for payload size).
    """

    embedding: list[float] | None = Field(
        default=None,
        description="1024-dimensional embedding vector. Must be populated before upsert.",
    )


class HasAppIdField(BaseModel):
    """Mixin for models that carry an app_id field for belt-and-suspenders tenancy."""

    app_id: AppId = Field(
        default_factory=lambda: AppId("neetpg"),
        description="Tenant scope — redundant with index name but enables cross-index queries.",
    )


# ---------------------------------------------------------------------------
# Raw input document — pre-chunking
# ---------------------------------------------------------------------------


class FullDoc(HasAppIdField):
    """Raw input document — the pre-chunking unit.

    Mirrors autogen.net FullDoc. Carries the original source content
    before it is split by the chunker (Phase 3 Day 11).
    """

    id: str = Field(description="Document identifier")
    content: str = Field(description="Raw document content (pre-chunking)")
    source: str = Field(default="", description="Provenance string (filename, URL, etc.)")


# ---------------------------------------------------------------------------
# Graph RAG models (Phase 3)
# ---------------------------------------------------------------------------


class TextChunk(HasEmbedding, HasAppIdField):
    """A chunk of text from a source document — the atomic retrieval unit.

    Index name pattern: ``textchunk_{app_id}_{dim}``.

    Field shape matches the plan and mirrors autogen.net's chunk row:
        full_doc_id   ← FK to FullDoc.id
        order         ← position within the source doc (0-based)
        tokens_count  ← cached token count for budget arithmetic
        keywords      ← optional keyword index (e.g., low-level entity hooks)
    """

    id: str = Field(description="Unique chunk identifier (e.g., 'chunk-3f9a...')")
    content: str = Field(description="The raw text content of this chunk")
    full_doc_id: str = Field(default="", description="ID of the source document")
    order: int = Field(default=0, description="0-based position of this chunk within full_doc")
    tokens_count: int = Field(default=0, description="Cached token count of the content")
    keywords: list[str] = Field(default_factory=list, description="Optional keyword tags")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityNode(HasEmbedding, HasAppIdField):
    """A knowledge graph entity node.

    Index name pattern: ``entitynode_{app_id}_{dim}``.

    Mirrors autogen.net EntityExtractor.cs:535 — when an entity is
    extracted from a chunk, its raw type goes into ``historical_entity_types``
    even if ``entity_type`` resolves to something different. This keeps the
    audit trail intact across multiple extraction passes.
    """

    id: str = Field(description="Entity identifier (e.g., 'ent-(aspirin)')")
    entity_name: str = Field(description="Canonical entity name")
    entity_type: str = Field(default="", description="Canonical type (after EntityTypeResolver)")
    description: str = Field(default="", description="Summary description")
    descriptions: list[str] = Field(
        default_factory=list,
        description="All per-source-pass descriptions (pre-summarization)",
    )
    historical_entity_types: list[str] = Field(
        default_factory=list,
        description="Every raw type this entity was extracted as, across passes",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of every chunk this entity was extracted from",
    )
    rank: int = Field(default=0, description="Entity rank (degree-based, set in Phase 3)")
    segment_content: str = Field(
        default="",
        description="Optional concatenated context segment used by Phase 3 Day 16",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityRelation(HasEmbedding, HasAppIdField):
    """A knowledge graph relationship between two entities (undirected).

    Index name pattern: ``entityrelation_{app_id}_{dim}``.

    The ID is derived from sorted(source_name, target_name) so the same
    undirected edge produces the same ID regardless of extraction order.
    Use ``EntityRelation.id_from_names("A", "B")`` to construct it.
    """

    id: str = Field(description="Relation identifier from id_from_names()")
    source_id: str = Field(default="", description="ID of the source EntityNode")
    target_id: str = Field(default="", description="ID of the target EntityNode")
    source_name: str = Field(description="Source entity name")
    target_name: str = Field(description="Target entity name")
    description: str = Field(default="", description="Summary description of the relation")
    descriptions: list[str] = Field(
        default_factory=list,
        description="All per-source-pass descriptions (pre-summarization)",
    )
    keywords: list[str] = Field(default_factory=list, description="Edge keywords")
    strength: float = Field(default=1.0, description="Edge weight (1.0 = strong, 0.0 = weak)")
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of every chunk this relation was extracted from",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def id_from_names(src: str, tgt: str) -> str:
        """Compute the canonical (order-invariant) relation ID for two entity names.

        ``id_from_names("Aspirin", "COX-1") == id_from_names("COX-1", "Aspirin")``.

        This prevents accidental duplicate edges going both ways and matches
        the .NET source's edge-id convention.
        """
        a, b = sorted([src.lower().strip(), tgt.lower().strip()])
        return f"rel-({a})-({b})"


# ---------------------------------------------------------------------------
# Multi-modal segment models (Phase 1 Day 4 / Phase 3)
# ---------------------------------------------------------------------------


class BookSegment(HasEmbedding, HasAppIdField):
    """A segment from a book.

    Index name pattern: ``booksegment_{app_id}_{dim}``.
    """

    id: str = Field(description="Segment identifier")
    content: str = Field(description="Segment text content")
    title: str = Field(default="", description="Book title")
    chapter: str = Field(default="", description="Chapter name/number")
    page_number: int = Field(default=0, description="Page number")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PdfSegment(HasEmbedding, HasAppIdField):
    """A segment extracted from a PDF document.

    Index name pattern: ``pdfsegment_{app_id}_{dim}``.
    """

    id: str = Field(description="Segment identifier")
    content: str = Field(description="Extracted text content")
    filename: str = Field(default="", description="PDF filename")
    page_number: int = Field(default=0, description="Page number")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSegment(HasEmbedding, HasAppIdField):
    """A segment from a web page.

    Index name pattern: ``websegment_{app_id}_{dim}``.
    """

    id: str = Field(description="Segment identifier")
    content: str = Field(description="Web page text content")
    url: str = Field(default="", description="Source URL")
    title: str = Field(default="", description="Page title")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageSegment(HasEmbedding, HasAppIdField):
    """An image with caption and OCR text.

    Index name pattern: ``imagesegment_{app_id}_{dim}``.
    """

    id: str = Field(description="Segment identifier")
    content: str = Field(description="Combined caption and OCR text")
    caption: str = Field(default="", description="Image caption")
    ocr_text: str = Field(default="", description="OCR-extracted text")
    image_url: str = Field(default="", description="Image URL or path")
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuestionSegment(HasEmbedding, HasAppIdField):
    """An exam question with answer key.

    Index name pattern: ``questionsegment_{app_id}_{dim}``.
    """

    id: str = Field(description="Segment identifier")
    content: str = Field(description="Question text content")
    question_id: str = Field(default="", description="Stable question identifier")
    question_text: str = Field(default="", description="The question itself")
    options: list[str] = Field(default_factory=list, description="Answer options")
    correct_answer: str = Field(default="", description="Correct answer key")
    explanation: str = Field(default="", description="Answer explanation")
    metadata: dict[str, Any] = Field(default_factory=dict)
