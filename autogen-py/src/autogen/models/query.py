"""Query models — knobs and result envelopes for the retrieval pipeline.

Defines the input shape (QueryParam) every Phase 3 query path accepts and
the output shape (CombinedContext) every Phase 3 query path returns to the
QnA agent (Phase 4). Both mirror the .NET source's QueryParam record and
CombinedContext class field-for-field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from autogen.models.enums import QueryMode
from autogen.models.storage import EntityNode, EntityRelation, TextChunk


class QueryParam(BaseModel):
    """Retrieval knobs — mirrors autogen.net QueryParam.

    Every Phase 3 query path (Local, Global, Hybrid, Naive) reads these knobs.
    Defaults match the .NET source's defaults for the same fields.
    """

    mode: QueryMode = QueryMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=200, description="Final result count after rerank")
    local_top_k: int = Field(default=10, ge=1, le=200, description="Top-k for local (entity-keyword) sub-retrieval")
    global_top_k: int = Field(default=10, ge=1, le=200, description="Top-k for global (relation-keyword) sub-retrieval")
    keyword_top_k: int = Field(default=10, ge=1, le=200, description="Top-k for keyword (chunk) sub-retrieval")
    max_tokens_for_context: int = Field(default=4000, ge=100, le=32000)
    max_tokens_for_entity_context: int = Field(default=2000, ge=100, le=32000)
    max_tokens_for_relation_context: int = Field(default=2000, ge=100, le=32000)
    response_model: str | None = Field(
        default=None,
        description="Optional override model id; falls back to tier default if None",
    )
    only_need_context: bool = Field(
        default=False,
        description="If true, return the retrieved context without invoking the LLM",
    )


class CombinedContext(BaseModel):
    """Retrieved context bundle — mirrors autogen.net CombinedContext.

    The output of any Phase 3 query path. Carries the three retrieval slices
    (entities, relationships, sources) that the QnA agent stitches into
    the final prompt.
    """

    entities: list[EntityNode] = Field(default_factory=list)
    relationships: list[EntityRelation] = Field(default_factory=list)
    sources: list[TextChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def build_context_string(self) -> str:
        """Render the bundle as the CSV-section string that the QnA prompt expects.

        Mirrors autogen.net CombinedContext.BuildContextString which produces
        three labelled CSV sections separated by ``-----Entities-----``,
        ``-----Relationships-----``, ``-----Sources-----``.

        The output is intentionally machine-formatted: every section gets a
        header row even when the slice is empty, so prompt templates can
        always rely on the layout.
        """
        sections: list[str] = []

        sections.append("-----Entities-----")
        sections.append("id,entity_name,entity_type,description,rank")
        for e in self.entities:
            sections.append(
                ",".join(
                    [
                        _csv_field(e.id),
                        _csv_field(e.entity_name),
                        _csv_field(e.entity_type),
                        _csv_field(e.description),
                        str(e.rank),
                    ]
                )
            )

        sections.append("-----Relationships-----")
        sections.append("id,source_name,target_name,description,keywords,strength")
        for r in self.relationships:
            sections.append(
                ",".join(
                    [
                        _csv_field(r.id),
                        _csv_field(r.source_name),
                        _csv_field(r.target_name),
                        _csv_field(r.description),
                        _csv_field("|".join(r.keywords)),
                        f"{r.strength:.4f}",
                    ]
                )
            )

        sections.append("-----Sources-----")
        sections.append("id,full_doc_id,order,content")
        for c in self.sources:
            sections.append(
                ",".join(
                    [
                        _csv_field(c.id),
                        _csv_field(c.full_doc_id),
                        str(c.order),
                        _csv_field(c.content),
                    ]
                )
            )

        return "\n".join(sections)


def _csv_field(value: str) -> str:
    """Quote a single CSV field if it contains commas, quotes, or newlines.

    Standard RFC-4180 quoting: surround with double-quotes and double any
    inner double-quotes.
    """
    if value is None:
        return ""
    needs_quoting = any(ch in value for ch in (",", '"', "\n", "\r"))
    if needs_quoting:
        escaped = value.replace('"', '""')
        return f'"{escaped}"'
    return value
