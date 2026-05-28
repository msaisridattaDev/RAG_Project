"""Reference model — the output of ReferenceFinder.find().

A Reference represents a single retrieved passage with its relevance score
and metadata. Returned by the /v1/{app_id}/search endpoint and consumed
by QnAAgent for answer generation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Reference(BaseModel):
    """A retrieved reference passage.

    Attributes:
        id: The document ID in the vector store.
        content: The passage text content.
        score: The relevance score (from reranker, or vector score as fallback).
        metadata: Arbitrary metadata (app_id, source_id, page_number, etc.).
    """

    id: str = Field(description="Document ID in the vector store")
    content: str = Field(description="Passage text content")
    score: float = Field(default=0.0, description="Relevance score (higher is better)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
