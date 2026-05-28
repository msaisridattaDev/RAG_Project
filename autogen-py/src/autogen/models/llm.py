"""LLM wire-format models — mirrors autogen.net LlmMessage / LlmUsage / LlmChunk.

These shapes flow through every LLM call in the system:
    - Phase 4 Day 18 QnAAgent.answer(): consumes ``LlmChunk`` from the LlmClient
      streaming protocol and aggregates ``LlmUsage`` per request.
    - Phase 5 Day 22 MCP audit tool: replays the recorded ``LlmMessage`` history.

Field naming snake_cases the .NET PascalCase but preserves the concepts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class LlmMessage(BaseModel):
    """A single chat message — mirrors autogen.net LlmMessage."""

    role: Role = Field(description="OpenAI-compatible role")
    content: str = Field(description="Message content")
    name: str | None = Field(
        default=None,
        description="Optional message author label (e.g., tool name)",
    )


class LlmUsage(BaseModel):
    """Per-call cost / token usage — mirrors autogen.net LlmUsage.

    Costs are computed by the UsageCollector from the loaded models.json
    pricing rows; this struct just carries the raw counts plus the derived
    total cost.
    """

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0, description="In USD")
    is_cached: bool = Field(default=False, description="Served from ResponseCache")
    model: str | None = Field(default=None, description="Model id that produced this row")


class LlmChunk(BaseModel):
    """One streaming chunk — mirrors autogen.net LlmChunk.

    Yielded by LlmClient.stream(). The final chunk carries ``finish_reason``
    and the aggregate ``usage`` row; earlier chunks carry only ``delta``.
    """

    delta: str = Field(default="", description="Incremental text since the last chunk")
    finish_reason: str | None = Field(
        default=None,
        description="OpenAI-compatible finish reason on the terminal chunk",
    )
    usage: LlmUsage | None = Field(
        default=None,
        description="Aggregate usage row on the terminal chunk (None earlier)",
    )
    is_cached: bool = Field(
        default=False,
        description="Whole stream served from the ResponseCache",
    )
