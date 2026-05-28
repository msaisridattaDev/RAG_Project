"""QnA streaming chunk models — Phase 4 Day 18.

QnAChunk is the structured event type emitted by QnAAgent.answer() and
QnAAgent.counter_answer().  Each chunk carries a ``kind`` tag so that the
client UI can render each event differently:

    thought    → typing-indicator status ("Loading references…")
    reference  → citation badges (sources the agent retrieved)
    answer     → a single streaming token or delta of the answer
    done       → end-of-stream marker with usage + conversation_id
    error      → mid-stream failure; stream ends after this chunk

CounterRequest is the request shape for follow-up questions (Day 19).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QnAChunkKind(StrEnum):
    """Discriminator tag for QnAChunk — drives UI rendering decisions."""

    THOUGHT = "thought"
    REFERENCE = "reference"
    ANSWER = "answer"
    DONE = "done"
    ERROR = "error"


class QnAChunk(BaseModel):
    """One structured event from the QnA agent streaming pipeline.

    Emitted by QnAAgent.answer() and QnAAgent.counter_answer().
    Transported as SSE events by Phase 5 REST/WebSocket layers.
    """

    kind: QnAChunkKind
    text: str = Field(default="", description="Human-readable text payload")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific data (refs list, usage dict, conversation_id, etc.)",
    )


class CounterRequest(BaseModel):
    """Request shape for a follow-up question (Day 19 counter_answer flow).

    The agent's bound app_id (from its AgentContext) determines retrieval
    scope — the caller cannot override it.
    """

    conversation_id: str = Field(description="Existing conversation to extend")
    follow_up: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The follow-up question or continuation",
    )
