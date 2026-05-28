"""WebSocket envelope protocol — Day 21.

Every message in both directions (client→server and server→client) is a JSON
object with this shape::

    {
        "type":           "question" | "pong" | "resume" | "chunk" | "ping" | "done" | "error",
        "correlation_id": "<uuid>",
        "payload":        { ... }   # type-specific content
    }

This lets us multiplex multiple concurrent Q&A exchanges over a single WebSocket
connection: each answer chunk carries the correlation_id of the question that
generated it, so the client can route chunks to the right UI element.

Client → server types:
    question  — new Q&A request; payload carries app_id, tier, question, conversation_id
    pong      — heartbeat reply to a server ping
    resume    — request to replay missed chunks from ring buffer

Server → client types:
    ping      — heartbeat probe (client must reply with pong within 60 s)
    chunk     — QnAChunk payload (thought | reference | answer | done | error)
    error     — protocol-level error (not a QnA error)
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    """Bidirectional WebSocket message envelope."""

    type: str = Field(..., description="Message type")
    correlation_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Links answer chunks back to their originating question",
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ping(cls) -> "Envelope":
        return cls(type="ping", payload={})

    @classmethod
    def error(cls, message: str, correlation_id: str = "") -> "Envelope":
        return cls(
            type="error",
            correlation_id=correlation_id or uuid.uuid4().hex,
            payload={"message": message},
        )

    @classmethod
    def chunk_from(cls, qna_chunk_dict: dict, correlation_id: str) -> "Envelope":
        return cls(type="chunk", correlation_id=correlation_id, payload=qna_chunk_dict)


class QuestionPayload(BaseModel):
    """Payload for type=='question' envelopes (client → server)."""

    question: str = Field(..., min_length=1, max_length=8000)
    app_id: str = Field(..., description="Exam dataset identifier")
    tier: str = Field(default="Free")
    user_id: str = Field(default="anon")
    conversation_id: str | None = Field(default=None)


class ResumePayload(BaseModel):
    """Payload for type=='resume' envelopes (client → server).

    The client sends the timestamp of the last chunk it received so the server
    can replay anything from the ring buffer that arrived after that point.
    """

    last_correlation_id: str = Field(default="")
    since_ts: float = Field(default=0.0, description="Unix timestamp of last received chunk")
