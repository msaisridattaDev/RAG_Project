"""Agent models — mirrors autogen.net QnAAgent and ConversationRuntimeContext.

Defines the conversation context and agent state that flows through
the two-level factory delegate hierarchy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from autogen.models.base import AppId, TenantId
from autogen.models.enums import AgentStatus, Tier


class AgentContext(BaseModel):
    """Conversation context — mirrors autogen.net ConversationRuntimeContext.

    Created by the inner factory (QnAAgentFactory) and passed to the agent at
    construction time. Carries all request-scoped data the agent needs to
    bind itself to a specific user + exam + conversation.

    Field naming matches the .NET source's vocabulary (``conversation_id``,
    ``user_id``, ``app_id``, ``tier``, ``metadata``). ``session_id`` is kept
    as a deprecated alias for back-compat; new code should use
    ``conversation_id``.
    """

    conversation_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Stable identifier for this conversation",
    )
    user_id: str = Field(description="Identifier of the user driving the conversation")
    app_id: AppId = Field(description="Tenant scope (exam dataset)")
    tenant_id: TenantId | None = Field(
        default=None,
        description="Optional sub-tenant within an app (institution, etc.)",
    )
    exam_id: str | None = Field(
        default=None,
        description="Optional specific exam ID (defaults to app_id at use-site)",
    )
    tier: Tier = Field(
        default=Tier.FREE,
        description="User tier — drives model routing and feature flags",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary per-conversation metadata (correlation_id, ab-test bucket, etc.)",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def session_id(self) -> str:
        """Deprecated alias for ``conversation_id``. Kept for back-compat."""
        return self.conversation_id


# Plan name → existing class. Re-exported so callers can import either symbol.
ConversationRuntimeContext = AgentContext


class QnAAgent(BaseModel):
    """QnA agent state — mirrors autogen.net QnALlmAgent.

    Represents an instantiated agent ready to process queries.
    The agent is the product of the two-level factory hierarchy::

        QnAAgentFactoryFactory.for_exam(exam_id)
            → QnAAgentFactory.create(ctx)
                → QnAAgent
    """

    agent_id: str = Field(default_factory=lambda: uuid4().hex)
    context: AgentContext
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_active_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
