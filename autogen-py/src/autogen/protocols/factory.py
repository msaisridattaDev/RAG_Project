"""Two-level factory delegate hierarchy — mirrors autogen.net's critical pattern.

This is the single most important architectural pattern ported from .NET:

    QnALlmAgentFactoryFactory(examId) → QnALlmAgentFactory(ctx) → Task<QnALlmAgent>

Rendered in Python as:

    QnAAgentFactoryFactory.for_exam(exam_id) → QnAAgentFactory.create(context) → QnAAgent

The outer level scopes by exam dataset (multi-tenancy).
The inner level scopes by conversation (request context).

Without this, multi-tenancy doesn't work and the Python port silently
diverges from the source.
"""

from __future__ import annotations

from typing import Protocol

from autogen.models.agent import AgentContext, QnAAgent


class QnAAgentFactory(Protocol):
    """Inner factory — creates an agent for a specific conversation context.

    Mirrors autogen.net's QnALlmAgentFactory(ctx) → Task<QnALlmAgent>.

    One instance per conversation. Created by QnAAgentFactoryFactory.for_exam().
    """

    async def create(self, context: AgentContext) -> QnAAgent:
        """Create a QnA agent for the given conversation context.

        Args:
            context: The conversation context (session_id, user_id, app_id, exam_id).

        Returns:
            A fully initialized QnAAgent ready to process queries.
        """
        ...


class QnAAgentFactoryFactory(Protocol):
    """Outer factory — scoped by exam dataset.

    Mirrors autogen.net's QnALlmAgentFactoryFactory(examId) → QnALlmAgentFactory.

    One instance per exam dataset. Creates inner factories that are
    scoped by conversation.
    """

    @staticmethod
    def for_exam(exam_id: str) -> QnAAgentFactory:
        """Create an agent factory scoped to a specific exam dataset.

        Args:
            exam_id: Identifies the exam dataset (e.g., "neetpg-2025", "neetug-2025").

        Returns:
            A QnAAgentFactory that creates agents for this exam.
        """
        ...
