"""LLM client protocols — mirrors autogen.net ILlmClient.

Two operations every Phase 4+ caller depends on:

    stream(messages, model, **kw) -> AsyncIterator[LlmChunk]
        Streaming generation — yields LlmChunk objects. The final chunk
        carries ``finish_reason`` and aggregate ``usage``.

    complete(messages, model, **kw) -> str
        One-shot generation — returns the full assembled text.

Plus two cross-cutting protocols Phase 4/5 wire around it:

    ResponseCache    — cached streaming output (so re-asking the same question
                       returns the cached chunks)
    UsageCollector   — aggregates LlmUsage rows for cost tracking
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from autogen.models.llm import LlmChunk, LlmMessage, LlmUsage


@runtime_checkable
class LlmClient(Protocol):
    """LLM gateway — mirrors autogen.net ILlmClient.

    Implementations: OpenAI / Anthropic / OpenRouter / Groq / DeepSeek /
    LiteLLM-multiplexer / self-hosted llama.cpp / Claude / Codex.
    """

    def stream(
        self,
        messages: list[LlmMessage],
        model: str,
        **kwargs: object,
    ) -> AsyncIterator[LlmChunk]:
        """Stream a chat completion. The final chunk carries usage.

        Note: this returns an async iterator, not an awaitable. Callers do::

            async for chunk in llm.stream(msgs, model="gpt-4o"):
                ...
        """
        ...

    async def complete(
        self,
        messages: list[LlmMessage],
        model: str,
        **kwargs: object,
    ) -> str:
        """One-shot chat completion. Returns the assembled assistant text."""
        ...


@runtime_checkable
class ResponseCache(Protocol):
    """Streaming-response cache — mirrors autogen.net IResponseCache.

    Keyed by the call's prompt + model + parameter fingerprint.
    On hit, the cached ``list[LlmChunk]`` is re-yielded as if the LLM had
    just produced it (with ``is_cached=True`` on every chunk).
    """

    async def get(self, key: str) -> list[LlmChunk] | None:
        """Return cached chunks for the key, or None on miss."""
        ...

    async def set(self, key: str, chunks: list[LlmChunk]) -> None:
        """Store the chunks under the key."""
        ...


@runtime_checkable
class UsageCollector(Protocol):
    """Cost accountant — mirrors autogen.net IUsageCollector.

    Aggregates ``LlmUsage`` rows by tag prefix so callers can ask for
    "total cost for conversation X" or "total cost for app Y".
    """

    def record(self, key: str, usage: LlmUsage) -> None:
        """Append a usage row under ``key``."""
        ...

    def total_cost(self, prefix: str = "") -> float:
        """Return the total USD cost for every recorded key whose name
        starts with ``prefix``. An empty prefix returns the grand total.
        """
        ...
