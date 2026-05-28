"""Usage collector + tracking decorator — mirrors autogen.net UsageCollector.

Three buckets per session key:
  _total   — every recorded call (cached + real)
  _cached  — only calls with usage.is_cached=True
  _real    — only calls with usage.is_cached=False

The ratio _real / _total is the cache effectiveness metric.

Session key is carried by a ContextVar (not passed through function signatures)
so deeply nested code can attribute costs without parameter threading.
The session key encodes (app_id, conversation_id, user_id)::

    "neetpg:conv-abc:user-42"

The UsageTrackingLlmClient is the OUTERMOST decorator in the stack. It sits
above CachingLlmClient so it sees every chunk — cached or real — and routes
the cost correctly.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any

from structlog import get_logger

from autogen.models.llm import LlmChunk, LlmMessage, LlmUsage
from autogen.protocols.llm import LlmClient

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ContextVar — async-safe request-scoped session identifier
# ---------------------------------------------------------------------------

SESSION_KEY_VAR: ContextVar[str] = ContextVar("session_key", default="anon")


def set_session_key(key: str) -> None:
    """Set the session key for the current async task.

    Called by the request handler (Phase 5 Day 20) before any LLM calls::

        SESSION_KEY_VAR.set(f"{app_id}:{conversation_id}:{user_id}")
    """
    SESSION_KEY_VAR.set(key)


# ---------------------------------------------------------------------------
# UsageBucket — tracks token + cost accumulation
# ---------------------------------------------------------------------------


class UsageBucket:
    """Accumulator for a single session's usage totals."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.call_count: int = 0

    def merge(self, usage: LlmUsage) -> None:
        """Add one usage row into this bucket."""
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.total_cost
        self.call_count += 1

    def to_usage(self, *, model: str | None = None) -> LlmUsage:
        """Export as an LlmUsage row (for API responses)."""
        return LlmUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            total_cost=self.total_cost,
            model=model,
        )

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
        }


# ---------------------------------------------------------------------------
# InMemoryUsageCollector
# ---------------------------------------------------------------------------


class InMemoryUsageCollector:
    """Three-bucket ledger: _total, _cached, _real.

    Thread/async-safe via asyncio.Lock. Keyed by session string.

    Usage::

        collector = InMemoryUsageCollector()
        collector.record("neetpg:conv-abc:user-42", usage_row)
        total = collector.total_cost("neetpg:")  # per-app aggregation
    """

    def __init__(self) -> None:
        self._total: dict[str, UsageBucket] = {}
        self._cached: dict[str, UsageBucket] = {}
        self._real: dict[str, UsageBucket] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    async def record(self, key: str, usage: LlmUsage) -> None:
        """Append one usage row to the correct buckets."""
        async with self._lock:
            self._ensure_bucket(key)
            self._total[key].merge(usage)
            if usage.is_cached:
                self._cached[key].merge(usage)
            else:
                self._real[key].merge(usage)

    def record_sync(self, key: str, usage: LlmUsage) -> None:
        """Synchronous record (for contexts where async lock isn't available).

        Uses a re-entrant scheme: if we're already inside the event loop
        with a running lock, this is unsafe. Prefer async record().
        """
        self._ensure_bucket(key)
        self._total[key].merge(usage)
        if usage.is_cached:
            self._cached[key].merge(usage)
        else:
            self._real[key].merge(usage)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def total_cost(self, prefix: str = "") -> float:
        """Sum total_cost for all session keys starting with *prefix*.

        An empty prefix returns the grand total across all sessions.
        """
        total = 0.0
        for key, bucket in self._total.items():
            if key.startswith(prefix):
                total += bucket.total_cost
        return total

    def snapshot(self, key: str) -> dict[str, object]:
        """Return the three-bucket snapshot for one session key."""
        return {
            "total": self._total.get(key, UsageBucket()).snapshot(),
            "real": self._real.get(key, UsageBucket()).snapshot(),
            "cached": self._cached.get(key, UsageBucket()).snapshot(),
        }

    def all_snapshots(self) -> dict[str, dict[str, object]]:
        """Return snapshots for all keys (useful for admin dashboards)."""
        return {k: self.snapshot(k) for k in self._total}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_bucket(self, key: str) -> None:
        """Ensure three dict entries exist for *key*."""
        for d in (self._total, self._cached, self._real):
            if key not in d:
                d[key] = UsageBucket()


# ---------------------------------------------------------------------------
# UsageTrackingLlmClient — outermost decorator
# ---------------------------------------------------------------------------


class UsageTrackingLlmClient:
    """Outermost decorator: records usage after every stream call.

    Wraps an inner LlmClient (typically CachingLlmClient wrapped around
    LiteLlmClient). Reads the session key from SESSION_KEY_VAR and records
    the final LlmUsage to the InMemoryUsageCollector.

    Being outermost means it sees ALL chunks — including cached ones that
    would be transparent to an inner decorator. This is correct: we want
    to count cached calls in _total and _cached.
    """

    def __init__(
        self,
        inner: LlmClient,
        collector: InMemoryUsageCollector,
    ) -> None:
        self._inner = inner
        self._collector = collector

    # ------------------------------------------------------------------
    # LlmClient protocol
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[LlmMessage],
        model: str,
        **kwargs: object,
    ) -> Any:  # AsyncIterator[LlmChunk]
        return _TrackingIterator(
            tracker=self,
            messages=messages,
            model=model,
            kwargs=kwargs,
        )

    async def complete(
        self,
        messages: list[LlmMessage],
        model: str,
        **kwargs: object,
    ) -> str:
        parts: list[str] = []
        async for chunk in self.stream(messages, model, **kwargs):
            if chunk.delta:
                parts.append(chunk.delta)
        return "".join(parts)


class _TrackingIterator:
    """Async iterator that collects the final chunk's usage and records it."""

    def __init__(
        self,
        tracker: UsageTrackingLlmClient,
        messages: list[LlmMessage],
        model: str,
        kwargs: dict[str, object],
    ) -> None:
        self._inner = tracker._inner
        self._collector = tracker._collector
        self._stream_iter = None
        self._messages = messages
        self._model = model
        self._kwargs = kwargs
        self._last_usage: LlmUsage | None = None
        self._recorded = False

    def __aiter__(self) -> "_TrackingIterator":
        self._stream_iter = self._inner.stream(
            self._messages, self._model, **self._kwargs
        )
        return self

    async def __anext__(self) -> LlmChunk:
        if self._stream_iter is None:
            raise StopAsyncIteration

        try:
            chunk = await self._stream_iter.__anext__()
        except StopAsyncIteration:
            # On stream end, record the usage
            if self._last_usage is not None and not self._recorded:
                await self._record(self._last_usage)
            raise

        # Track the final chunk's usage for recording on completion
        if chunk.usage is not None:
            self._last_usage = chunk.usage

        return chunk

    async def _record(self, usage: LlmUsage) -> None:
        """Record usage against the current session key."""
        key = SESSION_KEY_VAR.get()
        await self._collector.record(key, usage)
        self._recorded = True
        logger.debug(
            "usage.recorded",
            session_key=key,
            total_cost=usage.total_cost,
            total_tokens=usage.total_tokens,
            is_cached=usage.is_cached,
        )