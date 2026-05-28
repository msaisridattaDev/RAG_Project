"""LiteLLM-backed multi-provider LLM client — mirrors autogen.net's provider layer.

The innermost layer of the decorator stack. Wraps litellm.acompletion behind
the LlmClient protocol so every upstream caller (caching, tracking, QnAAgent)
sees the same interface regardless of which provider handles the request.

Select providers by prefixing the model name with "{provider}/":
    - "groq/llama-3.3-70b-versatile"
    - "openai/gpt-4o"
    - "anthropic/claude-3-5-sonnet-latest"

LiteLLM handles provider-specific API quirks, retries, and cost calculation.
"""

from __future__ import annotations

from typing import Any

import litellm
from structlog import get_logger

from autogen.models.llm import LlmChunk, LlmMessage, LlmUsage
from autogen.protocols.llm import LlmClient

logger = get_logger(__name__)


class LiteLlmClient:
    """Thin wrapper around litellm.acompletion — the innermost LlmClient decorator.

    Implements the LlmClient protocol: stream() for token-by-token generation,
    complete() for one-shot text assembly.
    """

    def __init__(self, default_temperature: float = 0.0) -> None:
        """Args:
            default_temperature: Used when the caller doesn't pass temperature.
                Default 0.0 because most system calls (extraction, classification)
                are deterministic-by-design — low temperature → reproducible output
                → better cache hit rates.
        """
        self._default_temperature = default_temperature

    # ------------------------------------------------------------------
    # Public API — LlmClient protocol
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[LlmMessage],
        model: str,
        **kwargs: object,
    ) -> Any:  # AsyncIterator[LlmChunk]
        """Return an async iterator that streams chat completion chunks.

        Callers use: ``async for chunk in llm.stream(msgs, model=...): ...``
        """
        return _StreamIterator(
            client=self,
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
        """One-shot completion — assembles the full text from stream().

        Used by code paths that don't need streaming (keyword extraction,
        name normalization, relevance checks).
        """
        parts: list[str] = []
        async for chunk in self.stream(messages, model, **kwargs):
            if chunk.delta:
                parts.append(chunk.delta)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Wire-format translation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_litellm(messages: list[LlmMessage]) -> list[dict[str, str]]:
        """Convert our Pydantic LlmMessage list to LiteLLM's expected dict shape.

        LiteLLM expects: [{"role": "user", "content": "..."}, ...]
        """
        out: list[dict[str, str]] = []
        for m in messages:
            entry: dict[str, str] = {"role": m.role, "content": m.content}
            if m.name:
                entry["name"] = m.name
            out.append(entry)
        return out

    @staticmethod
    def _build_kwargs(
        messages: list[LlmMessage],
        model: str,
        temperature: float | None,
        extra: dict[str, object],
    ) -> dict[str, object]:
        """Assemble the kwargs dict passed to litellm.acompletion."""
        kw: dict[str, object] = {
            "model": model,
            "messages": LiteLlmClient._to_litellm(messages),
            "stream": True,
        }
        kw.update(extra)
        # temperature from extra overrides the explicit parameter
        if "temperature" not in kw and temperature is not None:
            kw["temperature"] = temperature
        return kw

    @staticmethod
    def _extract_usage(
        raw_chunk: Any,
        model: str,
    ) -> LlmUsage | None:
        """Extract token counts + cost from the final streaming chunk.

        Uses litellm.completion_cost() for USD pricing (LiteLLM's internal
        price table, which we cross-check against models.json in Day 9).
        Falls back to tokens-only if cost calculation fails.
        """
        try:
            usage_obj = getattr(raw_chunk, "usage", None)
            if usage_obj is None:
                return None

            prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
            total_tokens = getattr(usage_obj, "total_tokens", 0) or (
                prompt_tokens + completion_tokens
            )

            total_cost: float = 0.0
            try:
                total_cost = litellm.completion_cost(
                    completion_response=raw_chunk,
                    model=model,
                )
                if not isinstance(total_cost, (int, float)) or total_cost < 0:
                    total_cost = 0.0
            except Exception as exc:
                logger.debug("cost_calculation_failed", model=model, error=str(exc))

            return LlmUsage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
                total_cost=float(total_cost),
                is_cached=False,  # LiteLLM never sets is_cached; the caching layer does
                model=model,
            )
        except Exception as exc:
            logger.debug("usage_extraction_failed", model=model, error=str(exc))
            return None


# ---------------------------------------------------------------------------
# Internal async-iterator helper — the shared generator for stream()
# ---------------------------------------------------------------------------


class _StreamIterator:
    """Async iterator returned by LiteLlmClient.stream().

    Splitting this into its own class keeps the generator logic testable
    and avoids cramming everything into one method.
    """

    def __init__(
        self,
        client: LiteLlmClient,
        messages: list[LlmMessage],
        model: str,
        kwargs: dict[str, object],
    ) -> None:
        self._client = client
        self._messages = messages
        self._model = model
        self._kwargs = kwargs

    def __aiter__(self) -> "_StreamIterator":
        return self

    async def _setup(self) -> None:
        """Lazily start the LiteLLM stream on the first iteration."""
        temperature = self._kwargs.pop("temperature", None)
        if temperature is None:
            temperature = self._client._default_temperature

        kw = self._client._build_kwargs(
            self._messages, self._model, temperature, self._kwargs
        )

        safe_kw = {k: v for k, v in kw.items() if "api_key" not in k.lower()}
        logger.debug("litellm.stream.start", model=self._model, kwargs=safe_kw)

        self._response = await litellm.acompletion(**kw)

    async def __anext__(self) -> LlmChunk:
        if not hasattr(self, "_response"):
            await self._setup()
        try:
            raw = await self._response.__anext__()
        except StopAsyncIteration:
            raise

        # Extract delta text from the chunk
        delta_text = _extract_delta_text(raw)

        # Check for finish reason
        finish_reason = _extract_finish_reason(raw)

        # On terminal chunk, extract usage
        usage = None
        if finish_reason is not None:
            usage = self._client._extract_usage(raw, self._model)
            logger.debug("litellm.stream.complete", finish_reason=finish_reason)

        return LlmChunk(
            delta=delta_text,
            finish_reason=finish_reason,
            usage=usage,
            is_cached=False,
        )


# ---------------------------------------------------------------------------
# LiteLLM chunk parsing helpers
# ---------------------------------------------------------------------------


def _extract_delta_text(raw_chunk: Any) -> str:
    """Safely extract delta content from a LiteLLM streaming chunk.

    LiteLLM normalizes provider-specific chunk shapes to an OpenAI-compatible
    structure: chunk.choices[0].delta.content.
    """
    try:
        choices = getattr(raw_chunk, "choices", None)
        if choices and len(choices) > 0:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                text = getattr(delta, "content", None)
                if text is not None:
                    return str(text)
    except Exception:
        pass
    return ""


def _extract_finish_reason(raw_chunk: Any) -> str | None:
    """Safely extract finish_reason from a LiteLLM streaming chunk."""
    try:
        choices = getattr(raw_chunk, "choices", None)
        if choices and len(choices) > 0:
            reason = getattr(choices[0], "finish_reason", None)
            if reason is not None:
                return str(reason) if reason else "stop"
    except Exception:
        pass
    return None