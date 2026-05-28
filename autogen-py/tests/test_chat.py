"""Day-10 smoke tests — /v1/{app_id}/chat and /v1/usage/{session_key}.

Exercises the full Phase 2 decorator stack with a mocked litellm.acompletion
so no real API keys are needed.

Verified behaviours:
  1. Single-model call streams tokens with is_cached=False on first hit.
  2. Identical call replays from cache with is_cached=True.
  3. /v1/usage/{session} returns the three-bucket snapshot:
       - total  = sum of both calls
       - real   = first call only
       - cached = second call only
  4. Premium + role=thinking fan-out returns parallel SSE events.
  5. The stack rejects requests that are missing X-LlmQuery-Token.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from autogen.api.app import create_app
from autogen.config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers — fake litellm streaming chunks
# ---------------------------------------------------------------------------

def _make_raw_chunk(content: str | None, finish: str | None = None, usage: Any = None):
    """Build a fake litellm streaming chunk object."""
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    choice.delta = delta
    choice.finish_reason = finish
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _make_usage_obj(prompt: int = 10, completion: int = 20):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    return u


async def _fake_acompletion_iter(tokens: list[str], cost: float = 0.00005):
    """Async generator yielding fake litellm chunks for a token sequence."""
    for tok in tokens:
        yield _make_raw_chunk(content=tok)
        await asyncio.sleep(0)  # yield control

    usage_obj = _make_usage_obj()
    final = _make_raw_chunk(content="", finish="stop", usage=usage_obj)
    # Patch litellm.completion_cost return value via the chunk itself
    yield final


class _FakeStreamResponse:
    """Mimics the async iterator returned by litellm.acompletion(stream=True)."""

    def __init__(self, tokens: list[str], cost: float = 0.00005):
        self._iter = _fake_acompletion_iter(tokens, cost)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._iter.__anext__()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_llm_stack():
    """Reset the module-level LLM stack singleton after each test.

    The chat route keeps a process-level singleton so the decorator stack is
    built only once in production. Between tests we must tear it down so each
    test gets a fresh stack scoped to its own tmp_path cache directory.
    """
    import autogen.api.routes.chat as _chat_module
    yield
    _chat_module._llm_client = None
    _chat_module._tier_router = None
    _chat_module._models_catalog = None


@pytest.fixture
def settings():
    s = Settings(env="test", _env_file=None)  # type: ignore[call-arg]
    s.llm_query_auth.allowed_token = "test-key"
    return s


@pytest.fixture
def app(settings, tmp_path):
    # Point cache at a temp dir so tests don't pollute each other
    settings.cache.base_path = str(tmp_path / "cache")
    return create_app(settings=settings)


@pytest.fixture
def auth_headers():
    return {"X-LlmQuery-Token": "test-key", "X-Session-Id": "smoke-1"}


@pytest.fixture
def chat_body():
    return {
        "tier": "Free",
        "role": "conversation",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "say hi in five words"}],
    }


# ---------------------------------------------------------------------------
# Helper — collect SSE events from a streaming response
# ---------------------------------------------------------------------------

def _parse_sse(text: str) -> list[dict]:
    """Parse SSE response body into a list of event dicts."""
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().splitlines()
        event_type = "message"
        data = None
        for line in lines:
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if data and event_type not in ("done",):
            try:
                events.append({"event": event_type, "data": json.loads(data)})
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Test 1 — first call streams real tokens (is_cached=False)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_call_is_not_cached(app, auth_headers, chat_body):
    tokens = ["Hello", " there", " world", "!", " How"]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, \
         patch("litellm.completion_cost", return_value=0.00005):
        mock_acompletion.return_value = _FakeStreamResponse(tokens)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/neetpg/chat",
                json=chat_body,
                headers=auth_headers,
            )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    message_events = [e for e in events if e["event"] == "message"]
    assert len(message_events) > 0

    # All real-call chunks should have is_cached=False
    for ev in message_events:
        assert ev["data"]["is_cached"] is False


# ---------------------------------------------------------------------------
# Test 2 — second identical call is served from cache (is_cached=True)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_call_is_cached(app, auth_headers, chat_body, tmp_path):
    settings = Settings(env="test", _env_file=None)  # type: ignore[call-arg]
    settings.llm_query_auth.allowed_token = "test-key"
    settings.cache.base_path = str(tmp_path / "cache2")
    fresh_app = create_app(settings=settings)

    tokens = ["Hi", " from", " cache"]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion, \
         patch("litellm.completion_cost", return_value=0.00005):
        mock_acompletion.return_value = _FakeStreamResponse(tokens)

        transport = ASGITransport(app=fresh_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First call — populates cache
            r1 = await client.post(
                "/v1/neetpg/chat",
                json=chat_body,
                headers=auth_headers,
            )
            assert r1.status_code == 200

            # Second call — identical inputs, should hit cache
            # litellm.acompletion should NOT be called again
            mock_acompletion.reset_mock()
            mock_acompletion.return_value = _FakeStreamResponse(tokens)

            r2 = await client.post(
                "/v1/neetpg/chat",
                json=chat_body,
                headers=auth_headers,
            )
            assert r2.status_code == 200

    events2 = _parse_sse(r2.text)
    message_events2 = [e for e in events2 if e["event"] == "message"]
    assert len(message_events2) > 0

    # All replayed chunks must carry is_cached=True
    for ev in message_events2:
        assert ev["data"]["is_cached"] is True

    # litellm should NOT have been called for the second request
    mock_acompletion.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — /v1/usage returns three-bucket snapshot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usage_endpoint_three_buckets(app, chat_body, tmp_path):
    settings = Settings(env="test", _env_file=None)  # type: ignore[call-arg]
    settings.llm_query_auth.allowed_token = "test-key"
    settings.cache.base_path = str(tmp_path / "cache3")
    fresh_app = create_app(settings=settings)

    headers_s2 = {"X-LlmQuery-Token": "test-key", "X-Session-Id": "session-usage"}
    tokens = ["A", "B", "C"]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac, \
         patch("litellm.completion_cost", return_value=0.0001):
        mock_ac.return_value = _FakeStreamResponse(tokens, cost=0.0001)

        transport = ASGITransport(app=fresh_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # First call — real
            mock_ac.return_value = _FakeStreamResponse(tokens, cost=0.0001)
            await client.post("/v1/neetpg/chat", json=chat_body, headers=headers_s2)

            # Second call — from cache
            mock_ac.return_value = _FakeStreamResponse(tokens, cost=0.0001)
            await client.post("/v1/neetpg/chat", json=chat_body, headers=headers_s2)

            # Query usage
            usage_resp = await client.get(
                "/v1/usage/neetpg:session-usage",
                headers={"X-LlmQuery-Token": "test-key"},
            )

    assert usage_resp.status_code == 200
    snap = usage_resp.json()
    assert "total" in snap
    assert "real" in snap
    assert "cached" in snap

    # Both calls contributed to total
    assert snap["total"]["call_count"] == 2
    # First call was real
    assert snap["real"]["call_count"] == 1
    # Second call was cached
    assert snap["cached"]["call_count"] == 1


# ---------------------------------------------------------------------------
# Test 4 — Premium tier + thinking role triggers parallel SSE events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_premium_parallel_thinking(app, tmp_path):
    from autogen.config.settings import QnATierConfig

    settings = Settings(env="test", _env_file=None)  # type: ignore[call-arg]
    settings.llm_query_auth.allowed_token = "test-key"
    settings.cache.base_path = str(tmp_path / "cache4")
    settings.qna.tier_configurations["Premium"] = QnATierConfig(
        explanation_model="test-model",
        conversation_model="test-model",
        thinking_model="test-model",
        parallel_thinking_models=["test-model-alpha", "test-model-beta", "test-model-gamma"],
        method_call_model="test-model",
        relevance_check_model="test-model",
        segment_finder_model="test-model",
        question_category_model="test-model",
        action_dispatcher_model="test-model",
        answer_extraction_model="test-model",
        option_explanation_model="test-model",
        detailed_explanation_model="test-model",
        short_explanation_model="test-model",
        hint_model="test-model",
    )
    fresh_app = create_app(settings=settings)

    body = {
        "tier": "Premium",
        "role": "thinking",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "complex reasoning question"}],
    }
    headers = {"X-LlmQuery-Token": "test-key", "X-Session-Id": "smoke-premium"}
    tokens = ["Think", "ing"]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac, \
         patch("litellm.completion_cost", return_value=0.001):
        # Return a fresh iterator each call (one per parallel model)
        mock_ac.side_effect = [
            _FakeStreamResponse(tokens),
            _FakeStreamResponse(tokens),
            _FakeStreamResponse(tokens),
        ]

        transport = ASGITransport(app=fresh_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/neetpg/chat", json=body, headers=headers
            )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    parallel_events = [e for e in events if e["event"] == "parallel"]

    # Premium has 3 thinking models → 3 parallel events
    assert len(parallel_events) == 3
    models_seen = {e["data"]["model"] for e in parallel_events}
    assert len(models_seen) == 3  # each parallel event names a different model


# ---------------------------------------------------------------------------
# Test 5 — missing auth header → 401
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_requires_auth(app, chat_body):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/neetpg/chat", json=chat_body)

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Test 6 — invalid tier → 400
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_tier_returns_400(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/neetpg/chat",
            json={
                "tier": "Bogus",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"X-LlmQuery-Token": "test-key"},
        )

    assert response.status_code == 400
