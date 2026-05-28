"""Day 10 — /v1/{app_id}/chat SSE endpoint & /v1/usage/{session_key}.

The smallest possible HTTP endpoints that exercise the full Phase 2 stack:
  - POST /v1/{app_id}/chat        — SSE stream from any provider via the decorator stack
  - GET  /v1/usage/{session_key}  — Three-bucket usage snapshot per session

These validate:
  - Real LLM streaming works (LiteLLM → Groq/OpenRouter/Anthropic)
  - Caching works (second identical call → cached with 20ms replay)
  - Usage tracking works (Total / Real / Cached buckets)
  - Tier-based model routing works (Free/Testing/Regular/Premium)
  - Parallel thinking fan-out works (Premium tier)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from structlog import get_logger

from autogen.api.deps import get_settings
from autogen.config.settings import Settings
from autogen.config.tiers import Tier
from autogen.llm import (
    SESSION_KEY_VAR,
    LlmChunk,
    LlmMessage,
    LlmUsage,
    build_llm_stack,
    create_router,
    get_usage_collector,
    load_models_catalog,
)
from autogen.protocols.llm import LlmClient

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Body for POST /v1/{app_id}/chat."""

    tier: str = "Free"  # Free | Testing | Regular | Premium
    messages: list[dict[str, str]]  # [{"role": "user", "content": "..."}]
    role: str = "conversation"  # One of the 13 MODEL_ROLES
    temperature: float = 0.0
    model: str | None = None  # Override — bypass router if set


class UsageResponse(BaseModel):
    """Three-bucket usage snapshot for a session."""

    total: dict[str, Any]
    real: dict[str, Any]
    cached: dict[str, Any]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


async def _sse_event(chunk: LlmChunk, event: str = "message") -> str:
    """Serialize one LlmChunk as an SSE event."""
    data = chunk.model_dump_json()
    return f"event: {event}\ndata: {data}\n\n"


async def _sse_error(message: str) -> str:
    """Serialize an error as an SSE event."""
    data = json.dumps({"error": message})
    return f"event: error\ndata: {data}\n\n"


# ---------------------------------------------------------------------------
# Internal — build the LLM stack (lazy singleton per process)
# ---------------------------------------------------------------------------

_llm_client: LlmClient | None = None
_tier_router: Any = None  # TierModelRouter
_models_catalog: Any = None  # ModelsCatalog


def _ensure_stack(settings: Settings) -> tuple[LlmClient, Any, Any]:
    """Lazily build the Phase 2 stack if not already built."""
    global _llm_client, _tier_router, _models_catalog

    if _llm_client is None:
        cache_dir = Path(settings.cache.base_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        _llm_client = build_llm_stack(
            cache_dir=cache_dir,
            memory_size=settings.cache.memory_size,
            memory_ttl=settings.cache.memory_ttl_seconds,
        )

        catalog_path = settings.qna.models_catalog_path
        _models_catalog = load_models_catalog(Path(catalog_path))

        _tier_router = create_router(
            tier_configurations=settings.qna.tier_configurations,
            catalog_path=Path(catalog_path),
        )

        logger.info(
            "phase2.stack_initialized",
            cache_dir=str(cache_dir),
            catalog_path=str(catalog_path),
            tier_count=len(settings.qna.tier_configurations),
        )

    return _llm_client, _tier_router, _models_catalog


# ---------------------------------------------------------------------------
# POST /v1/{app_id}/chat — SSE streaming endpoint
# ---------------------------------------------------------------------------


@router.post("/{app_id}/chat")
async def chat_stream(
    app_id: str,
    body: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Stream an LLM response via SSE, exercising the full Phase 2 stack.

    Headers:
        X-LlmQuery-Token: <token>  (validated by AuthMiddleware)
        X-Session-Id: <opt>        (optional — auto-generated if missing)

    Body (JSON):
        tier: "Free" | "Testing" | "Regular" | "Premium"
        messages: [{"role": "user", "content": "..."}]
        role: "conversation" | "thinking" | "explanation" | ...
        temperature: 0.0  (included in cache key)
        model: null  (optional override)

    Response: SSE stream of LlmChunk events (event: message)
              Final chunk carries usage info.
    """
    # --- Auth check (belt-and-suspenders; AuthMiddleware already validates) ---
    token = request.headers.get("X-LlmQuery-Token")
    if not token:
        raise HTTPException(status_code=401, detail="Missing X-LlmQuery-Token header")

    # --- Session key: encode app_id into it for multi-tenant usage attribution ---
    session_id = request.headers.get("X-Session-Id") or str(uuid.uuid4())
    session_key = f"{app_id}:{session_id}"
    SESSION_KEY_VAR.set(session_key)

    # --- Build/warm the stack ---
    llm_client, tier_router, catalog = _ensure_stack(settings)

    # --- Resolve model ---
    if body.model:
        model = body.model
    else:
        try:
            tier = Tier(body.tier)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier: {body.tier}. Must be one of {[t.value for t in Tier]}",
            )
        model = tier_router.model_for(tier, body.role)

    logger.info(
        "chat.stream.start",
        app_id=app_id,
        session_key=session_key,
        tier=body.tier,
        role=body.role,
        model=model,
        message_count=len(body.messages),
    )

    # --- Convert dict messages to LlmMessage Pydantic models ---
    messages = [
        LlmMessage(role=m["role"], content=m["content"]) for m in body.messages
    ]

    # --- Detect parallel thinking (Premium tier, complex query, thinking role) ---
    parallel_models: list[str] = []
    try:
        tier = Tier(body.tier)
        if body.role == "thinking" and tier == Tier.PREMIUM:
            parallel_models = tier_router.parallel_thinking_models(tier)
            # If only 1 model, it's not a true fan-out — clear the list
            if len(parallel_models) <= 1:
                parallel_models = []
    except ValueError:
        pass  # Invalid tier → fall back to single-model path below

    async def _generate_sse():
        """Generate SSE events from the LLM call(s)."""
        try:
            if parallel_models:
                # Premium parallel thinking: fan out to N models concurrently
                async for event in _run_parallel_thinking(
                    llm_client, messages, parallel_models, body.temperature
                ):
                    yield event
            else:
                # Single-model path
                async for event in _run_single_stream(
                    llm_client, messages, model, body.temperature
                ):
                    yield event

            # Final DONE event
            yield "event: done\ndata: {}\n\n"

        except Exception as exc:
            logger.error("chat.stream.error", error=str(exc), model=model)
            yield await _sse_error(str(exc))

    return StreamingResponse(
        _generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


async def _run_single_stream(
    llm_client: LlmClient,
    messages: list[LlmMessage],
    model: str,
    temperature: float,
):
    """Execute a single-model stream and yield SSE events."""
    async for chunk in llm_client.stream(
        messages=messages,
        model=model,
        temperature=temperature,
    ):
        yield await _sse_event(chunk)


async def _run_parallel_thinking(
    llm_client: LlmClient,
    messages: list[LlmMessage],
    parallel_models: list[str],
    temperature: float,
):
    """Fan out to N thinking models concurrently, yield merged SSE stream."""
    logger.info("chat.parallel_thinking.start", model_count=len(parallel_models))

    async def _stream_one(model: str) -> tuple[str, list[LlmChunk]]:
        """Stream from one model, return (output, chunks)."""
        output = ""
        chunks: list[LlmChunk] = []
        async for chunk in llm_client.stream(
            messages=messages,
            model=model,
            temperature=temperature,
        ):
            output += chunk.delta or ""
            chunks.append(chunk)
        return output, chunks

    # Launch all N streams concurrently
    tasks = [_stream_one(m) for m in parallel_models]
    results = await asyncio.gather(*tasks)

    # Yield merged results — one "parallel" event per model
    for model_name, (output, chunks) in zip(parallel_models, results):
        usage_chunk = chunks[-1] if chunks else LlmChunk(delta="")
        data = json.dumps({
            "model": model_name,
            "output": output,
            "usage": usage_chunk.usage.model_dump() if usage_chunk.usage else None,
        })
        yield f"event: parallel\ndata: {data}\n\n"

    logger.info("chat.parallel_thinking.complete", model_count=len(parallel_models))
    return


# ---------------------------------------------------------------------------
# GET /v1/usage/{session_key} — three-bucket usage snapshot
# ---------------------------------------------------------------------------


@router.get("/usage/{session_key:path}", response_model=UsageResponse)
async def get_usage(
    session_key: str,
    request: Request,
):
    """Return the three-bucket usage snapshot for a session.

    Session key format: {app_id}:{session_id} (e.g. "neetpg:smoke-1")

    Response:
        {
          "total":  {"total_cost": 0.052, "total_tokens": 6840, ...},
          "real":   {"total_cost": 0.052, ...},
          "cached": {"total_cost": 0.0, ...}
        }

    If the session key starts with a known app_id prefix, all calls
    under that prefix are aggregated (for per-tenant billing).
    """
    collector = get_usage_collector()
    snap = collector.snapshot(session_key)
    return UsageResponse(
        total=snap["total"],
        real=snap["real"],
        cached=snap["cached"],
    )


# ---------------------------------------------------------------------------
# GET /v1/catalog — inspect the loaded models catalog (debug)
# ---------------------------------------------------------------------------


@router.get("/catalog")
async def get_catalog(
    settings: Settings = Depends(get_settings),
):
    """Return the loaded models catalog (for debugging)."""
    _, _, catalog = _ensure_stack(settings)
    return catalog.model_dump()