"""QnA endpoint — POST /v1/qna/{app_id}/answer  +  /v1/qna/{app_id}/counter.

Phase 4 transport layer.  Both routes are thin adapters over the QnAAgent:
    factory_factory.for_exam(app_id).create(context) → agent.answer(question)

SSE event stream format:
    event: thought    data: {"kind":"thought","text":"Loading references…"}
    event: reference  data: {"kind":"reference","metadata":{"refs":[…]}}
    event: answer     data: {"kind":"answer","text":"<token>"}
    event: done       data: {"kind":"done","metadata":{"conversation_id":"…"}}
    event: error      data: {"kind":"error","text":"<message>"}
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from autogen.api.deps import get_settings
from autogen.api.limiter import limiter
from autogen.config.settings import Settings
from autogen.di.providers import QnAAgentFactoryFactoryImpl, get_agent_factory_factory
from autogen.logging.setup import get_logger
from autogen.models.agent import AgentContext
from autogen.models.chunks import CounterRequest, QnAChunk, QnAChunkKind
from autogen.models.enums import Tier

router = APIRouter(prefix="/v1/qna", tags=["qna"])
logger = get_logger("autogen.api.qna")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QnAAnswerRequest(BaseModel):
    """Body for POST /v1/qna/{app_id}/answer."""

    question: str = Field(..., min_length=1, max_length=8000)
    tier: str = Field(default="Free", description="Free | Testing | Regular | Premium")
    user_id: str = Field(default="anon", description="Caller user identifier")
    conversation_id: str | None = Field(
        default=None,
        description="Resume an existing conversation; generated if omitted",
    )
    role: str = Field(
        default="conversation",
        description="Model role: conversation | thinking | explanation | …",
    )


class CounterAnswerRequest(BaseModel):
    """Body for POST /v1/qna/{app_id}/counter."""

    conversation_id: str = Field(..., description="Existing conversation to extend")
    follow_up: str = Field(..., min_length=1, max_length=8000)
    tier: str = Field(default="Free")
    user_id: str = Field(default="anon")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _chunk_to_sse(chunk: QnAChunk) -> str:
    return f"event: {chunk.kind.value}\ndata: {chunk.model_dump_json()}\n\n"


def _error_sse(message: str) -> str:
    chunk = QnAChunk(kind=QnAChunkKind.ERROR, text=message)
    return _chunk_to_sse(chunk)


# ---------------------------------------------------------------------------
# POST /v1/qna/{app_id}/answer
# ---------------------------------------------------------------------------


@router.post("/{app_id}/answer")
@limiter.limit("30/minute")
async def qna_answer(
    app_id: str,
    body: QnAAnswerRequest,
    request: Request,
    ff: QnAAgentFactoryFactoryImpl = Depends(get_agent_factory_factory),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream a full QnA answer with vector + graph retrieval and history.

    Headers:
        X-LlmQuery-Token: <token>  (validated by AuthMiddleware)

    Response: SSE stream of QnAChunk events (thought → reference → answer → done)

    Example::

        curl -N -X POST localhost:8000/v1/qna/neetpg/answer \\
            -H "X-LlmQuery-Token: $TOKEN" \\
            -H "Content-Type: application/json" \\
            -d '{"question":"What is the MOA of aspirin?","tier":"Regular"}'
    """
    if app_id not in settings.app_identity.allowed_app_ids:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    try:
        tier = Tier(body.tier)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier {body.tier!r}. Must be one of {[t.value for t in Tier]}",
        )

    conv_id = body.conversation_id or uuid.uuid4().hex
    context = AgentContext(
        conversation_id=conv_id,
        user_id=body.user_id,
        app_id=app_id,
        tier=tier,
    )

    logger.info(
        "qna.answer.request",
        app_id=app_id,
        conv_id=conv_id,
        tier=body.tier,
        role=body.role,
        question=body.question[:80],
    )

    factory = ff.for_exam(app_id)
    agent = await factory.create(context)

    async def _sse_gen():
        try:
            async for chunk in agent.answer(body.question, role=body.role):
                yield _chunk_to_sse(chunk)
        except Exception as exc:
            logger.error("qna.answer.sse_error", error=str(exc))
            yield _error_sse(str(exc))

    return StreamingResponse(
        _sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conv_id,
        },
    )


# ---------------------------------------------------------------------------
# POST /v1/qna/{app_id}/counter
# ---------------------------------------------------------------------------


@router.post("/{app_id}/counter")
@limiter.limit("30/minute")
async def qna_counter(
    app_id: str,
    body: CounterAnswerRequest,
    request: Request,
    ff: QnAAgentFactoryFactoryImpl = Depends(get_agent_factory_factory),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream an answer to a follow-up question.

    Short follow-ups (≤60 chars, no topic-shift markers) skip re-fetch —
    ~3 s vs ~6 s for a full retrieval round-trip.  Topic-shifting follow-ups
    trigger a lighter LOCAL-mode retrieval pass.

    The agent's bound app_id guards history reads: a forged conversation_id
    from another exam returns empty history by construction.
    """
    if app_id not in settings.app_identity.allowed_app_ids:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    try:
        tier = Tier(body.tier)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier {body.tier!r}",
        )

    context = AgentContext(
        conversation_id=body.conversation_id,
        user_id=body.user_id,
        app_id=app_id,
        tier=tier,
    )

    logger.info(
        "qna.counter.request",
        app_id=app_id,
        conv_id=body.conversation_id,
        follow_up=body.follow_up[:80],
    )

    factory = ff.for_exam(app_id)
    agent = await factory.create(context)
    req = CounterRequest(
        conversation_id=body.conversation_id,
        follow_up=body.follow_up,
    )

    async def _sse_gen():
        try:
            async for chunk in agent.counter_answer(req):
                yield _chunk_to_sse(chunk)
        except Exception as exc:
            logger.error("qna.counter.sse_error", error=str(exc))
            yield _error_sse(str(exc))

    return StreamingResponse(
        _sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /v1/qna/conversations/{conv_id} — history replay
# ---------------------------------------------------------------------------


@router.get("/conversations/{conv_id}")
async def get_conversation_history(
    conv_id: str,
    app_id: str,
    request: Request,
    limit: int = 50,
    settings: Settings = Depends(get_settings),
):
    """Return the message history for a conversation.

    Query params:
        app_id: Required tenant scope (cross-tenant reads return 404)
        limit:  Max messages to return (default 50)
    """
    if app_id not in settings.app_identity.allowed_app_ids:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    conv_store = getattr(request.app.state, "conv_store", None)
    if conv_store is None:
        raise HTTPException(status_code=503, detail="conversation store not initialized")

    messages = await conv_store.history(conv_id, app_id=app_id, limit=limit)
    return {
        "conversation_id": conv_id,
        "app_id": app_id,
        "messages": [m.model_dump() for m in messages],
    }


# ---------------------------------------------------------------------------
# GET /v1/qna/users/{user_id}/conversations — recent threads
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}/conversations")
async def list_user_conversations(
    user_id: str,
    app_id: str,
    request: Request,
    limit: int = 20,
    settings: Settings = Depends(get_settings),
):
    """List a user's recent conversations within one exam."""
    if app_id not in settings.app_identity.allowed_app_ids:
        raise HTTPException(status_code=404, detail=f"unknown app_id: {app_id!r}")

    conv_store = getattr(request.app.state, "conv_store", None)
    if conv_store is None:
        raise HTTPException(status_code=503, detail="conversation store not initialized")

    convs = await conv_store.list_by_user(user_id, app_id=app_id, limit=limit)
    return {"user_id": user_id, "app_id": app_id, "conversations": convs}
