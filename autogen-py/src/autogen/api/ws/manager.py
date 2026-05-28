"""WebSocket manager — Day 21.

Handles the persistent bidirectional channel at /QnA/studypal/ws
(exact source path — mirrors Program.cs:139, 361).

Architecture:
    One connection → three concurrent asyncio tasks:
        _receive_loop  — reads client messages, dispatches question tasks
        _send_loop     — drains the bounded send queue, writes to WebSocket,
                         appends to ring buffer
        _heartbeat     — pings every 30 s, closes if no pong within 60 s

Backpressure:
    send_channel is asyncio.Queue(maxsize=100). When it's full, _dispatch
    blocks on put(), preventing the server from buffering unbounded memory
    for a slow client.

Resume:
    ring_buffer is deque(maxlen=200) of (timestamp, envelope_json) tuples.
    On a "resume" message the server replays entries newer than since_ts.
    The buffer lives for the lifetime of the connection — if the server
    restarts the buffer is gone (documented limitation).

Multi-exam:
    Each "question" envelope carries its own app_id in the payload.
    _dispatch builds a fresh ConversationRuntimeContext and factory chain
    per question, so one connection can interleave questions across exams.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from autogen.api.ws.envelope import Envelope, QuestionPayload, ResumePayload
from autogen.config.settings import Settings
from autogen.di.providers import QnAAgentFactoryFactoryImpl
from autogen.logging.setup import get_logger
from autogen.models.agent import AgentContext
from autogen.models.enums import Tier

logger = get_logger("autogen.api.ws")

_HEARTBEAT_INTERVAL = 30.0   # send ping every N seconds
_PONG_DEADLINE = 60.0        # close connection if no pong within N seconds
_QUEUE_MAXSIZE = 100         # backpressure bound
_RING_BUFFER_SIZE = 200      # resume ring buffer capacity


# ---------------------------------------------------------------------------
# Per-connection state
# ---------------------------------------------------------------------------


@dataclass
class _ConnectionState:
    send_channel: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=_QUEUE_MAXSIZE))
    ring_buffer: deque = field(default_factory=lambda: deque(maxlen=_RING_BUFFER_SIZE))
    last_pong: float = field(default_factory=time.monotonic)
    active_tasks: list[asyncio.Task] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class QnAWebSocketManager:
    """Manages one WebSocket connection's full lifecycle.

    Usage::

        manager = QnAWebSocketManager(factory_factory=get_agent_factory_factory())
        await manager.handle(websocket, settings=settings)
    """

    def __init__(self, factory_factory: QnAAgentFactoryFactoryImpl) -> None:
        self._ff = factory_factory

    async def handle(self, ws: WebSocket, settings: Settings) -> None:
        """Accept and drive the WebSocket connection until it closes."""
        await ws.accept()
        state = _ConnectionState()

        # Launch the three concurrent tasks
        receive_task = asyncio.create_task(self._receive_loop(ws, state, settings))
        send_task = asyncio.create_task(self._send_loop(ws, state))
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws, state))

        driver_tasks = {receive_task, send_task, heartbeat_task}
        state.active_tasks.extend(driver_tasks)

        try:
            # Wait until any driver task exits (normally or via exception)
            done, pending = await asyncio.wait(driver_tasks, return_when=asyncio.FIRST_COMPLETED)

            # Propagate the exception from the first failed task (for logging)
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    logger.warning("ws.driver_task_failed", error=str(exc))
        finally:
            # Cancel everything — question dispatch tasks too
            for t in list(state.active_tasks):
                if not t.done():
                    t.cancel()
            # Give tasks a moment to acknowledge cancellation
            await asyncio.gather(*state.active_tasks, return_exceptions=True)
            try:
                await ws.close()
            except Exception:
                pass
            logger.info("ws.connection_closed")

    # ---------------------------------------------------------------------------
    # Receive loop
    # ---------------------------------------------------------------------------

    async def _receive_loop(
        self, ws: WebSocket, state: _ConnectionState, settings: Settings
    ) -> None:
        """Read client messages and dispatch question tasks."""
        allowed_app_ids = set(settings.app_identity.allowed_app_ids)

        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect:
                logger.info("ws.client_disconnected")
                return

            try:
                env = Envelope.model_validate_json(raw)
            except (ValidationError, Exception) as exc:
                err = Envelope.error(f"malformed envelope: {exc}")
                await self._enqueue(state, err)
                continue

            if env.type == "pong":
                state.last_pong = time.monotonic()

            elif env.type == "question":
                try:
                    q_payload = QuestionPayload.model_validate(env.payload)
                except ValidationError as exc:
                    await self._enqueue(state, Envelope.error(str(exc), env.correlation_id))
                    continue

                if q_payload.app_id not in allowed_app_ids:
                    await self._enqueue(
                        state,
                        Envelope.error(
                            f"unknown app_id: {q_payload.app_id!r}", env.correlation_id
                        ),
                    )
                    continue

                task = asyncio.create_task(
                    self._dispatch(env.correlation_id, q_payload, state)
                )
                state.active_tasks.append(task)

            elif env.type == "resume":
                try:
                    r_payload = ResumePayload.model_validate(env.payload)
                except ValidationError:
                    r_payload = ResumePayload()
                await self._replay(state, r_payload.since_ts)

            else:
                logger.debug("ws.unknown_envelope_type", type=env.type)

    # ---------------------------------------------------------------------------
    # Send loop
    # ---------------------------------------------------------------------------

    async def _send_loop(self, ws: WebSocket, state: _ConnectionState) -> None:
        """Drain send_channel, write each envelope to the WebSocket, record in ring buffer."""
        while True:
            env: Envelope = await state.send_channel.get()
            json_str = env.model_dump_json()
            try:
                await ws.send_text(json_str)
                state.ring_buffer.append((time.monotonic(), json_str))
            except Exception as exc:
                logger.warning("ws.send_failed", error=str(exc))
                return

    # ---------------------------------------------------------------------------
    # Heartbeat loop
    # ---------------------------------------------------------------------------

    async def _heartbeat_loop(self, ws: WebSocket, state: _ConnectionState) -> None:
        """Ping every 30 s; raise if no pong received within 60 s."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            await self._enqueue(state, Envelope.ping())

            if time.monotonic() - state.last_pong > _PONG_DEADLINE:
                logger.warning("ws.heartbeat_timeout — closing connection")
                raise RuntimeError("WebSocket pong deadline exceeded")

    # ---------------------------------------------------------------------------
    # Question dispatch
    # ---------------------------------------------------------------------------

    async def _dispatch(
        self,
        correlation_id: str,
        payload: QuestionPayload,
        state: _ConnectionState,
    ) -> None:
        """Build agent for this question's app_id and stream answer chunks."""
        conv_id = payload.conversation_id or uuid.uuid4().hex

        try:
            tier = Tier(payload.tier)
        except ValueError:
            tier = Tier.FREE

        ctx = AgentContext(
            conversation_id=conv_id,
            user_id=payload.user_id,
            app_id=payload.app_id,
            tier=tier,
        )

        try:
            factory = self._ff.for_exam(payload.app_id)
            agent = await factory.create(ctx)
        except Exception as exc:
            await self._enqueue(state, Envelope.error(f"agent init failed: {exc}", correlation_id))
            return

        try:
            async for chunk in agent.answer(payload.question):
                out_env = Envelope.chunk_from(chunk.model_dump(), correlation_id)
                await self._enqueue(state, out_env)
        except asyncio.CancelledError:
            # Client disconnected mid-stream — clean exit, no log spam
            raise
        except Exception as exc:
            logger.error("ws.dispatch_error", correlation_id=correlation_id, error=str(exc))
            await self._enqueue(
                state,
                Envelope.error(f"answer stream failed: {exc}", correlation_id),
            )

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    async def _enqueue(self, state: _ConnectionState, env: Envelope) -> None:
        """Put envelope on the send channel.

        Uses put() (blocking) so slow clients create backpressure rather than
        unbounded memory growth. Drops the envelope and logs if it can't be
        placed within 5 s (extreme slow client — preferable to OOM).
        """
        try:
            await asyncio.wait_for(state.send_channel.put(env), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "ws.backpressure_drop",
                type=env.type,
                correlation_id=env.correlation_id,
            )

    async def _replay(self, state: _ConnectionState, since_ts: float) -> None:
        """Re-enqueue ring buffer entries newer than since_ts."""
        count = 0
        for ts, json_str in state.ring_buffer:
            if ts >= since_ts:
                try:
                    env = Envelope.model_validate_json(json_str)
                    await state.send_channel.put(env)
                    count += 1
                except Exception:
                    pass
        logger.info("ws.resume_replay", count=count, since_ts=since_ts)
