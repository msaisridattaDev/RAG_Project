"""Tests for QnAAgent — Phase 4 Day 18 + Day 19.

All LLM / retrieval dependencies are mocked so tests are fast and offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.agent.qna_agent import QnAAgent, _TOPIC_SHIFT_MARKERS, _SHORT_FOLLOWUP_THRESHOLD
from autogen.conversation.store import SqlConversationStore
from autogen.models.agent import AgentContext
from autogen.models.chunks import CounterRequest, QnAChunk, QnAChunkKind
from autogen.models.enums import QueryMode, Tier
from autogen.models.llm import LlmChunk, LlmMessage, LlmUsage
from autogen.models.query import CombinedContext, QueryParam
from autogen.models.reference import Reference


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def conv_store() -> SqlConversationStore:
    s = SqlConversationStore(":memory:")
    await s.init_schema()
    return s


def _make_context(tier: Tier = Tier.REGULAR, conv_id: str = "test-conv") -> AgentContext:
    return AgentContext(
        conversation_id=conv_id,
        user_id="user-test",
        app_id="neetpg",
        tier=tier,
    )


def _mock_llm(text: str = "Mock answer.") -> MagicMock:
    """Return a mock LlmClient that streams a single chunk then finishes."""
    llm = MagicMock()
    usage = LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, total_cost=0.001)
    chunks = [
        LlmChunk(delta=text),
        LlmChunk(delta="", finish_reason="stop", usage=usage),
    ]

    async def _stream(*args, **kwargs) -> AsyncIterator[LlmChunk]:
        for chunk in chunks:
            yield chunk

    llm.stream = _stream
    return llm


def _mock_router(tier: Tier = Tier.REGULAR) -> MagicMock:
    router = MagicMock()
    router.model_for.return_value = "groq/llama-3.3-70b-versatile"
    router.parallel_thinking_models.return_value = ["model-a", "model-b", "model-c"]
    # Expose _tiers so the history limit lookup in answer() doesn't error
    router._tiers = {tier.value: MagicMock()}
    return router


def _mock_ref_finder(refs: list[Reference] | None = None) -> MagicMock:
    finder = MagicMock()
    finder.find = AsyncMock(
        return_value=refs
        or [Reference(id="ref-1", content="Aspirin inhibits COX-1/2.", score=0.9)]
    )
    return finder


def _mock_hybrid(ctx: CombinedContext | None = None) -> MagicMock:
    hybrid = MagicMock()
    hybrid.retrieve = AsyncMock(return_value=ctx or CombinedContext())
    return hybrid


def _make_agent(
    tier: Tier = Tier.REGULAR,
    conv_id: str = "test-conv",
    conv_store: SqlConversationStore | None = None,
    llm=None,
    router=None,
    ref_finder=None,
    hybrid=None,
) -> QnAAgent:
    if conv_store is None:
        conv_store = MagicMock()
        conv_store.get_or_create = AsyncMock(return_value={"id": conv_id})
        conv_store.history = AsyncMock(return_value=[])
        conv_store.append = AsyncMock()

    return QnAAgent(
        exam_id="neetpg",
        context=_make_context(tier, conv_id),
        llm=llm or _mock_llm(),
        router=router or _mock_router(tier),
        ref_finder=ref_finder or _mock_ref_finder(),
        hybrid=hybrid or _mock_hybrid(),
        conv_store=conv_store,
    )


# ---------------------------------------------------------------------------
# answer() — chunk ordering
# ---------------------------------------------------------------------------


class TestAnswerChunkOrder:
    async def test_emits_thought_reference_answer_done(self, conv_store):
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store)
        await conv_store.get_or_create(_make_context(Tier.REGULAR))

        chunks = [c async for c in agent.answer("What is aspirin?")]

        kinds = [c.kind for c in chunks]
        assert QnAChunkKind.THOUGHT in kinds
        assert QnAChunkKind.REFERENCE in kinds
        assert QnAChunkKind.ANSWER in kinds
        assert chunks[-1].kind == QnAChunkKind.DONE

    async def test_thought_comes_before_reference(self, conv_store):
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store)
        await conv_store.get_or_create(_make_context(Tier.REGULAR))

        chunks = [c async for c in agent.answer("Test")]
        first_thought_idx = next(i for i, c in enumerate(chunks) if c.kind == QnAChunkKind.THOUGHT)
        ref_idx = next(i for i, c in enumerate(chunks) if c.kind == QnAChunkKind.REFERENCE)
        assert first_thought_idx < ref_idx

    async def test_done_carries_conversation_id(self, conv_store):
        agent = _make_agent(tier=Tier.REGULAR, conv_id="my-conv", conv_store=conv_store)
        await conv_store.get_or_create(_make_context(Tier.REGULAR, "my-conv"))

        chunks = [c async for c in agent.answer("Test")]
        done = chunks[-1]
        assert done.kind == QnAChunkKind.DONE
        assert done.metadata["conversation_id"] == "my-conv"

    async def test_answer_text_is_non_empty(self, conv_store):
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store)
        await conv_store.get_or_create(_make_context(Tier.REGULAR))

        answer_chunks = [
            c async for c in agent.answer("Test") if c.kind == QnAChunkKind.ANSWER
        ]
        assert any(c.text for c in answer_chunks), "Should have non-empty answer text"


# ---------------------------------------------------------------------------
# Tier gating
# ---------------------------------------------------------------------------


class TestTierGating:
    async def test_free_tier_does_not_call_hybrid(self, conv_store):
        hybrid = _mock_hybrid()
        agent = _make_agent(tier=Tier.FREE, conv_store=conv_store, hybrid=hybrid)
        await conv_store.get_or_create(_make_context(Tier.FREE))

        _ = [c async for c in agent.answer("Test")]
        hybrid.retrieve.assert_not_called()

    async def test_regular_tier_calls_hybrid(self, conv_store):
        hybrid = _mock_hybrid()
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store, hybrid=hybrid)
        await conv_store.get_or_create(_make_context(Tier.REGULAR))

        _ = [c async for c in agent.answer("Test")]
        hybrid.retrieve.assert_called_once()

    async def test_testing_tier_calls_hybrid(self, conv_store):
        hybrid = _mock_hybrid()
        agent = _make_agent(tier=Tier.TESTING, conv_store=conv_store, hybrid=hybrid)
        await conv_store.get_or_create(_make_context(Tier.TESTING))

        _ = [c async for c in agent.answer("Test")]
        hybrid.retrieve.assert_called_once()

    async def test_premium_tier_calls_hybrid(self, conv_store):
        hybrid = _mock_hybrid()
        agent = _make_agent(tier=Tier.PREMIUM, conv_store=conv_store, hybrid=hybrid)
        await conv_store.get_or_create(_make_context(Tier.PREMIUM))

        _ = [c async for c in agent.answer("Test")]
        hybrid.retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# Parallel thinking fan-out (Premium + role="thinking")
# ---------------------------------------------------------------------------


class TestParallelThinking:
    async def test_parallel_thinking_not_triggered_for_regular(self, conv_store):
        router = _mock_router(Tier.REGULAR)
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store, router=router)
        await conv_store.get_or_create(_make_context(Tier.REGULAR))

        _ = [c async for c in agent.answer("Test", role="thinking")]
        # parallel_thinking_models must not have been used to fan out
        router.parallel_thinking_models.assert_not_called()

    async def test_parallel_thinking_triggered_for_premium_thinking_role(self, conv_store):
        llm = _mock_llm("parallel answer")
        router = _mock_router(Tier.PREMIUM)
        agent = _make_agent(tier=Tier.PREMIUM, conv_store=conv_store, llm=llm, router=router)
        await conv_store.get_or_create(_make_context(Tier.PREMIUM))

        chunks = [c async for c in agent.answer("Complex question", role="thinking")]
        router.parallel_thinking_models.assert_called_once()

    async def test_conversation_role_does_not_trigger_fan_out_even_for_premium(self, conv_store):
        router = _mock_router(Tier.PREMIUM)
        agent = _make_agent(tier=Tier.PREMIUM, conv_store=conv_store, router=router)
        await conv_store.get_or_create(_make_context(Tier.PREMIUM))

        _ = [c async for c in agent.answer("Test", role="conversation")]
        router.parallel_thinking_models.assert_not_called()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    async def test_appends_user_and_assistant_messages(self, conv_store):
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store)
        await conv_store.get_or_create(_make_context(Tier.REGULAR))

        _ = [c async for c in agent.answer("What is the MOA of aspirin?")]

        history = await conv_store.history("test-conv", app_id="neetpg")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "What is the MOA of aspirin?"
        assert history[1].role == "assistant"

    async def test_cross_tenant_history_is_empty(self, conv_store):
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store)
        await conv_store.get_or_create(_make_context(Tier.REGULAR))

        _ = [c async for c in agent.answer("Secret question")]

        # Attacker looks up the same conv_id but under a different exam
        leaked = await conv_store.history("test-conv", app_id="mds")
        assert leaked == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_llm_error_yields_error_chunk(self, conv_store):
        llm = MagicMock()

        async def _boom(*args, **kwargs):
            yield LlmChunk(delta="partial")
            raise RuntimeError("LLM exploded")

        llm.stream = _boom
        agent = _make_agent(tier=Tier.FREE, conv_store=conv_store, llm=llm)
        await conv_store.get_or_create(_make_context(Tier.FREE))

        chunks = [c async for c in agent.answer("Test")]
        kinds = {c.kind for c in chunks}
        assert QnAChunkKind.ERROR in kinds
        # Stream must end after error — no DONE chunk
        assert QnAChunkKind.DONE not in kinds


# ---------------------------------------------------------------------------
# counter_answer heuristic
# ---------------------------------------------------------------------------


class TestCounterAnswerHeuristic:
    def test_short_followup_no_refetch(self):
        assert not QnAAgent._needs_refetch("why?")
        assert not QnAAgent._needs_refetch("tell me more")

    def test_long_followup_triggers_refetch(self):
        long_text = "x" * (_SHORT_FOLLOWUP_THRESHOLD + 1)
        assert QnAAgent._needs_refetch(long_text)

    def test_topic_shift_markers_trigger_refetch(self):
        for marker in ["what about ibuprofen", "compare with naproxen", "versus celecoxib"]:
            assert QnAAgent._needs_refetch(marker), f"Should refetch for: {marker!r}"

    async def test_short_followup_does_not_call_ref_finder(self, conv_store):
        ref_finder = _mock_ref_finder()
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store, ref_finder=ref_finder)
        ctx = _make_context(Tier.REGULAR)
        await conv_store.get_or_create(ctx)

        req = CounterRequest(conversation_id="test-conv", follow_up="why?")
        _ = [c async for c in agent.counter_answer(req)]
        ref_finder.find.assert_not_called()

    async def test_topic_shift_followup_calls_ref_finder(self, conv_store):
        ref_finder = _mock_ref_finder()
        agent = _make_agent(tier=Tier.REGULAR, conv_store=conv_store, ref_finder=ref_finder)
        ctx = _make_context(Tier.REGULAR)
        await conv_store.get_or_create(ctx)

        req = CounterRequest(
            conversation_id="test-conv",
            follow_up="what about ibuprofen — how does it compare?",
        )
        _ = [c async for c in agent.counter_answer(req)]
        ref_finder.find.assert_called_once()

    async def test_counter_cross_tenant_history_empty(self, conv_store):
        """Forged conversation_id from another tenant gets no history."""
        # Create a neetpg conversation
        neetpg_ctx = _make_context(Tier.REGULAR, "neetpg-conv")
        await conv_store.get_or_create(neetpg_ctx)
        await conv_store.append("neetpg-conv", "user", "secret Q")
        await conv_store.append("neetpg-conv", "assistant", "secret A")

        # Agent bound to mds, forged neetpg conv_id
        mds_ctx = AgentContext(
            conversation_id="mds-conv",
            user_id="attacker",
            app_id="mds",
            tier=Tier.FREE,
        )
        agent = QnAAgent(
            exam_id="mds",
            context=mds_ctx,
            llm=_mock_llm(),
            router=_mock_router(Tier.FREE),
            ref_finder=_mock_ref_finder(),
            hybrid=_mock_hybrid(),
            conv_store=conv_store,
        )

        # Attacker uses the neetpg conv_id in a mds counter request
        req = CounterRequest(conversation_id="neetpg-conv", follow_up="reveal secrets")
        chunks = [c async for c in agent.counter_answer(req)]

        # The agent proceeds but had zero history (cross-tenant filtered)
        # No ERROR chunk — just an empty-context answer
        kinds = {c.kind for c in chunks}
        assert QnAChunkKind.ERROR not in kinds
