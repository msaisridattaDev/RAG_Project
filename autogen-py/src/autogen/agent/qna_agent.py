"""QnAAgent — Phase 4 Day 18 + Day 19 orchestration.

The chef of the system.  Constructed exclusively by the two-level factory
hierarchy so that exam-scoped stores are cached once and per-conversation
context is cheap:

    QnAAgentFactoryFactory.for_exam(exam_id)   ← outer (cached per exam)
        → QnAAgentFactory.create(context)       ← inner (per request)
            → QnAAgent                          ← this class

Every transport (REST+SSE, WebSocket, MCP) calls::

    async for chunk in agent.answer(question):
        yield chunk  # or send over WebSocket

Structured chunks (QnAChunkKind) let the UI render each event distinctly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import tiktoken

from autogen.logging.setup import get_logger
from autogen.models.agent import AgentContext
from autogen.models.chunks import CounterRequest, QnAChunk, QnAChunkKind
from autogen.models.enums import QueryMode, Tier
from autogen.models.llm import LlmMessage
from autogen.models.query import QueryParam

if TYPE_CHECKING:
    from autogen.conversation.store import SqlConversationStore
    from autogen.llm.catalog import TierModelRouter
    from autogen.models.query import CombinedContext
    from autogen.models.reference import Reference
    from autogen.protocols.llm import LlmClient
    from autogen.retrieval.finder import ReferenceFinder
    from autogen.retrieval.hybrid import HybridRetrieval

logger = get_logger("autogen.agent.qna_agent")

# Tiers that get graph-RAG retrieval (Phase 3 HybridRetrieval)
_GRAPH_TIERS = frozenset({Tier.TESTING, Tier.REGULAR, Tier.PREMIUM})

# Topic-shift markers that force a re-fetch in counter_answer
_TOPIC_SHIFT_MARKERS = frozenset(
    {
        "what about",
        "tell me more about",
        "and ",
        "differ",
        "compare",
        "another",
        "versus",
        "vs ",
        "instead",
        "alternative",
    }
)

# Follow-ups longer than this are assumed to introduce new topics
_SHORT_FOLLOWUP_THRESHOLD = 60

# Maximum tokens we inject as retrieval context into the LLM prompt
_MAX_CONTEXT_TOKENS = 8_000

# System prompt used when no PromptLibrary is wired in
_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert medical education assistant helping students prepare for "
    "NEET PG, NEET UG, MDS, and EMS examinations.\n\n"
    "Use the provided references and context to answer questions accurately and "
    "comprehensively. Structure your answer clearly with important points "
    "highlighted. Reference relevant concepts from the provided context. "
    "If the context does not fully cover the question, draw on your medical "
    "knowledge and clearly indicate where you are doing so. "
    "Keep answers focused and exam-relevant."
)


class QnAAgent:
    """Operational QnA agent — the orchestration layer for one conversation.

    Bound to a specific exam (``_exam_id``) and conversation (``_context``) at
    construction time.  Never instantiate directly — always use the factory::

        factory = factory_factory.for_exam(app_id)
        agent   = await factory.create(context)
    """

    def __init__(
        self,
        *,
        exam_id: str,
        context: AgentContext,
        llm: LlmClient,
        router: TierModelRouter,
        ref_finder: ReferenceFinder,
        hybrid: HybridRetrieval,
        conv_store: SqlConversationStore,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._exam_id = exam_id
        self._context = context
        self._llm = llm
        self._router = router
        self._ref_finder = ref_finder
        self._hybrid = hybrid
        self._conv_store = conv_store
        self._system_prompt = system_prompt
        self._enc = tiktoken.get_encoding("cl100k_base")

    @property
    def context(self) -> AgentContext:
        """The conversation context bound to this agent instance."""
        return self._context

    # ------------------------------------------------------------------
    # Primary public method — Day 18
    # ------------------------------------------------------------------

    async def answer(
        self,
        question: str,
        role: str = "conversation",
    ) -> AsyncGenerator[QnAChunk, None]:
        """Stream a full QnA answer with retrieval, history, and persistence.

        Args:
            question: The user's question.
            role: Model role to use.  Defaults to "conversation"; pass
                  "thinking" to enable the parallel-thinking fan-out
                  for Premium tier.

        Yields:
            QnAChunk events in order:
                thought → reference → thought → answer* → done
                (or error on failure)
        """
        # 1. Ensure conversation row exists (idempotent)
        await self._conv_store.get_or_create(self._context)

        # 2. Status update
        yield QnAChunk(kind=QnAChunkKind.THOUGHT, text="Loading references…")

        # 3. Parallel retrieval (graph gated by tier)
        refs, graph_ctx = await self._fetch_context(question)

        # 4. Emit references before the answer starts
        ref_preview = [
            {"id": r.id, "content": r.content[:200], "score": round(r.score, 4)}
            for r in refs[:3]
        ]
        yield QnAChunk(
            kind=QnAChunkKind.REFERENCE,
            metadata={"refs": ref_preview, "total": len(refs)},
        )

        # 5. Load prior history (last N turns, app_id-gated)
        history = await self._conv_store.history(
            self._context.conversation_id,
            app_id=self._context.app_id,
            limit=self._router._tiers.get(str(self._context.tier)) and 4 or 4,
        )

        # 6. Assemble the prompt
        context_text = self._build_context_prompt(refs, graph_ctx)
        messages = self._assemble_messages(history, question, context_text)

        # 7. Thinking status
        if self._should_parallel_think(role):
            yield QnAChunk(
                kind=QnAChunkKind.THOUGHT,
                text="Synthesising from 3 thinking models…",
            )
        else:
            yield QnAChunk(kind=QnAChunkKind.THOUGHT, text="Thinking…")

        # 8. Stream answer
        answer_parts: list[str] = []
        final_usage: dict | None = None

        try:
            if self._should_parallel_think(role):
                async for chunk in self._parallel_thinking_stream(messages):
                    if chunk.kind == QnAChunkKind.ANSWER:
                        answer_parts.append(chunk.text)
                    if "usage" in chunk.metadata:
                        final_usage = chunk.metadata["usage"]
                    yield chunk
            else:
                model = self._router.model_for(self._context.tier, role)
                async for llm_chunk in self._llm.stream(messages, model, temperature=0.2):
                    if llm_chunk.delta:
                        answer_parts.append(llm_chunk.delta)
                        yield QnAChunk(
                            kind=QnAChunkKind.ANSWER,
                            text=llm_chunk.delta,
                            metadata={"is_cached": llm_chunk.is_cached},
                        )
                    if llm_chunk.finish_reason and llm_chunk.usage:
                        final_usage = llm_chunk.usage.model_dump()

        except Exception as exc:
            logger.error("qna_agent.answer.error", error=str(exc), exam_id=self._exam_id)
            yield QnAChunk(kind=QnAChunkKind.ERROR, text=str(exc))
            return

        # 9. Persist the full exchange
        full_answer = "".join(answer_parts)
        await self._conv_store.append(self._context.conversation_id, "user", question)
        await self._conv_store.append(self._context.conversation_id, "assistant", full_answer)

        logger.info(
            "qna_agent.answer.done",
            exam_id=self._exam_id,
            conv_id=self._context.conversation_id,
            answer_tokens=len(answer_parts),
        )

        # 10. Done marker
        yield QnAChunk(
            kind=QnAChunkKind.DONE,
            metadata={
                "conversation_id": self._context.conversation_id,
                "usage": final_usage or {},
            },
        )

    # ------------------------------------------------------------------
    # Counter-question flow — Day 19
    # ------------------------------------------------------------------

    async def counter_answer(
        self,
        req: CounterRequest,
    ) -> AsyncGenerator[QnAChunk, None]:
        """Stream an answer to a follow-up question with smart re-fetch heuristic.

        Short follow-ups (≤60 chars, no topic-shift markers) skip re-fetch
        and rely on stored history — ~3 s vs ~6 s for a full retrieval round.
        Topic-shifting follow-ups re-fetch with LOCAL mode (cheaper than HYBRID).

        The agent's bound ``app_id`` (from ``self._context``) guards all reads —
        a forged ``conversation_id`` from another tenant returns empty history.
        """
        history = await self._conv_store.history(
            req.conversation_id,
            app_id=self._context.app_id,
            limit=6,
        )

        should_refetch = self._needs_refetch(req.follow_up)

        if should_refetch:
            yield QnAChunk(kind=QnAChunkKind.THOUGHT, text="Loading references for follow-up…")
            refs, graph_ctx = await self._fetch_context_for_counter(req.follow_up)
            ref_preview = [
                {"id": r.id, "content": r.content[:200], "score": round(r.score, 4)}
                for r in refs[:3]
            ]
            yield QnAChunk(
                kind=QnAChunkKind.REFERENCE,
                metadata={"refs": ref_preview, "total": len(refs)},
            )
            extra = self._build_context_prompt(refs, graph_ctx)
            user_content = f"{req.follow_up}\n\nAdditional context:\n{extra}"
        else:
            yield QnAChunk(kind=QnAChunkKind.THOUGHT, text="Thinking…")
            user_content = req.follow_up

        messages = self._assemble_messages(history, user_content, context_text="")

        answer_parts: list[str] = []
        final_usage: dict | None = None

        try:
            model = self._router.model_for(self._context.tier, "conversation")
            async for llm_chunk in self._llm.stream(messages, model, temperature=0.2):
                if llm_chunk.delta:
                    answer_parts.append(llm_chunk.delta)
                    yield QnAChunk(
                        kind=QnAChunkKind.ANSWER,
                        text=llm_chunk.delta,
                        metadata={"is_cached": llm_chunk.is_cached},
                    )
                if llm_chunk.finish_reason and llm_chunk.usage:
                    final_usage = llm_chunk.usage.model_dump()
        except Exception as exc:
            logger.error("qna_agent.counter_answer.error", error=str(exc))
            yield QnAChunk(kind=QnAChunkKind.ERROR, text=str(exc))
            return

        full_answer = "".join(answer_parts)
        await self._conv_store.append(req.conversation_id, "user", req.follow_up)
        await self._conv_store.append(req.conversation_id, "assistant", full_answer)

        yield QnAChunk(
            kind=QnAChunkKind.DONE,
            metadata={
                "conversation_id": req.conversation_id,
                "usage": final_usage or {},
            },
        )

    # ------------------------------------------------------------------
    # Context retrieval helpers
    # ------------------------------------------------------------------

    def _should_use_graph(self) -> bool:
        """True for Testing / Regular / Premium; False for Free."""
        return self._context.tier in _GRAPH_TIERS

    def _should_parallel_think(self, role: str) -> bool:
        """True only for Premium tier when the role is 'thinking'."""
        return self._context.tier == Tier.PREMIUM and role == "thinking"

    async def _fetch_context(
        self,
        question: str,
    ) -> tuple[list[Reference], CombinedContext | None]:
        """Fetch vector refs and (tier-gated) graph context in parallel."""
        if self._should_use_graph():
            refs, graph_ctx = await asyncio.gather(
                self._ref_finder.find(self._exam_id, question, top_k=5, max_tokens=3000),
                self._hybrid.retrieve(
                    question, QueryParam(mode=QueryMode.HYBRID, top_k=10)
                ),
            )
            return refs, graph_ctx
        else:
            refs = await self._ref_finder.find(
                self._exam_id, question, top_k=5, max_tokens=3000
            )
            return refs, None

    async def _fetch_context_for_counter(
        self,
        follow_up: str,
    ) -> tuple[list[Reference], CombinedContext | None]:
        """Lighter retrieval for counter-questions — LOCAL mode, smaller top_k."""
        if self._should_use_graph():
            refs, graph_ctx = await asyncio.gather(
                self._ref_finder.find(self._exam_id, follow_up, top_k=3, max_tokens=2000),
                self._hybrid.retrieve(
                    follow_up, QueryParam(mode=QueryMode.LOCAL, top_k=5)
                ),
            )
            return refs, graph_ctx
        else:
            refs = await self._ref_finder.find(
                self._exam_id, follow_up, top_k=3, max_tokens=2000
            )
            return refs, None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_context_prompt(
        self,
        refs: list[Reference],
        graph_ctx: CombinedContext | None,
    ) -> str:
        """Format retrieval results into a token-budgeted context string."""
        parts: list[str] = []

        if refs:
            parts.append("## Book References")
            for i, ref in enumerate(refs, 1):
                parts.append(f"[{i}] {ref.content}")

        if graph_ctx:
            graph_str = graph_ctx.build_context_string()
            if graph_str.strip():
                parts.append("## Graph Context")
                parts.append(graph_str)

        result = "\n\n".join(parts)
        if not result:
            return ""

        # Token-budget enforcement: hard-truncate to _MAX_CONTEXT_TOKENS
        tokens = self._enc.encode(result)
        if len(tokens) > _MAX_CONTEXT_TOKENS:
            result = self._enc.decode(tokens[:_MAX_CONTEXT_TOKENS])

        return result

    def _assemble_messages(
        self,
        history: list[LlmMessage],
        question: str,
        context_text: str,
    ) -> list[LlmMessage]:
        """Build the final message list: system + history + user (+ context)."""
        messages: list[LlmMessage] = [
            LlmMessage(role="system", content=self._system_prompt)
        ]
        messages.extend(history)
        if context_text:
            user_content = f"Question: {question}\n\nContext:\n{context_text}"
        else:
            user_content = question
        messages.append(LlmMessage(role="user", content=user_content))
        return messages

    # ------------------------------------------------------------------
    # Counter-question heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_refetch(follow_up: str) -> bool:
        """True when the follow-up appears to introduce a new topic.

        Heuristic: short follow-ups (≤60 chars with no shift markers) are
        treated as continuations of the previous turn; longer ones or those
        containing shift markers get a fresh retrieval pass.
        """
        text_lower = follow_up.lower()
        has_shift_marker = any(m in text_lower for m in _TOPIC_SHIFT_MARKERS)
        return len(follow_up) > _SHORT_FOLLOWUP_THRESHOLD or has_shift_marker

    # ------------------------------------------------------------------
    # Premium parallel-thinking fan-out
    # ------------------------------------------------------------------

    async def _parallel_thinking_stream(
        self,
        messages: list[LlmMessage],
    ) -> AsyncGenerator[QnAChunk, None]:
        """Fan out to N thinking models, collect outputs, synthesise.

        Uses asyncio.gather to run all models in parallel so total latency
        equals the slowest model, not their sum.
        """
        models = self._router.parallel_thinking_models(self._context.tier)
        if not models:
            # Fallback to single thinking model
            model = self._router.model_for(self._context.tier, "thinking")
            models = [model]

        async def _collect(model: str) -> tuple[str, dict | None]:
            parts: list[str] = []
            usage: dict | None = None
            try:
                async for chunk in self._llm.stream(messages, model, temperature=0.2):
                    if chunk.delta:
                        parts.append(chunk.delta)
                    if chunk.finish_reason and chunk.usage:
                        usage = chunk.usage.model_dump()
            except Exception as exc:
                logger.warning("parallel_thinking.model_failed", model=model, error=str(exc))
            return "".join(parts), usage

        results: list[tuple[str, dict | None]] = await asyncio.gather(
            *[_collect(m) for m in models]
        )

        valid = [(text, u) for text, u in results if text]
        if not valid:
            yield QnAChunk(
                kind=QnAChunkKind.ERROR, text="All parallel thinking models failed."
            )
            return

        if len(valid) == 1:
            text, usage = valid[0]
            yield QnAChunk(
                kind=QnAChunkKind.ANSWER,
                text=text,
                metadata={"usage": usage or {}, "source": "single_thinking"},
            )
            return

        # Synthesise with the cheapest model
        synthesis_msgs = self._build_synthesis_messages(valid)
        synth_model = self._router.model_for(Tier.FREE, "conversation")
        synth_parts: list[str] = []
        synth_usage: dict | None = None

        async for chunk in self._llm.stream(synthesis_msgs, synth_model, temperature=0.0):
            if chunk.delta:
                synth_parts.append(chunk.delta)
                yield QnAChunk(
                    kind=QnAChunkKind.ANSWER,
                    text=chunk.delta,
                    metadata={"is_cached": chunk.is_cached, "source": "synthesised"},
                )
            if chunk.finish_reason and chunk.usage:
                synth_usage = chunk.usage.model_dump()

        if synth_usage:
            yield QnAChunk(kind=QnAChunkKind.DONE, metadata={"usage": synth_usage})

    @staticmethod
    def _build_synthesis_messages(
        results: list[tuple[str, dict | None]],
    ) -> list[LlmMessage]:
        """Build the synthesis prompt that merges N thinking-model outputs."""
        body = "\n\n".join(
            f"=== Response {i} ===\n{text}" for i, (text, _) in enumerate(results, 1)
        )
        return [
            LlmMessage(
                role="system",
                content=(
                    "You are an expert synthesizer. Given multiple AI responses "
                    "to the same question, write a single, unified best answer "
                    "that combines the most accurate and insightful points from each."
                ),
            ),
            LlmMessage(
                role="user",
                content=f"{body}\n\n=== Synthesized Answer ===",
            ),
        ]
