"""Tests for SqlConversationStore — Phase 4 Day 17.

All tests run against SQLite :memory: so they are fast, isolated, and require
no external services.  Cross-tenant isolation is the critical invariant.
"""

from __future__ import annotations

import pytest

from autogen.conversation.store import SqlConversationStore
from autogen.models.agent import AgentContext
from autogen.models.enums import Tier


@pytest.fixture
async def store() -> SqlConversationStore:
    """Fresh in-memory store initialised before each test."""
    s = SqlConversationStore(":memory:")
    await s.init_schema()
    return s


@pytest.fixture
def neetpg_ctx() -> AgentContext:
    return AgentContext(
        conversation_id="conv-neetpg-1",
        user_id="user-1",
        app_id="neetpg",
        tier=Tier.REGULAR,
    )


@pytest.fixture
def mds_ctx() -> AgentContext:
    return AgentContext(
        conversation_id="conv-mds-1",
        user_id="user-1",
        app_id="mds",
        tier=Tier.FREE,
    )


# ---------------------------------------------------------------------------
# get_or_create
# ---------------------------------------------------------------------------


class TestGetOrCreate:
    async def test_creates_new_conversation(self, store, neetpg_ctx):
        row = await store.get_or_create(neetpg_ctx)
        assert row["id"] == "conv-neetpg-1"
        assert row["app_id"] == "neetpg"
        assert row["user_id"] == "user-1"

    async def test_idempotent_on_same_id(self, store, neetpg_ctx):
        row1 = await store.get_or_create(neetpg_ctx)
        row2 = await store.get_or_create(neetpg_ctx)
        assert row1["id"] == row2["id"]

    async def test_same_user_different_exams(self, store, neetpg_ctx, mds_ctx):
        r1 = await store.get_or_create(neetpg_ctx)
        r2 = await store.get_or_create(mds_ctx)
        assert r1["app_id"] == "neetpg"
        assert r2["app_id"] == "mds"


# ---------------------------------------------------------------------------
# append + history
# ---------------------------------------------------------------------------


class TestAppendAndHistory:
    async def test_empty_history_returns_empty_list(self, store, neetpg_ctx):
        await store.get_or_create(neetpg_ctx)
        msgs = await store.history("conv-neetpg-1", app_id="neetpg")
        assert msgs == []

    async def test_history_returns_messages_in_chronological_order(self, store, neetpg_ctx):
        await store.get_or_create(neetpg_ctx)
        await store.append("conv-neetpg-1", "user", "What is aspirin?")
        await store.append("conv-neetpg-1", "assistant", "Aspirin inhibits COX-1/2.")
        await store.append("conv-neetpg-1", "user", "Why?")

        msgs = await store.history("conv-neetpg-1", app_id="neetpg")
        assert len(msgs) == 3
        assert msgs[0].role == "user"
        assert msgs[0].content == "What is aspirin?"
        assert msgs[1].role == "assistant"
        assert msgs[2].role == "user"
        assert msgs[2].content == "Why?"

    async def test_limit_respected(self, store, neetpg_ctx):
        await store.get_or_create(neetpg_ctx)
        for i in range(10):
            await store.append("conv-neetpg-1", "user", f"question {i}")

        msgs = await store.history("conv-neetpg-1", app_id="neetpg", limit=4)
        assert len(msgs) == 4
        # With limit=4 we get the 4 most recent (reversed to chronological)
        assert msgs[-1].content == "question 9"

    # Critical: cross-tenant isolation
    async def test_wrong_app_id_returns_empty(self, store, neetpg_ctx):
        await store.get_or_create(neetpg_ctx)
        await store.append("conv-neetpg-1", "user", "secret neetpg message")

        # Attacker uses the correct conv_id but the wrong app_id
        msgs = await store.history("conv-neetpg-1", app_id="mds")
        assert msgs == [], "Cross-tenant read must return empty list"

    async def test_nonexistent_conv_returns_empty(self, store):
        msgs = await store.history("nonexistent-id", app_id="neetpg")
        assert msgs == []


# ---------------------------------------------------------------------------
# list_by_user
# ---------------------------------------------------------------------------


class TestListByUser:
    async def test_lists_only_matching_app_id(self, store, neetpg_ctx, mds_ctx):
        await store.get_or_create(neetpg_ctx)
        await store.get_or_create(mds_ctx)

        neetpg_convs = await store.list_by_user("user-1", app_id="neetpg")
        assert len(neetpg_convs) == 1
        assert neetpg_convs[0]["app_id"] == "neetpg"

        mds_convs = await store.list_by_user("user-1", app_id="mds")
        assert len(mds_convs) == 1
        assert mds_convs[0]["app_id"] == "mds"

    async def test_limit_respected(self, store):
        for i in range(5):
            ctx = AgentContext(
                conversation_id=f"conv-{i}",
                user_id="user-multi",
                app_id="neetpg",
                tier=Tier.FREE,
            )
            await store.get_or_create(ctx)

        result = await store.list_by_user("user-multi", app_id="neetpg", limit=3)
        assert len(result) == 3

    async def test_different_user_not_included(self, store):
        ctx_a = AgentContext(
            conversation_id="conv-a", user_id="alice", app_id="neetpg", tier=Tier.FREE
        )
        ctx_b = AgentContext(
            conversation_id="conv-b", user_id="bob", app_id="neetpg", tier=Tier.FREE
        )
        await store.get_or_create(ctx_a)
        await store.get_or_create(ctx_b)

        alice_convs = await store.list_by_user("alice", app_id="neetpg")
        assert all(c["user_id"] == "alice" for c in alice_convs)


# ---------------------------------------------------------------------------
# schema idempotency
# ---------------------------------------------------------------------------


class TestSchemaIdempotency:
    async def test_init_schema_twice_does_not_raise(self):
        s = SqlConversationStore(":memory:")
        await s.init_schema()
        await s.init_schema()  # second call must be safe
