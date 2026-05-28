"""Day 2 core-model tests — the vocabulary contract for Phases 3-5.

Every assertion below is from the Phase 0 plan:

    * Every model roundtrips JSON cleanly.
    * EntityRelation.id_from_names is order-invariant.
    * EntityNode.historical_entity_types is preserved across roundtrip.
    * Each of the five multi-modal segments roundtrips with its distinct fields.
    * UserTier has exactly four members.
    * CombinedContext.build_context_string produces the labelled CSV sections.
"""

from __future__ import annotations

import pytest

from autogen.models import (
    AgentContext,
    AppId,
    BookSegment,
    CombinedContext,
    ConversationRuntimeContext,
    EntityNode,
    EntityRelation,
    FullDoc,
    ImageSegment,
    LlmChunk,
    LlmMessage,
    LlmUsage,
    PdfSegment,
    QnAAgent,
    QueryMode,
    QueryParam,
    QuestionSegment,
    Reference,
    TextChunk,
    Tier,
    UserTier,
    WebSegment,
)

# ---------------------------------------------------------------------------
# JSON roundtrips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        FullDoc(id="doc-1", content="hello", source="src", app_id=AppId("neetpg")),
        TextChunk(
            id="chunk-1",
            content="hi",
            full_doc_id="doc-1",
            order=0,
            tokens_count=2,
            keywords=["greeting"],
            embedding=[0.1, 0.2],
            app_id=AppId("neetpg"),
        ),
        EntityNode(
            id="ent-(aspirin)",
            entity_name="Aspirin",
            entity_type="DRUG",
            description="NSAID",
            descriptions=["NSAID-pass1", "Antiplatelet-pass2"],
            historical_entity_types=["DRUG", "MEDICATION"],
            source_ids=["chunk-1", "chunk-2"],
            rank=3,
            segment_content="Aspirin inhibits COX-1.",
            embedding=[0.1] * 4,
            app_id=AppId("neetpg"),
        ),
        EntityRelation(
            id=EntityRelation.id_from_names("Aspirin", "COX-1"),
            source_id="ent-(aspirin)",
            target_id="ent-(cox-1)",
            source_name="Aspirin",
            target_name="COX-1",
            description="inhibits",
            descriptions=["inhibits"],
            keywords=["inhibition"],
            strength=0.9,
            source_ids=["chunk-1"],
            embedding=[0.1] * 4,
            app_id=AppId("neetpg"),
        ),
        BookSegment(
            id="bk-1",
            content="ch1 body",
            title="Pharm",
            chapter="1",
            page_number=12,
            app_id=AppId("neetpg"),
        ),
        PdfSegment(id="pdf-1", content="page 1", filename="x.pdf", page_number=1, app_id=AppId("neetpg")),
        WebSegment(id="web-1", content="page", url="https://x", title="t", app_id=AppId("neetpg")),
        ImageSegment(
            id="img-1",
            content="caption + ocr",
            caption="cat",
            ocr_text="meow",
            image_url="https://img",
            app_id=AppId("neetpg"),
        ),
        QuestionSegment(
            id="q-1",
            content="Q text",
            question_id="qid-1",
            question_text="What is X?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation="Because A.",
            app_id=AppId("neetpg"),
        ),
        QueryParam(mode=QueryMode.HYBRID, top_k=5),
        LlmMessage(role="user", content="hi"),
        LlmUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, total_cost=0.001),
        LlmChunk(delta="hi", finish_reason="stop", usage=LlmUsage(total_tokens=2)),
        Reference(id="r-1", content="passage", score=0.9, metadata={"app_id": "neetpg"}),
        AgentContext(user_id="u1", app_id=AppId("neetpg"), tier=Tier.PREMIUM),
    ],
    ids=lambda m: type(m).__name__,
)
def test_model_roundtrips_json(model) -> None:
    dumped = model.model_dump_json()
    reloaded = type(model).model_validate_json(dumped)
    # Pydantic v2 .model_dump() comparison is the canonical equality check
    assert reloaded.model_dump() == model.model_dump()


# ---------------------------------------------------------------------------
# EntityRelation.id_from_names is order-invariant
# ---------------------------------------------------------------------------


class TestEntityRelationId:
    def test_order_invariant(self) -> None:
        assert EntityRelation.id_from_names("Aspirin", "COX-1") == EntityRelation.id_from_names(
            "COX-1", "Aspirin"
        )

    def test_case_and_whitespace_normalised(self) -> None:
        assert EntityRelation.id_from_names("  Aspirin ", "cox-1") == EntityRelation.id_from_names(
            "ASPIRIN", "COX-1"
        )

    def test_format(self) -> None:
        rid = EntityRelation.id_from_names("Aspirin", "COX-1")
        # rel-(a)-(b) where a,b are lowercased and alphabetical
        assert rid.startswith("rel-(")
        assert rid.endswith(")")


# ---------------------------------------------------------------------------
# Entity audit-trail preservation (historical_entity_types + descriptions)
# ---------------------------------------------------------------------------


def test_entity_node_historical_types_preserved() -> None:
    ent = EntityNode(
        id="ent-(aspirin)",
        entity_name="Aspirin",
        entity_type="DRUG",
        historical_entity_types=["DRUG", "MEDICATION"],
        descriptions=["d1", "d2"],
        source_ids=["c1", "c2", "c3"],
        app_id=AppId("neetpg"),
    )
    reloaded = EntityNode.model_validate_json(ent.model_dump_json())
    assert reloaded.historical_entity_types == ["DRUG", "MEDICATION"]
    assert reloaded.descriptions == ["d1", "d2"]
    assert reloaded.source_ids == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Tier / UserTier
# ---------------------------------------------------------------------------


class TestTier:
    def test_user_tier_is_alias_of_tier(self) -> None:
        assert UserTier is Tier

    def test_tier_has_exactly_four_members(self) -> None:
        assert {t.value for t in Tier} == {"Free", "Testing", "Regular", "Premium"}


# ---------------------------------------------------------------------------
# ConversationRuntimeContext alias + new fields
# ---------------------------------------------------------------------------


class TestConversationRuntimeContext:
    def test_alias_of_agent_context(self) -> None:
        assert ConversationRuntimeContext is AgentContext

    def test_has_tier_and_metadata(self) -> None:
        ctx = ConversationRuntimeContext(
            user_id="u1",
            app_id=AppId("neetpg"),
            tier=Tier.REGULAR,
            metadata={"corr": "abc-123"},
        )
        assert ctx.tier == Tier.REGULAR
        assert ctx.metadata == {"corr": "abc-123"}
        assert ctx.session_id == ctx.conversation_id  # deprecated alias still works


# ---------------------------------------------------------------------------
# CombinedContext.build_context_string
# ---------------------------------------------------------------------------


class TestCombinedContextString:
    def test_empty_produces_three_labelled_sections(self) -> None:
        s = CombinedContext().build_context_string()
        assert "-----Entities-----" in s
        assert "-----Relationships-----" in s
        assert "-----Sources-----" in s

    def test_sections_appear_in_order(self) -> None:
        s = CombinedContext().build_context_string()
        ei = s.index("-----Entities-----")
        ri = s.index("-----Relationships-----")
        si = s.index("-----Sources-----")
        assert ei < ri < si

    def test_rows_emit_csv(self) -> None:
        ctx = CombinedContext(
            entities=[
                EntityNode(
                    id="ent-(aspirin)",
                    entity_name="Aspirin",
                    entity_type="DRUG",
                    description="NSAID",
                    rank=2,
                    app_id=AppId("neetpg"),
                )
            ],
            relationships=[
                EntityRelation(
                    id=EntityRelation.id_from_names("Aspirin", "COX-1"),
                    source_name="Aspirin",
                    target_name="COX-1",
                    description="inhibits",
                    keywords=["inhibition"],
                    strength=0.9,
                    app_id=AppId("neetpg"),
                )
            ],
            sources=[
                TextChunk(
                    id="chunk-1",
                    content="passage body",
                    full_doc_id="doc-1",
                    order=0,
                    app_id=AppId("neetpg"),
                )
            ],
        )
        s = ctx.build_context_string()
        assert "ent-(aspirin),Aspirin,DRUG,NSAID,2" in s
        assert "Aspirin,COX-1,inhibits,inhibition,0.9000" in s
        assert "chunk-1,doc-1,0,passage body" in s

    def test_csv_quoting_for_commas_and_quotes(self) -> None:
        ctx = CombinedContext(
            entities=[
                EntityNode(
                    id="ent-1",
                    entity_name="A, B",
                    entity_type='C"D',
                    description="x",
                    app_id=AppId("neetpg"),
                )
            ]
        )
        s = ctx.build_context_string()
        # A, B → quoted with the comma escaped by quoting
        assert '"A, B"' in s
        # C"D → quoted with the inner " doubled (RFC-4180)
        assert '"C""D"' in s


# ---------------------------------------------------------------------------
# QueryParam defaults match the plan
# ---------------------------------------------------------------------------


def test_query_param_defaults() -> None:
    p = QueryParam()
    assert p.mode == QueryMode.HYBRID
    assert p.top_k == 10
    assert p.max_tokens_for_context == 4000
    assert p.only_need_context is False


# ---------------------------------------------------------------------------
# QnAAgent is constructable with the new context
# ---------------------------------------------------------------------------


def test_qna_agent_constructable() -> None:
    ctx = AgentContext(user_id="u1", app_id=AppId("neetpg"))
    agent = QnAAgent(context=ctx)
    assert agent.context.user_id == "u1"
    assert agent.agent_id  # uuid hex string
