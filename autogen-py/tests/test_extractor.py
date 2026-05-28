"""Tests for EntityExtractor, KeywordExtractor — Phase 3."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.extraction.extractor import EntityExtractor, ExtractionResponse
from autogen.extraction.keywords import KeywordExtractor, _parse_json
from autogen.extraction.prompts import COMPLETION_DELIMITER, RESPONSE_DELIMITER, TUPLE_DELIMITER
from autogen.models.storage import EntityNode, EntityRelation, TextChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "Aspirin inhibits COX-1 and COX-2 enzymes.",
    chunk_id: str = "chunk-001",
    app_id: str = "test",
) -> TextChunk:
    return TextChunk(id=chunk_id, content=content, app_id=app_id)


def _mock_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


def _delim_response(
    entities: list[tuple[str, str, str]],
    relations: list[tuple[str, str, str, str, str]] | None = None,
    keywords: str = "",
) -> str:
    """Build a delimiter-based extractor response.

    entities: [(name, type, desc), ...]
    relations: [(src, tgt, desc, kw_csv, strength_str), ...]
    """
    parts: list[str] = []

    # Entity lines in first section
    ent_lines = "\n".join(f"{n}{TUPLE_DELIMITER}{t}{TUPLE_DELIMITER}{d}" for n, t, d in entities)
    parts.append(ent_lines)

    # Relation lines (one section per relation)
    for src, tgt, desc, kw, strength in (relations or []):
        rel_line = f"{src}{TUPLE_DELIMITER}{tgt}{TUPLE_DELIMITER}{desc}{TUPLE_DELIMITER}{kw}{TUPLE_DELIMITER}{strength}"
        parts.append(rel_line)

    # Keywords section
    if keywords:
        parts.append(keywords)

    return RESPONSE_DELIMITER.join(parts) + COMPLETION_DELIMITER


_ENTITY_RESPONSE = _delim_response(
    entities=[
        ("Aspirin", "DRUG", "An NSAID."),
        ("COX-1", "PROTEIN", "Cyclooxygenase 1."),
    ],
    relations=[
        ("Aspirin", "COX-1", "Aspirin inhibits COX-1.", "inhibition", "0.8"),
    ],
    keywords="aspirin, COX",
)


# ---------------------------------------------------------------------------
# EntityExtractor — basic extraction
# ---------------------------------------------------------------------------


class TestEntityExtractor:
    @pytest.mark.asyncio
    async def test_extract_returns_nodes_and_relations(self):
        llm = _mock_llm(_ENTITY_RESPONSE)
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=0)
        result: ExtractionResponse = await extractor.extract_from_chunk(_make_chunk())

        assert len(result.nodes) == 2
        assert len(result.relations) == 1

    @pytest.mark.asyncio
    async def test_extracted_node_fields(self):
        llm = _mock_llm(_ENTITY_RESPONSE)
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=0)
        result = await extractor.extract_from_chunk(_make_chunk())

        aspirin = next(n for n in result.nodes if n.entity_name == "Aspirin")
        assert aspirin.entity_type == "DRUG"
        assert aspirin.description == "An NSAID."
        assert "chunk-001" in aspirin.source_ids
        assert aspirin.app_id == "test"

    @pytest.mark.asyncio
    async def test_extracted_relation_fields(self):
        llm = _mock_llm(_ENTITY_RESPONSE)
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=0)
        result = await extractor.extract_from_chunk(_make_chunk())

        rel = result.relations[0]
        assert rel.source_name == "Aspirin"
        assert rel.target_name == "COX-1"
        assert "inhibition" in rel.keywords
        assert 0.0 < rel.strength <= 1.0

    @pytest.mark.asyncio
    async def test_llm_failure_propagates(self):
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=0)
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await extractor.extract_from_chunk(_make_chunk())

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_result(self):
        llm = _mock_llm(COMPLETION_DELIMITER)
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=0)
        result = await extractor.extract_from_chunk(_make_chunk())

        assert result.nodes == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_entity_id_is_stable(self):
        llm = _mock_llm(_ENTITY_RESPONSE)
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=0)
        result_a = await extractor.extract_from_chunk(_make_chunk())
        result_b = await extractor.extract_from_chunk(_make_chunk())

        ids_a = {n.id for n in result_a.nodes}
        ids_b = {n.id for n in result_b.nodes}
        assert ids_a == ids_b

    def test_relation_id_is_order_invariant(self):
        assert EntityRelation.id_from_names("Aspirin", "COX-1") == \
               EntityRelation.id_from_names("COX-1", "Aspirin")

    @pytest.mark.asyncio
    async def test_content_keywords_extracted(self):
        llm = _mock_llm(_ENTITY_RESPONSE)
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=0)
        result = await extractor.extract_from_chunk(_make_chunk())
        assert len(result.content_keywords) > 0


# ---------------------------------------------------------------------------
# EntityExtractor — gleaning
# ---------------------------------------------------------------------------


class TestGleaning:
    @pytest.mark.asyncio
    async def test_gleaning_recovers_missed_entities(self):
        first_response = _delim_response(
            entities=[("Aspirin", "DRUG", "An NSAID.")],
        )
        gleaning_response = _delim_response(
            entities=[("COX-1", "PROTEIN", "cyclooxygenase")],
        )
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=[first_response, gleaning_response])
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=1)
        result = await extractor.extract_from_chunk(_make_chunk())

        names = {n.entity_name for n in result.nodes}
        assert "Aspirin" in names
        assert "COX-1" in names

    @pytest.mark.asyncio
    async def test_gleaning_stops_on_complete_marker(self):
        call_count = 0

        async def _complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _delim_response(entities=[("Aspirin", "DRUG", "NSAID")])
            return COMPLETION_DELIMITER  # signals "nothing missed"

        llm = MagicMock()
        llm.complete = _complete
        extractor = EntityExtractor(llm=llm, model="test-model", max_gleaning=3)
        await extractor.extract_from_chunk(_make_chunk())
        # First call = primary, second call = gleaning returns COMPLETE → stop
        assert call_count == 2


# ---------------------------------------------------------------------------
# KeywordExtractor
# ---------------------------------------------------------------------------


class TestKeywordExtractor:
    @pytest.mark.asyncio
    async def test_extract_returns_local_and_global(self):
        response = json.dumps({
            "local": ["aspirin", "COX-1", "anti-inflammatory"],
            "global": ["pharmacology", "NSAIDs"],
        })
        llm = _mock_llm(response)
        extractor = KeywordExtractor(llm=llm, model="test-model")
        local, global_ = await extractor.extract("What is the MOA of aspirin?")

        assert "aspirin" in local
        assert "pharmacology" in global_

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        extractor = KeywordExtractor(llm=llm, model="test-model")
        local, global_ = await extractor.extract("aspirin inflammation")

        assert len(local) > 0
        assert global_ == []

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back(self):
        llm = _mock_llm("not json")
        extractor = KeywordExtractor(llm=llm, model="test-model")
        local, global_ = await extractor.extract("aspirin COX")
        assert isinstance(local, list)
        assert isinstance(global_, list)


# ---------------------------------------------------------------------------
# JSON parse helper
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_direct_json(self):
        data = _parse_json('{"a": 1}')
        assert data == {"a": 1}

    def test_fenced_json(self):
        text = '```json\n{"a": 2}\n```'
        data = _parse_json(text)
        assert data == {"a": 2}

    def test_json_with_surrounding_prose(self):
        text = 'Here is the result: {"a": 3} — done.'
        data = _parse_json(text)
        assert data == {"a": 3}

    def test_empty_returns_none(self):
        assert _parse_json("") is None

    def test_invalid_json_returns_none(self):
        assert _parse_json("not json at all") is None
