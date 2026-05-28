"""Tests for Neo4jGraphStore — Phase 3 Day 14.

All Neo4j driver calls are mocked via AsyncMock so no live Neo4j instance
is required. Tests verify MERGE Cypher patterns, namespace scoping,
merge semantics, and degree queries.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from autogen.models.storage import EntityNode, EntityRelation
from autogen.storage.neo4j_graph import Neo4jGraphStore, Neo4jGraphStoreFactory


# ---------------------------------------------------------------------------
# Helpers — build a fake driver / session
# ---------------------------------------------------------------------------


def _make_result(records: list[dict]) -> MagicMock:
    """Return a mock neo4j Result whose .data() returns records."""
    result = MagicMock()
    result.data = AsyncMock(return_value=records)
    return result


def _make_session(result: MagicMock | None = None) -> MagicMock:
    """Return a mock session context manager whose .run() returns result."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.run = AsyncMock(return_value=result or _make_result([]))
    return session


def _make_driver(session: MagicMock) -> MagicMock:
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    return driver


def _store(namespace: str = "neetpg") -> tuple[Neo4jGraphStore, MagicMock, MagicMock]:
    session = _make_session()
    driver = _make_driver(session)
    store = Neo4jGraphStore(driver=driver, namespace=namespace)
    return store, driver, session


def _node(name: str = "Aspirin", etype: str = "Drug") -> EntityNode:
    return EntityNode(
        id=f"ent-({name.lower()})",
        entity_name=name,
        entity_type=etype,
        description=f"{name} is a {etype}.",
        source_ids=["chunk-001"],
        app_id="neetpg",
    )


def _edge(src: str = "Aspirin", tgt: str = "COX-1") -> EntityRelation:
    return EntityRelation(
        id=EntityRelation.id_from_names(src, tgt),
        source_id=f"ent-({src.lower()})",
        target_id=f"ent-({tgt.lower()})",
        source_name=src,
        target_name=tgt,
        description=f"{src} relates to {tgt}.",
        keywords=["inhibition"],
        strength=0.8,
        source_ids=["chunk-001"],
        app_id="neetpg",
    )


# ---------------------------------------------------------------------------
# Namespace property
# ---------------------------------------------------------------------------


class TestNamespace:
    def test_namespace_matches_constructor(self):
        store, _, _ = _store("my_namespace")
        assert store.namespace == "my_namespace"


# ---------------------------------------------------------------------------
# ensure_constraints
# ---------------------------------------------------------------------------


class TestEnsureConstraints:
    @pytest.mark.asyncio
    async def test_runs_create_constraint_cypher(self):
        store, _, session = _store()
        await store.ensure_constraints()
        assert session.run.called
        cypher = session.run.call_args[0][0]
        assert "CREATE CONSTRAINT" in cypher or "IF NOT EXISTS" in cypher


# ---------------------------------------------------------------------------
# upsert_node
# ---------------------------------------------------------------------------


class TestUpsertNode:
    @pytest.mark.asyncio
    async def test_upsert_node_calls_merge(self):
        store, _, session = _store()
        await store.upsert_node(_node())
        assert session.run.called
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher

    @pytest.mark.asyncio
    async def test_upsert_node_scopes_to_namespace(self):
        store, _, session = _store(namespace="mds")
        await store.upsert_node(_node())
        kwargs = session.run.call_args[1]
        assert kwargs.get("namespace") == "mds"

    @pytest.mark.asyncio
    async def test_upsert_nodes_batch(self):
        store, _, session = _store()
        nodes = [_node("Aspirin"), _node("Ibuprofen")]
        await store.upsert_nodes(nodes)
        assert session.run.call_count == len(nodes)


# ---------------------------------------------------------------------------
# upsert_edge
# ---------------------------------------------------------------------------


class TestUpsertEdge:
    @pytest.mark.asyncio
    async def test_upsert_edge_calls_merge(self):
        store, _, session = _store()
        await store.upsert_edge(_edge())
        cypher = session.run.call_args[0][0]
        assert "MERGE" in cypher

    @pytest.mark.asyncio
    async def test_upsert_edge_scopes_to_namespace(self):
        store, _, session = _store(namespace="neetug")
        await store.upsert_edge(_edge())
        kwargs = session.run.call_args[1]
        assert kwargs.get("namespace") == "neetug"

    @pytest.mark.asyncio
    async def test_upsert_edges_batch(self):
        store, _, session = _store()
        edges = [_edge("Aspirin", "COX-1"), _edge("Ibuprofen", "COX-2")]
        await store.upsert_edges(edges)
        assert session.run.call_count == len(edges)


# ---------------------------------------------------------------------------
# node_degree
# ---------------------------------------------------------------------------


class TestNodeDegree:
    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty_dict(self):
        store, _, _ = _store()
        result = await store.node_degree([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_degree_per_entity(self):
        records = [
            {"entity_id": "ent-(aspirin)", "degree": 3},
            {"entity_id": "ent-(cox-1)", "degree": 2},
        ]
        session = _make_session(_make_result(records))
        driver = _make_driver(session)
        store = Neo4jGraphStore(driver=driver, namespace="neetpg")

        result = await store.node_degree(["ent-(aspirin)", "ent-(cox-1)"])
        assert result == {"ent-(aspirin)": 3, "ent-(cox-1)": 2}

    @pytest.mark.asyncio
    async def test_zero_degree_entities_omitted(self):
        records = [{"entity_id": "ent-(aspirin)", "degree": 0}]
        session = _make_session(_make_result(records))
        driver = _make_driver(session)
        store = Neo4jGraphStore(driver=driver, namespace="neetpg")

        result = await store.node_degree(["ent-(aspirin)"])
        assert result == {}


# ---------------------------------------------------------------------------
# get_nodes
# ---------------------------------------------------------------------------


class TestGetNodes:
    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty(self):
        store, _, _ = _store()
        result = await store.get_nodes([])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_entity_nodes(self):
        records = [
            {
                "e": {
                    "id": "ent-(aspirin)",
                    "entity_name": "Aspirin",
                    "entity_type": "Drug",
                    "description": "NSAID",
                    "descriptions": ["NSAID"],
                    "historical_entity_types": ["Drug"],
                    "source_ids": ["chunk-001"],
                    "rank": 0,
                    "app_id": "neetpg",
                }
            }
        ]
        session = _make_session(_make_result(records))
        driver = _make_driver(session)
        store = Neo4jGraphStore(driver=driver, namespace="neetpg")

        nodes = await store.get_nodes(["ent-(aspirin)"])
        assert len(nodes) == 1
        assert nodes[0].entity_name == "Aspirin"
        assert nodes[0].entity_type == "Drug"


# ---------------------------------------------------------------------------
# get_relations
# ---------------------------------------------------------------------------


class TestGetRelations:
    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty(self):
        store, _, _ = _store()
        result = await store.get_relations([])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_entity_relations(self):
        records = [
            {
                "r": {
                    "id": "rel-(aspirin)-(cox-1)",
                    "description": "inhibits",
                    "descriptions": ["inhibits"],
                    "keywords": ["inhibition"],
                    "strength": 0.8,
                    "source_ids": ["chunk-001"],
                },
                "source_id": "ent-(aspirin)",
                "source_name": "Aspirin",
                "target_id": "ent-(cox-1)",
                "target_name": "COX-1",
            }
        ]
        session = _make_session(_make_result(records))
        driver = _make_driver(session)
        store = Neo4jGraphStore(driver=driver, namespace="neetpg")

        rels = await store.get_relations(["rel-(aspirin)-(cox-1)"])
        assert len(rels) == 1
        assert rels[0].source_name == "Aspirin"
        assert rels[0].target_name == "COX-1"


# ---------------------------------------------------------------------------
# Neo4jGraphStoreFactory
# ---------------------------------------------------------------------------


class TestNeo4jGraphStoreFactory:
    def test_create_returns_scoped_store(self):
        with pytest.MonkeyPatch().context() as mp:
            # Patch AsyncGraphDatabase.driver to avoid real connection
            mock_driver = MagicMock()
            mock_driver.close = AsyncMock()
            mock_driver.verify_connectivity = AsyncMock()

            import autogen.storage.neo4j_graph as neo4j_mod
            original = neo4j_mod.AsyncGraphDatabase

            class FakeGDB:
                @staticmethod
                def driver(uri, auth):
                    return mock_driver

            mp.setattr(neo4j_mod, "AsyncGraphDatabase", FakeGDB)

            factory = Neo4jGraphStoreFactory(
                uri="bolt://localhost:7687", user="neo4j", password="pw"
            )
            store = factory.create("my_app")
            assert store.namespace == "my_app"
            assert isinstance(store, Neo4jGraphStore)
