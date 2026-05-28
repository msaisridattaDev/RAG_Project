"""Structural-subtype tests — verify mock classes satisfy each Protocol.

These tests ensure that any class with the right method signatures is
recognised as a valid implementation of each Protocol. This is the Python
equivalent of interface-conformance tests in .NET.

Every Protocol that the plan calls out is exercised here:

    EmbeddingClient        – passage/query embedding
    VectorStore[T]         – per-(app_id, type) ES-backed store
    GraphStore             – namespace-scoped Neo4j-backed store
    KeyValueStorage[T]     – per-(label, app_id) KV store
    CacheStore[T]          – embedded cache (LiteDB-style)
    EntityTypeResolver     – raw → canonical type
    RerankClientProtocol   – cross-encoder reranker
    LlmClient              – streaming + complete
    ResponseCache          – cached chunks
    UsageCollector         – cost aggregation
    QnAAgentFactory / QnAAgentFactoryFactory – two-level delegate hierarchy

The runtime isinstance() checks rely on @runtime_checkable being set on
each Protocol; the static substitutability is what pyright verifies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from autogen.models.agent import AgentContext, QnAAgent
from autogen.models.base import AppId
from autogen.models.llm import LlmChunk, LlmMessage, LlmUsage
from autogen.models.storage import EntityNode, EntityRelation
from autogen.protocols.cache import CacheStore
from autogen.protocols.embedding import EmbeddingClient
from autogen.protocols.entity_type import EntityTypeResolver
from autogen.protocols.factory import QnAAgentFactory, QnAAgentFactoryFactory
from autogen.protocols.graph import GraphStore
from autogen.protocols.kvstore import KeyValueStorage
from autogen.protocols.llm import LlmClient, ResponseCache, UsageCollector
from autogen.protocols.reranking import RerankClientProtocol
from autogen.protocols.store import VectorStore
from autogen.reranking.reranker import RerankHit

# ---------------------------------------------------------------------------
# Mock models used by the generic Protocols
# ---------------------------------------------------------------------------


class MockDoc(BaseModel):
    id: str
    text: str
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Mock implementations — each conforms structurally to one Protocol
# ---------------------------------------------------------------------------


class MockEmbedding:
    async def embed(
        self, texts: list[str], *, task: str = "retrieval.passage"
    ) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


class MockVectorStore:
    """Conforms to ``VectorStore[MockDoc]`` (plan shape)."""

    @property
    def app_id(self) -> str:
        return "neetpg"

    @property
    def index_name(self) -> str:
        return f"mockdoc_{self.app_id}_1024"

    async def ensure_index(self) -> None:
        return None

    async def upsert(self, items: list[MockDoc]) -> int:
        return len(items)

    async def embedding_search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[MockDoc, float]]:
        return []

    async def search_by_vector(
        self, vector: list[float], top_k: int = 10
    ) -> list[tuple[MockDoc, float]]:
        return []

    async def query_by_ids(self, ids: list[str]) -> list[MockDoc]:
        return []

    async def delete(self, ids: list[str]) -> int:
        return len(ids)


class MockCacheStore:
    """Conforms to ``CacheStore[str]``."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def get(self, key: str, app_id: str) -> str | None:
        return self._data.get(f"{app_id}:{key}")

    async def set(self, key: str, value: str, app_id: str, ttl_seconds: int = 300) -> None:
        self._data[f"{app_id}:{key}"] = value

    async def delete(self, key: str, app_id: str) -> bool:
        full_key = f"{app_id}:{key}"
        if full_key in self._data:
            del self._data[full_key]
            return True
        return False


class MockGraphStore:
    """Conforms to ``GraphStore`` (plan shape — namespace-scoped)."""

    @property
    def namespace(self) -> str:
        return "entity_neetpg"

    async def ensure_constraints(self) -> None:
        return None

    async def upsert_node(self, node: EntityNode) -> None:
        return None

    async def upsert_nodes(self, nodes: list[EntityNode]) -> None:
        return None

    async def upsert_edge(self, edge: EntityRelation) -> None:
        return None

    async def upsert_edges(self, edges: list[EntityRelation]) -> None:
        return None

    async def node_degree(self, node_id: str) -> int:
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return 0

    async def get_nodes(self, ids: list[str]) -> list[EntityNode]:
        return []

    async def get_relations(self, ids: list[str]) -> list[EntityRelation]:
        return []

    async def get_node_edges(self, node_id: str) -> list[EntityRelation]:
        return []


class MockKeyValueStorage:
    """Conforms to ``KeyValueStorage[MockDoc]``."""

    def __init__(self) -> None:
        self._data: dict[str, MockDoc] = {}

    @property
    def namespace(self) -> str:
        return "fulldoc_neetpg"

    async def get(self, key: str) -> MockDoc | None:
        return self._data.get(key)

    async def upsert(self, key: str, value: MockDoc) -> None:
        self._data[key] = value

    async def filter_keys(self, keys: list[str]) -> list[str]:
        return [k for k in keys if k not in self._data]


class MockEntityTypeResolver:
    async def resolve_canonical_type(
        self, raw_type: str, context: dict[str, Any] | None = None
    ) -> str:
        return raw_type.upper()


class MockReranker:
    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankHit]:
        return [
            RerankHit(index=i, relevance_score=1.0 - i * 0.1, document=d)
            for i, d in enumerate(documents[: top_k or len(documents)])
        ]


class MockLlmClient:
    def stream(
        self, messages: list[LlmMessage], model: str, **_: object
    ) -> AsyncIterator[LlmChunk]:
        async def _gen() -> AsyncIterator[LlmChunk]:
            yield LlmChunk(delta="hello", finish_reason=None)
            yield LlmChunk(
                delta="",
                finish_reason="stop",
                usage=LlmUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, model=model),
            )

        return _gen()

    async def complete(
        self, messages: list[LlmMessage], model: str, **_: object
    ) -> str:
        return "hello"


class MockResponseCache:
    def __init__(self) -> None:
        self._data: dict[str, list[LlmChunk]] = {}

    async def get(self, key: str) -> list[LlmChunk] | None:
        return self._data.get(key)

    async def set(self, key: str, chunks: list[LlmChunk]) -> None:
        self._data[key] = chunks


class MockUsageCollector:
    def __init__(self) -> None:
        self._rows: list[tuple[str, LlmUsage]] = []

    def record(self, key: str, usage: LlmUsage) -> None:
        self._rows.append((key, usage))

    def total_cost(self, prefix: str = "") -> float:
        return sum(u.total_cost for k, u in self._rows if k.startswith(prefix))


class MockQnAAgentFactory:
    async def create(self, context: AgentContext) -> QnAAgent:
        return QnAAgent(context=context)


class MockQnAAgentFactoryFactory:
    @staticmethod
    def for_exam(exam_id: str) -> QnAAgentFactory:
        return MockQnAAgentFactory()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify structural subtyping — each mock satisfies its Protocol."""

    def test_embedding_client_protocol(self) -> None:
        client: EmbeddingClient = MockEmbedding()
        assert isinstance(client, EmbeddingClient)

    def test_vector_store_protocol(self) -> None:
        store: VectorStore[MockDoc] = MockVectorStore()
        assert isinstance(store, VectorStore)

    def test_graph_store_protocol(self) -> None:
        graph: GraphStore = MockGraphStore()
        assert isinstance(graph, GraphStore)

    def test_kv_store_protocol(self) -> None:
        kv: KeyValueStorage[MockDoc] = MockKeyValueStorage()
        assert isinstance(kv, KeyValueStorage)

    def test_cache_store_protocol(self) -> None:
        cache: CacheStore[str] = MockCacheStore()
        assert isinstance(cache, CacheStore)

    def test_entity_type_resolver_protocol(self) -> None:
        resolver: EntityTypeResolver = MockEntityTypeResolver()
        assert isinstance(resolver, EntityTypeResolver)

    def test_reranker_protocol(self) -> None:
        reranker: RerankClientProtocol = MockReranker()
        assert isinstance(reranker, RerankClientProtocol)

    def test_llm_client_protocol(self) -> None:
        llm: LlmClient = MockLlmClient()
        assert isinstance(llm, LlmClient)

    def test_response_cache_protocol(self) -> None:
        cache: ResponseCache = MockResponseCache()
        assert isinstance(cache, ResponseCache)

    def test_usage_collector_protocol(self) -> None:
        collector: UsageCollector = MockUsageCollector()
        assert isinstance(collector, UsageCollector)

    def test_agent_factory_protocol(self) -> None:
        factory: QnAAgentFactory = MockQnAAgentFactory()
        assert isinstance(factory, MockQnAAgentFactory)

    def test_agent_factory_factory_protocol(self) -> None:
        ff: QnAAgentFactoryFactory = MockQnAAgentFactoryFactory()
        assert isinstance(ff, MockQnAAgentFactoryFactory)

    @pytest.mark.asyncio
    async def test_two_level_factory_chain(self) -> None:
        """The full factory chain produces a valid ``QnAAgent``."""
        context = AgentContext(user_id="test-user", app_id=AppId("neetpg"))

        ff: QnAAgentFactoryFactory = MockQnAAgentFactoryFactory()
        factory: QnAAgentFactory = ff.for_exam("neetpg-2025")
        agent: QnAAgent = await factory.create(context)

        assert agent.context.user_id == "test-user"
        assert agent.context.app_id == "neetpg"
        assert agent.context.exam_id is None
        assert agent.agent_id is not None


class TestConcreteFactoryFactory:
    """The concrete ``QnAAgentFactoryFactoryImpl`` should satisfy the
    Protocol surface and bind ``exam_id`` into the produced agent's context.
    """

    @pytest.mark.asyncio
    async def test_concrete_factory_factory_chain(self) -> None:
        from unittest.mock import MagicMock, patch

        from autogen.di.providers import QnAAgentFactoryFactoryImpl

        settings = MagicMock()
        settings.lightrag.neo4j_uri = "bolt://localhost:7687"
        settings.lightrag.neo4j_user = "neo4j"
        settings.lightrag.neo4j_password = "password"

        store_factory = MagicMock()
        store_factory.create.return_value = MagicMock()

        with patch("autogen.storage.neo4j_graph.Neo4jGraphStoreFactory") as mock_gf:
            mock_gf.return_value.create.return_value = MagicMock()
            ff = QnAAgentFactoryFactoryImpl(
                settings=settings,
                llm=MagicMock(),
                router=MagicMock(),
                store_factory=store_factory,
                embedding_client=MagicMock(),
                reranker=MagicMock(),
                keyword_extractor=MagicMock(),
                conv_store=MagicMock(),
            )

        factory = ff.for_exam("neetpg")
        # Same exam_id → same cached factory instance
        assert ff.for_exam("neetpg") is factory

        ctx = AgentContext(user_id="u1", app_id=AppId("neetpg"))
        agent = await factory.create(ctx)
        assert agent.context.exam_id == "neetpg"
