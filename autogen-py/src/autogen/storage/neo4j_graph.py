"""Neo4j Graph Store — namespace-scoped entity and relation persistence.

Mirrors autogen.net's Neo4jGraphStore (IGraphStorage interface) with
composite uniqueness on (id, namespace) so multiple exam datasets
coexist in one Neo4j instance without cross-tenant leakage.

Every node and edge carries a ``namespace`` property. All MATCH/MERGE
operations include ``{namespace: $ns}`` to scope queries to the
calling app_id. The composite uniqueness constraint is::

    CREATE CONSTRAINT entity_id_namespace IF NOT EXISTS
    FOR (e:Entity) REQUIRE (e.id, e.namespace) IS UNIQUE

Usage::

    factory = Neo4jGraphStoreFactory(uri="bolt://localhost:7687")
    store = factory.create("neetpg")
    await store.ensure_constraints()
    await store.upsert_node(entity_node)
    await store.upsert_edge(entity_relation)
"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase  # pyright: ignore[reportMissingImports]

from autogen.logging.setup import get_logger
from autogen.models.storage import EntityNode, EntityRelation

logger = get_logger("autogen.storage.neo4j_graph")

# ---------------------------------------------------------------------------
# Neo4jGraphStore — the core persistence layer
# ---------------------------------------------------------------------------


class Neo4jGraphStore:
    """Namespace-scoped Neo4j graph persistence.

    All operations are scoped by ``namespace`` (the app_id). Nodes and
    edges are upserted with merge semantics: descriptions append,
    source_ids union, historical_entity_types union, edge strengths
    accumulate.

    Usage:
        store = Neo4jGraphStore(driver, namespace="neetpg")
        await store.ensure_constraints()
        await store.upsert_node(entity)
    """

    def __init__(
        self,
        driver: Any,
        namespace: str,
    ) -> None:
        """Create a namespace-scoped graph store.

        Args:
            driver: A neo4j async driver instance (neo4j.AsyncGraphDatabase.driver()).
            namespace: The app_id / tenant key (e.g., "neetpg", "mds").
        """
        self._driver = driver
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        return self._namespace

    # ------------------------------------------------------------------
    # ensure_constraints — composite uniqueness on (id, namespace)
    # ------------------------------------------------------------------

    async def ensure_constraints(self) -> None:
        """Create the composite uniqueness constraint if it doesn't exist.

        Idempotent — safe to call multiple times. Only the first call
        creates the constraint.

        Constraint: (e:Entity) REQUIRE (e.id, e.namespace) IS UNIQUE
        """
        async with self._driver.session() as session:
            try:
                await session.run(
                    """
                    CREATE CONSTRAINT entity_id_namespace IF NOT EXISTS
                    FOR (e:Entity) REQUIRE (e.id, e.namespace) IS UNIQUE
                    """
                )
                logger.info(
                    "neo4j.constraints.ok",
                    namespace=self._namespace,
                )
            except Exception as exc:
                logger.error(
                    "neo4j.constraints.failed",
                    namespace=self._namespace,
                    error=str(exc),
                )
                raise

    # ------------------------------------------------------------------
    # upsert_node — MERGE entity with additive semantics
    # ------------------------------------------------------------------

    async def upsert_node(self, node: EntityNode) -> None:
        """Upsert a single entity node into the graph.

        MERGE on (id, namespace):
            - ON CREATE: set all fields.
            - ON MATCH: append description (if new), accumulate
              descriptions[], union source_ids, union
              historical_entity_types.

        The ``description`` field is concatenated with newlines. The
        ``descriptions`` list preserves the full audit trail of every
        description ever seen for this entity within the namespace.

        Args:
            node: The EntityNode to upsert. Must have .id, .entity_name,
                .entity_type, .description, .descriptions, .source_ids,
                .historical_entity_types, .app_id populated.
        """
        async with self._driver.session() as session:
            try:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id, namespace: $namespace})
                    ON CREATE SET
                        e.entity_name = $entity_name,
                        e.entity_type = $entity_type,
                        e.description = $description,
                        e.descriptions = $descriptions,
                        e.source_ids = $source_ids,
                        e.historical_entity_types = $historical_entity_types,
                        e.app_id = $app_id,
                        e.rank = $rank
                    ON MATCH SET
                        e.description = CASE
                            WHEN $description IN coalesce(e.descriptions, [])
                            THEN e.description
                            ELSE coalesce(e.description, '') +
                                 CASE WHEN coalesce(e.description, '') = '' THEN ''
                                      ELSE '\n' END
                                 + $description
                        END,
                        e.descriptions = CASE
                            WHEN $description IN coalesce(e.descriptions, [])
                            THEN e.descriptions
                            ELSE coalesce(e.descriptions, []) + [$description]
                        END,
                        e.source_ids = apoc.coll.toSet(
                            coalesce(e.source_ids, []) + $source_ids
                        ),
                        e.historical_entity_types = apoc.coll.toSet(
                            coalesce(e.historical_entity_types, []) + $historical_entity_types
                        ),
                        e.entity_type = CASE
                            WHEN e.entity_type IS NULL OR e.entity_type = ''
                            THEN $entity_type
                            ELSE e.entity_type
                        END
                    """,
                    id=node.id,
                    namespace=self._namespace,
                    entity_name=node.entity_name,
                    entity_type=node.entity_type,
                    description=node.description,
                    descriptions=node.descriptions or [node.description],
                    source_ids=node.source_ids,
                    historical_entity_types=node.historical_entity_types,
                    app_id=node.app_id,
                    rank=node.rank,
                )
                logger.debug(
                    "neo4j.upsert_node",
                    namespace=self._namespace,
                    entity_id=node.id,
                    entity_name=node.entity_name,
                )
            except Exception as exc:
                logger.error(
                    "neo4j.upsert_node.failed",
                    namespace=self._namespace,
                    entity_id=node.id,
                    error=str(exc),
                )
                raise

    async def upsert_nodes(self, nodes: list[EntityNode]) -> None:
        """Upsert multiple entity nodes sequentially.

        Args:
            nodes: The EntityNode objects to upsert.
        """
        for node in nodes:
            await self.upsert_node(node)

    # ------------------------------------------------------------------
    # upsert_edge — MERGE relationship with strength accumulation
    # ------------------------------------------------------------------

    async def upsert_edge(self, edge: EntityRelation) -> None:
        """Upsert a single relationship into the graph.

        The edge is stored as an undirected relationship between two
        Entity nodes. MERGE on source_id, target_id, and namespace
        ensures idempotency. On match, strength accumulates and
        keywords / descriptions union.

        Args:
            edge: The EntityRelation to upsert. Must have .id, .source_id,
                .source_name, .target_id, .target_name, .description,
                .descriptions, .keywords, .strength populated.
        """
        async with self._driver.session() as session:
            try:
                await session.run(
                    """
                    MATCH (src:Entity {id: $source_id, namespace: $namespace})
                    MATCH (tgt:Entity {id: $target_id, namespace: $namespace})
                    MERGE (src)-[r:RELATES_TO {id: $edge_id}]->(tgt)
                    ON CREATE SET
                        r.source_name = $source_name,
                        r.target_name = $target_name,
                        r.description = $description,
                        r.descriptions = $descriptions,
                        r.keywords = $keywords,
                        r.strength = $strength,
                        r.source_ids = $source_ids,
                        r.namespace = $namespace
                    ON MATCH SET
                        r.description = CASE
                            WHEN $description IN coalesce(r.descriptions, [])
                            THEN r.description
                            ELSE coalesce(r.description, '') +
                                 CASE WHEN coalesce(r.description, '') = '' THEN ''
                                      ELSE '\n' END
                                 + $description
                        END,
                        r.descriptions = CASE
                            WHEN $description IN coalesce(r.descriptions, [])
                            THEN r.descriptions
                            ELSE coalesce(r.descriptions, []) + [$description]
                        END,
                        r.keywords = apoc.coll.toSet(
                            coalesce(r.keywords, []) + $keywords
                        ),
                        r.strength = coalesce(r.strength, 0.0) + $strength,
                        r.source_ids = apoc.coll.toSet(
                            coalesce(r.source_ids, []) + $source_ids
                        )
                    """,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    edge_id=edge.id,
                    source_name=edge.source_name,
                    target_name=edge.target_name,
                    description=edge.description,
                    descriptions=edge.descriptions or [edge.description],
                    keywords=edge.keywords,
                    strength=edge.strength,
                    source_ids=edge.source_ids,
                    namespace=self._namespace,
                )
                logger.debug(
                    "neo4j.upsert_edge",
                    namespace=self._namespace,
                    edge_id=edge.id,
                    source=edge.source_name,
                    target=edge.target_name,
                )
            except Exception as exc:
                logger.error(
                    "neo4j.upsert_edge.failed",
                    namespace=self._namespace,
                    edge_id=edge.id,
                    error=str(exc),
                )
                raise

    async def upsert_edges(self, edges: list[EntityRelation]) -> None:
        """Upsert multiple edges sequentially.

        Args:
            edges: The EntityRelation objects to upsert.
        """
        for edge in edges:
            await self.upsert_edge(edge)

    # ------------------------------------------------------------------
    # node_degree — get degrees for a list of entity IDs
    # ------------------------------------------------------------------

    async def node_degree(self, ids: list[str]) -> dict[str, int]:
        """Get the degree (number of relationships) for each entity ID.

        Args:
            ids: A list of entity IDs.

        Returns:
            A dict mapping entity ID → degree. Entities with degree 0
            are omitted.
        """
        if not ids:
            return {}

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {namespace: $namespace})
                WHERE e.id IN $ids
                OPTIONAL MATCH (e)-[r:RELATES_TO]-()
                RETURN e.id AS entity_id, count(r) AS degree
                """,
                ids=ids,
                namespace=self._namespace,
            )
            records = await result.data()
            return {
                record["entity_id"]: record["degree"]
                for record in records
                if record["degree"] > 0
            }

    # ------------------------------------------------------------------
    # edge_degree — get degrees for a list of edge IDs
    # ------------------------------------------------------------------

    async def edge_degree(self, ids: list[str]) -> dict[str, int]:
        """Get the source+target entity degrees for each edge ID.

        This is a proxy for edge importance — an edge whose endpoints
        are well-connected is more central to the knowledge graph.

        Args:
            ids: A list of edge IDs.

        Returns:
            A dict mapping edge ID → (source_degree + target_degree).
            Edges where either endpoint is missing are omitted.
        """
        if not ids:
            return {}

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (src:Entity {namespace: $namespace})-[r:RELATES_TO]->(tgt:Entity {namespace: $namespace})
                WHERE r.id IN $edge_ids
                OPTIONAL MATCH (src)-[sr:RELATES_TO]-()
                OPTIONAL MATCH (tgt)-[tr:RELATES_TO]-()
                RETURN r.id AS edge_id, count(sr) + count(tr) AS total_degree
                """,
                edge_ids=ids,
                namespace=self._namespace,
            )
            records = await result.data()
            return {
                record["edge_id"]: record["total_degree"]
                for record in records
                if record["total_degree"] > 0
            }

    # ------------------------------------------------------------------
    # get_nodes — fetch entities by ID
    # ------------------------------------------------------------------

    async def get_nodes(self, ids: list[str]) -> list[EntityNode]:
        """Fetch entity nodes by their string IDs.

        All queries include ``namespace = $ns`` so cross-tenant lookups
        return empty. Missing IDs are silently omitted.

        Args:
            ids: A list of entity IDs.

        Returns:
            A list of EntityNode objects reconstructed from graph state.
        """
        if not ids:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {namespace: $namespace})
                WHERE e.id IN $ids
                RETURN e
                """,
                ids=ids,
                namespace=self._namespace,
            )
            records = await result.data()
            nodes: list[EntityNode] = []
            for record in records:
                e = record["e"]
                nodes.append(
                    EntityNode(
                        id=e.get("id", ""),
                        entity_name=e.get("entity_name", ""),
                        entity_type=e.get("entity_type", ""),
                        description=e.get("description", ""),
                        descriptions=e.get("descriptions", []),
                        historical_entity_types=e.get("historical_entity_types", []),
                        source_ids=e.get("source_ids", []),
                        rank=e.get("rank", 0),
                        segment_content=e.get("description", ""),
                        app_id=e.get("app_id", self._namespace),
                    )
                )
            return nodes

    # ------------------------------------------------------------------
    # get_relations — fetch edges by ID
    # ------------------------------------------------------------------

    async def get_relations(self, ids: list[str]) -> list[EntityRelation]:
        """Fetch relationships by their string IDs.

        Args:
            ids: A list of edge IDs.

        Returns:
            A list of EntityRelation objects reconstructed from graph state.
        """
        if not ids:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (src:Entity {namespace: $namespace})-[r:RELATES_TO]->(tgt:Entity {namespace: $namespace})
                WHERE r.id IN $ids
                RETURN r, src.id AS source_id, src.entity_name AS source_name,
                       tgt.id AS target_id, tgt.entity_name AS target_name
                """,
                ids=ids,
                namespace=self._namespace,
            )
            records = await result.data()
            edges: list[EntityRelation] = []
            for record in records:
                r = record["r"]
                edges.append(
                    EntityRelation(
                        id=r.get("id", ""),
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        source_name=record["source_name"],
                        target_name=record["target_name"],
                        description=r.get("description", ""),
                        descriptions=r.get("descriptions", []),
                        keywords=r.get("keywords", []),
                        strength=r.get("strength", 1.0),
                        source_ids=r.get("source_ids", []),
                        app_id=self._namespace,
                    )
                )
            return edges

    # ------------------------------------------------------------------
    # get_node_edges — fetch 1-hop neighbors
    # ------------------------------------------------------------------

    async def get_node_edges(self, entity_ids: list[str]) -> list[EntityRelation]:
        """Fetch all relationships for the given entity IDs (1-hop neighbors).

        Returns all RELATES_TO edges where either endpoint is in
        ``entity_ids``, within the scoped namespace.

        Args:
            entity_ids: A list of entity IDs.

        Returns:
            A list of EntityRelation objects for all incident edges.
        """
        if not entity_ids:
            return []

        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (src:Entity {namespace: $namespace})-[r:RELATES_TO]->(tgt:Entity {namespace: $namespace})
                WHERE src.id IN $entity_ids OR tgt.id IN $entity_ids
                RETURN r, src.id AS source_id, src.entity_name AS source_name,
                       tgt.id AS target_id, tgt.entity_name AS target_name
                ORDER BY r.strength DESC
                """,
                entity_ids=entity_ids,
                namespace=self._namespace,
            )
            records = await result.data()
            edges: list[EntityRelation] = []
            seen: set[str] = set()
            for record in records:
                r = record["r"]
                edge_id = r.get("id", "")
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                edges.append(
                    EntityRelation(
                        id=edge_id,
                        source_id=record["source_id"],
                        target_id=record["target_id"],
                        source_name=record["source_name"],
                        target_name=record["target_name"],
                        description=r.get("description", ""),
                        descriptions=r.get("descriptions", []),
                        keywords=r.get("keywords", []),
                        strength=r.get("strength", 1.0),
                        source_ids=r.get("source_ids", []),
                        app_id=self._namespace,
                    )
                )
            return edges

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def delete_namespace(self) -> int:
        """Delete all nodes and edges for this namespace.

        WARNING: Irreversible. Used for dataset teardown/re-ingestion.

        Returns:
            The number of nodes deleted.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {namespace: $namespace})
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
                namespace=self._namespace,
            )
            records = await result.data()
            count = records[0]["deleted"] if records else 0
            logger.info(
                "neo4j.delete_namespace",
                namespace=self._namespace,
                deleted=count,
            )
            return count


# ---------------------------------------------------------------------------
# Neo4jGraphStoreFactory — mirrors IGraphStorageFactory
# ---------------------------------------------------------------------------


class Neo4jGraphStoreFactory:
    """Factory for creating namespace-scoped Neo4jGraphStore instances.

    Mirrors autogen.net's IGraphStorageFactory (LightRag.cs:41)::

        graphStorageFactory.Create("neetpg")  → Neo4jGraphStore(namespace="neetpg")

    Usage::

        factory = Neo4jGraphStoreFactory(uri="bolt://localhost:7687", user="neo4j", password="password")
        neetpg_store = factory.create("neetpg")
        mds_store = factory.create("mds")
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ) -> None:
        """Initialize the factory with Neo4j connection parameters.

        Args:
            uri: Neo4j bolt URI (default: bolt://localhost:7687).
            user: Neo4j username (default: neo4j).
            password: Neo4j password (default: password).
        """
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._uri = uri

    def create(self, namespace: str) -> Neo4jGraphStore:
        """Create a namespace-scoped graph store.

        Args:
            namespace: The app_id / tenant key (e.g., "neetpg").

        Returns:
            A Neo4jGraphStore bound to the given namespace.
        """
        return Neo4jGraphStore(driver=self._driver, namespace=namespace)

    async def close(self) -> None:
        """Close the underlying Neo4j driver connection."""
        await self._driver.close()

    async def verify_connectivity(self) -> None:
        """Verify the Neo4j connection is healthy.

        Raises:
            Exception: If the connection cannot be established.
        """
        await self._driver.verify_connectivity()
        logger.info(
            "neo4j.connected",
            uri=self._uri,
        )