"""Graph store protocol — mirrors autogen.net IGraphStore.

Phase 3 (Days 13-16) builds against this interface. A concrete Neo4j
implementation will be added in Phase 3 Day 14. The namespace is bound at
construction time (``{label_prefix}_{app_id}``) so every method below
operates within the per-tenant subgraph automatically — there is no
``app_id`` argument on the methods.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from autogen.models.storage import EntityNode, EntityRelation


@runtime_checkable
class GraphStore(Protocol):
    """Per-app graph store — mirrors autogen.net IGraphStore.

    Every operation is scoped to ``self.namespace`` so callers can't
    accidentally cross-tenant.
    """

    @property
    def namespace(self) -> str:
        """The per-app namespace (typically ``"{label}_{app_id}"``)."""
        ...

    async def ensure_constraints(self) -> None:
        """Create indexes/constraints if they do not exist (idempotent)."""
        ...

    async def upsert_node(self, node: EntityNode) -> None:
        """Insert or merge an entity node. Merges on ``node.id``."""
        ...

    async def upsert_nodes(self, nodes: list[EntityNode]) -> None:
        """Bulk insert or merge entity nodes."""
        ...

    async def upsert_edge(self, edge: EntityRelation) -> None:
        """Insert or merge an undirected edge. Merges on ``edge.id``."""
        ...

    async def upsert_edges(self, edges: list[EntityRelation]) -> None:
        """Bulk insert or merge undirected edges."""
        ...

    async def node_degree(self, node_id: str) -> int:
        """Return the degree (incident-edge count) of a node."""
        ...

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Return the count of parallel edges between two nodes."""
        ...

    async def get_nodes(self, ids: list[str]) -> list[EntityNode]:
        """Batch-fetch nodes by their IDs. Preserves input order."""
        ...

    async def get_relations(self, ids: list[str]) -> list[EntityRelation]:
        """Batch-fetch edges by their IDs. Preserves input order."""
        ...

    async def get_node_edges(self, node_id: str) -> list[EntityRelation]:
        """Return every edge incident on the given node."""
        ...
