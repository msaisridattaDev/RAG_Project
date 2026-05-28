"""Base value objects and mixins — mirrors autogen.net tenancy model.

Every entity in the system is scoped by appId (the primary multi-tenancy
dimension). This module provides the foundational types that enforce
that scoping at the type level.
"""

from __future__ import annotations

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class AppId(str):
    """Application identifier — the primary tenancy dimension.

    Mirrors autogen.net's AppId type. Values include:
    - neetpg (NEET PG)
    - neetug (NEET UG)
    - mds (MDS)
    - ems (EMS)

    Every vector index, KV store namespace, Neo4j graph namespace,
    and agent factory is keyed by AppId.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.str_schema()


class TenantId(str):
    """Tenant identifier — secondary tenancy dimension within an app.

    Mirrors autogen.net's TenantId type. Typically maps to an
    organization or institution within the app.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.str_schema()


class HasAppId(BaseModel):
    """Mixin that adds appId tenancy to any Pydantic model.

    Every model that participates in multi-tenant storage should
    inherit from this mixin.
    """

    app_id: AppId
