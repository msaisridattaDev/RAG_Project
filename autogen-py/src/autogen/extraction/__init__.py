"""Entity and relationship extraction from text chunks (Phase 3 Days 12-13).

Public API:
    EntityExtractor  — main entry point, runs per-chunk extraction with gleaning + recovery
    EntityTypeResolver — canonical type normalization (DRUG ≡ Pharmaceutical → "DRUG")
    NormalizeNames   — global synonym clustering (MI ≡ Myocardial Infarction)
    ExtractionResult — dataclass returned by extract_from_chunk()
"""
from __future__ import annotations

from autogen.extraction.extractor import EntityExtractor, ExtractionResult
from autogen.extraction.normalizer import NormalizeNames
from autogen.extraction.type_resolver import EntityTypeResolver

__all__ = [
    "EntityExtractor",
    "EntityTypeResolver",
    "ExtractionResult",
    "NormalizeNames",
]