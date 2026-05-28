"""CheckpointManager — atomic JSON checkpoint file for resumable pipelines.

Phase 3 Day 15.  Each stage writes a flag to a single JSON file in the
working directory.  On restart, stages whose flags are marked complete
are skipped.

The file looks like::

    {
      "app_id": "neetpg",
      "completed_stages": {
        "ProcessBookSegments": 1700000000,
        "ExtractEntitiesAndRelations": 1700000001
      }
    }

Where values are UTC timestamps (epoch seconds) recording when the stage
finished.  Writes use a temp-file + rename strategy to be crash-safe.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Stage names — MUST match EntityExtractionPipeline.cs:67-107
# ---------------------------------------------------------------------------
STAGE_PROCESS_BOOK_SEGMENTS = "ProcessBookSegments"
STAGE_EXTRACT_ENTITIES_AND_RELATIONS = "ExtractEntitiesAndRelations"
STAGE_MERGE_ENTITIES = "MergeEntities"
STAGE_MERGE_RELATIONS = "MergeRelations"
STAGE_GENERATE_ENTITY_DESCRIPTIONS_AND_EMBEDDINGS = (
    "GenerateEntityDescriptionsAndEmbeddings"
)
STAGE_GENERATE_RELATION_DESCRIPTIONS_AND_EMBEDDINGS = (
    "GenerateRelationDescriptionsAndEmbeddings"
)
STAGE_STORE_INTO_DATABASES = "StoreIntoDatabases"

# Ordered list for reporting / progress display
ALL_STAGES = [
    STAGE_PROCESS_BOOK_SEGMENTS,
    STAGE_EXTRACT_ENTITIES_AND_RELATIONS,
    STAGE_MERGE_ENTITIES,
    STAGE_MERGE_RELATIONS,
    STAGE_GENERATE_ENTITY_DESCRIPTIONS_AND_EMBEDDINGS,
    STAGE_GENERATE_RELATION_DESCRIPTIONS_AND_EMBEDDINGS,
    STAGE_STORE_INTO_DATABASES,
]


class CheckpointManager:
    """Read/write a single JSON checkpoint file atomically.

    Usage::

        ckpt = CheckpointManager("./workspace/neetpg")
        if ckpt.is_stage_complete("ProcessBookSegments"):
            print("skip stage 1")
        ...
        await ckpt.mark_stage_complete("ProcessBookSegments")
    """

    def __init__(self, workspace_dir: str | Path, app_id: str) -> None:
        """
        Args:
            workspace_dir: Directory where the checkpoint file lives.
            app_id: Tenant identifier; used for logging and metadata.
        """
        self._workspace = Path(workspace_dir)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._app_id = app_id
        self._path = self._workspace / "pipeline.checkpoint.json"
        self._state: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_stage_complete(self, stage_name: str) -> bool:
        """Return True if *stage_name* is already marked complete."""
        return stage_name in self._state.get("completed_stages", {})

    def mark_stage_complete(self, stage_name: str) -> None:
        """Persist *stage_name* as complete with current timestamp.

        Uses temp-file + rename for crash safety.
        """
        self._state.setdefault("completed_stages", {})[stage_name] = int(
            time.time()
        )
        self._save()

    def reset(self) -> None:
        """Clear all stage completions (useful for forced reingest)."""
        self._state["completed_stages"] = {}
        self._save()

    @property
    def completed_stages(self) -> dict[str, int]:
        """Dict of {stage_name: completed_timestamp}."""
        return dict(self._state.get("completed_stages", {}))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        """Load existing state or return a fresh skeleton."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"app_id": self._app_id, "completed_stages": {}}

    def _save(self) -> None:
        """Atomic write: temp file → rename."""
        tmp_path = self._path.with_suffix(".tmp")
        self._state["app_id"] = self._app_id  # always sync
        tmp_path.write_text(
            json.dumps(self._state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Atomic rename on Windows requires removing the target first
        try:
            tmp_path.replace(self._path)
        except OSError:
            # Fallback for edge cases on some file systems
            if self._path.exists():
                self._path.unlink()
            tmp_path.rename(self._path)