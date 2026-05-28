"""NormalizeNames — global entity-name synonym clustering (Phase 3 Day 13).

After per-chunk extraction, the same real-world concept appears under many
surface forms: "MI", "Myocardial Infarction", "myocardial infarct".
NormalizeNames runs one LLM call across ALL extracted names to produce a
mapping {original_name → canonical_name}, then the pipeline applies it to
every EntityNode and EntityRelation.

Different from EntityTypeResolver:
    - EntityTypeResolver    — normalizes *types* (DRUG ≡ Pharmaceutical)
                              per-entity in-flight during extraction.
    - NormalizeNames        — normalizes *names* (MI ≡ Myocardial Infarction)
                              in a global post-pass across the corpus.

Usage::

    normalizer = NormalizeNames(llm, model="gpt-4o-mini")
    mapping = await normalizer.normalize(all_entity_names)
    # mapping == {"MI": "Myocardial Infarction", "heart attack": "Myocardial Infarction", ...}
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from autogen.extraction.prompts import NORMALIZE_NAMES_PROMPT

if TYPE_CHECKING:
    from autogen.protocols.llm import LlmClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Max number of names to send in a single normalization call.
# If there are more, we batch them.  LLMs handle ~200-300 names well.
# ---------------------------------------------------------------------------
_MAX_NAMES_PER_BATCH = 250


class NormalizeNames:
    """Global entity-name synonym clustering via LLM.

    Stateless — call normalize() with a list of names and get back a dict.
    """

    def __init__(
        self,
        llm: LlmClient,
        *,
        model: str = "gpt-4o-mini",
        max_per_batch: int = _MAX_NAMES_PER_BATCH,
    ) -> None:
        """
        Args:
            llm: LLM client for normalization calls.
            model: Model name (cheap/fast is fine — classification task).
            max_per_batch: Max names per LLM call.  Default 250.
        """
        self._llm = llm
        self._model = model
        self._max_per_batch = max_per_batch

    async def normalize(self, names: list[str]) -> dict[str, str]:
        """Return a mapping from every original name to its canonical form.

        Names that are already canonical map to themselves.

        Args:
            names: Flat list of all extracted entity names (deduplicated).

        Returns:
            Dict[str, str] where keys are original names, values are canonical.
        """
        if not names:
            return {}

        # Deduplicate but preserve order for reproducibility
        unique = list(dict.fromkeys(n.lower().strip() for n in names))
        logger.info("Normalizing %d unique entity names", len(unique))

        mapping: dict[str, str] = {}
        # All names are self-mapping by default
        for name in unique:
            mapping[name] = name

        # Batch if needed
        batches = [
            unique[i : i + self._max_per_batch]
            for i in range(0, len(unique), self._max_per_batch)
        ]

        for batch_idx, batch in enumerate(batches):
            logger.debug(
                "Normalization batch %d/%d (%d names)",
                batch_idx + 1,
                len(batches),
                len(batch),
            )
            try:
                batch_mapping = await self._normalize_batch(batch)
                # Merge: the LLM output is {original → canonical}
                # Keys in the LLM output may be in original casing;
                # we lower them to match our deduped keys.
                for orig, canonical in batch_mapping.items():
                    key = orig.lower().strip()
                    if key in mapping:
                        mapping[key] = canonical.strip()
            except Exception:
                logger.warning(
                    "Normalization batch %d failed; keeping self-mappings",
                    batch_idx,
                    exc_info=True,
                )

        return mapping

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _normalize_batch(self, names: list[str]) -> dict[str, str]:
        """Send one batch of names to the LLM for clustering."""
        names_str = "\n".join(f"- {n}" for n in names)
        prompt = NORMALIZE_NAMES_PROMPT.format(entity_names=names_str)

        response = await self._llm.complete(
            [{"role": "user", "content": prompt}],  # type: ignore[arg-type]
            model=self._model,
        )

        # Try to extract JSON from the response
        return self._parse_json_response(response)

    @staticmethod
    def _parse_json_response(response: str) -> dict[str, str]:
        """Extract and parse the JSON mapping from the LLM response.

        Handles common LLM output patterns:
            - Pure JSON: {"key": "value", ...}
            - JSON in markdown fence: ```json ... ```
            - JSON with trailing text
        """
        text = response.strip()

        # Pattern 1: Markdown code fence
        if "```" in text:
            # Extract content between first ``` and last ```
            parts = text.split("```")
            # Usually the JSON is in parts[1] (after opening ```json)
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue

        # Pattern 2: Raw JSON (possibly with trailing text)
        # Find the outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(
            "Could not parse JSON from normalization response: %s",
            text[:200],
        )
        return {}