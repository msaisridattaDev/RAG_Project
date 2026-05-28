"""Keyword extractor — local + global keywords from a query string.

Phase 3 query path. Mirrors autogen.net's LightRag.cs keyword-extraction step:
  local_keywords  → entity-focused retrieval (LOCAL mode)
  global_keywords → relationship/topic retrieval (GLOBAL mode)
  combined        → HYBRID mode uses both

Falls back to a simple word-split if the LLM call fails so the query path
is never broken by an LLM outage.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from autogen.logging.setup import get_logger

if TYPE_CHECKING:
    from autogen.protocols.llm import LlmClient

logger = get_logger("autogen.extraction.keywords")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a keyword extractor for medical and exam content. "
    "Output valid JSON only — no prose, no code fences."
)

_USER = """\
Extract keywords from the query below for two retrieval paths.

local_keywords  : specific entity names, drug names, diseases, symptoms,
                  procedures, anatomy, genes — directly mentioned or implied.
                  Aim for 3–8 items.

global_keywords : broader topic categories, themes, subject areas the query
                  belongs to.
                  Aim for 2–5 items.

Output exactly:
{{"local": ["kw1", "kw2"], "global": ["topic1", "topic2"]}}

Query: {query}"""


# ---------------------------------------------------------------------------
# KeywordExtractor
# ---------------------------------------------------------------------------


class KeywordExtractor:
    """Extract local and global keywords from a user query via LLM.

    Usage::

        extractor = KeywordExtractor(llm, model="groq/llama-3.1-8b-instant")
        local_kws, global_kws = await extractor.extract("MOA of aspirin?")
        # local_kws  → ["aspirin", "mechanism of action", "COX inhibitor"]
        # global_kws → ["pharmacology", "NSAIDs"]
    """

    def __init__(self, llm: LlmClient, model: str) -> None:
        self._llm = llm
        self._model = model

    async def extract(self, query: str) -> tuple[list[str], list[str]]:
        """Return (local_keywords, global_keywords) for the query.

        Never raises — falls back to word-split on any failure.
        """
        from autogen.models.llm import LlmMessage

        prompt = _USER.format(query=query)
        messages = [
            LlmMessage(role="system", content=_SYSTEM),
            LlmMessage(role="user", content=prompt),
        ]
        try:
            raw = await self._llm.complete(messages, self._model, temperature=0.0)
            data = _parse_json(raw)
            if data:
                local = [str(k).strip() for k in data.get("local", []) if k]
                global_ = [str(k).strip() for k in data.get("global", []) if k]
                logger.debug(
                    "keywords.ok",
                    query=query[:80],
                    local=len(local),
                    global_=len(global_),
                )
                return local, global_
        except Exception as exc:
            logger.debug("keywords.llm_failed", query=query[:80], error=str(exc))

        # Fallback: words longer than 2 chars become local keywords
        fallback = [w.strip() for w in query.split() if len(w.strip()) > 2]
        return fallback[:8], []


# ---------------------------------------------------------------------------
# JSON parse helper
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First { … last }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            pass
    return None
