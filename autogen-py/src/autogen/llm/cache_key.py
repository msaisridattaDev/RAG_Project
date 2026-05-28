"""SHA256 cache key — mirrors autogen.net's CachingMiddleware keying logic.

Two requests with bit-identical inputs produce the same 64-character hex
key. Different by even one character → different key. This is exact-match
caching, not semantic.

The payload includes messages (canonical JSON), model string, temperature,
and response_format so that parameter variations map to different cache slots.
"""

from __future__ import annotations

import hashlib
import json

from autogen.models.llm import LlmMessage


def cache_key(
    messages: list[LlmMessage],
    model: str,
    temperature: float = 0.0,
    response_format: object = None,
) -> str:
    """Return a deterministic SHA256 hex digest for this request.

    Canonicalization rules:
        - dump LlmMessage as dicts (Pydantic .model_dump())
        - sort_keys=True so dict key order doesn't matter
        - compact separators (",", ":") — no trailing whitespace variance
        - encode to UTF-8 bytes before hashing
    """
    payload: dict[str, object] = {
        "messages": [m.model_dump() for m in messages],
        "model": model,
        "temperature": temperature,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()