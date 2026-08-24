from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Above this similarity ratio (on normalized, whitespace-tokenized text), chosen/rejected
# are treated as near-duplicates rather than a genuine preference pair. Token-level (not
# character-level) comparison is used so that a single meaningfully-different word or
# operator (e.g. a subtle "off-by-one-symbol" bug in an otherwise-similar answer) doesn't
# get flagged just because most characters are still shared.
NEAR_DUPLICATE_THRESHOLD = 0.9


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for duplicate/near-duplicate comparisons."""
    return re.sub(r"\s+", " ", text).strip().lower()


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is None:
            return rejected
        norm_chosen, norm_rejected = _normalize(chosen), _normalize(rejected)
        if norm_chosen == norm_rejected:
            raise ValueError(
                "chosen and rejected must differ (identical after whitespace/case normalization)"
            )
        similarity = SequenceMatcher(None, norm_chosen.split(), norm_rejected.split()).ratio()
        if similarity >= NEAR_DUPLICATE_THRESHOLD:
            raise ValueError(
                f"chosen and rejected are near-duplicates (similarity={similarity:.3f} >= "
                f"{NEAR_DUPLICATE_THRESHOLD}); rejected must be a meaningfully different response"
            )
        return rejected
