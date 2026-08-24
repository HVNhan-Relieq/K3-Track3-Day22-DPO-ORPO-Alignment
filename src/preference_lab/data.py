from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


def _normalize_prompt(prompt: str) -> str:
    """Normalize a prompt for duplicate detection (whitespace/case-insensitive)."""
    return re.sub(r"\s+", " ", prompt).strip().lower()


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Raises ValueError with the offending line number for malformed JSON, schema
    violations, and duplicate prompts (duplicates are rejected rather than silently
    kept, since they would let the same prompt leak across train/val splits).
    """
    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{path}:{lineno}: invalid preference example ({exc})") from exc

            norm_prompt = _normalize_prompt(example.prompt)
            if norm_prompt in seen_prompts:
                first_lineno = seen_prompts[norm_prompt]
                raise ValueError(
                    f"{path}:{lineno}: duplicate prompt (first seen on line {first_lineno}): "
                    f"{example.prompt!r}"
                )
            seen_prompts[norm_prompt] = lineno
            examples.append(example)
    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid leakage.

    Groups examples by (normalized) prompt, deterministically shuffles the groups
    using `seed`, and allocates whole groups to validation until at least
    `validation_ratio` of examples are covered. This guarantees no prompt appears
    in both splits, while remaining reproducible across runs.
    """
    if not examples:
        return [], []
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0.0, 1.0)")

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(_normalize_prompt(example.prompt), []).append(example)

    prompt_keys = list(groups.keys())
    random.Random(seed).shuffle(prompt_keys)

    target_val_count = round(len(examples) * validation_ratio)
    val: list[PreferenceExample] = []
    train: list[PreferenceExample] = []
    for key in prompt_keys:
        group = groups[key]
        if len(val) < target_val_count:
            val.extend(group)
        else:
            train.extend(group)

    return train, val
