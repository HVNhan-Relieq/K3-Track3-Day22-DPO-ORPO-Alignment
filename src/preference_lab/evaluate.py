from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import PreferenceExample


def score_response(prompt: str, response: str) -> float:
    """Deterministic, model-free quality proxy for a single response.

    Combines three cheap, reference-free signals so runs are fully reproducible
    without a real policy model (suitable for CPU-only lab environments):
      - relevance: fraction of "content" prompt words (len > 3) echoed in the response
      - diversity: unique-token ratio (penalizes repetition/filler)
      - detail: response length, capped so verbosity alone can't dominate

    This is a heuristic stand-in for real logprob-based scoring, not a claim of
    factual correctness -- it exists so `pref-lab evaluate` reports a genuine,
    reproducible signal instead of a hardcoded 1.0/0.0 placeholder.
    """
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    response_tokens = _tokens(response)
    if not response_tokens:
        return 0.0

    prompt_content_words = {w for w in _tokens(prompt) if len(w) > 3}
    if prompt_content_words:
        overlap = len(prompt_content_words & set(response_tokens)) / len(prompt_content_words)
    else:
        overlap = 0.0

    diversity = len(set(response_tokens)) / len(response_tokens)
    detail = min(len(response_tokens) / 30.0, 1.0)

    return 0.4 * overlap + 0.3 * diversity + 0.3 * detail


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return the fraction of pairs where the chosen score beats the rejected score.

    Ties (equal scores) are handled explicitly and counted as half a win, matching
    the standard convention for pairwise preference accuracy rather than being
    silently counted as either a win or a loss.
    """
    if not examples:
        return 0.0
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError(
            "chosen_scores and rejected_scores must have the same length as examples "
            f"(got {len(chosen_scores)}, {len(rejected_scores)}, {len(examples)})"
        )

    total = 0.0
    for c, r in zip(chosen_scores, rejected_scores, strict=True):
        if c > r:
            total += 1.0
        elif c == r:
            total += 0.5
    return total / len(examples)

def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
