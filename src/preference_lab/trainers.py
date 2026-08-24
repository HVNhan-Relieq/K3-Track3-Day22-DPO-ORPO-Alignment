from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .losses import dpo_loss, orpo_loss
from .schemas import PreferenceExample


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2


def _pseudo_logprob(text: str, max_length: int) -> float:
    """Deterministic stand-in for a real sequence log-probability.

    Hashes `text` to a stable value in [0, 1) so the same text always yields
    the same score, then blends it with a length penalty into a plausible
    negative log-probability. This has no relationship to any actual model's
    P(text) -- it exists purely so the mock trainer below has *something*
    numeric and reproducible to feed into `dpo_loss`/`orpo_loss`.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:4], "big") / 2**32
    length_penalty = min(len(text), max_length) / max_length
    return float(-0.5 - 4.5 * length_penalty - 2.0 * fraction)


class PreferenceTrainer:
    """Mock CPU trainer for DPO/ORPO.

    No neural network is trained here -- there is no torch/transformers/trl
    dependency and no GPU requirement. Instead, each example's chosen/rejected
    responses get a deterministic simulated log-probability (`_pseudo_logprob`),
    and each epoch nudges the simulated *policy* log-probabilities to prefer
    `chosen` a little more (while the simulated *reference* log-probabilities
    stay fixed), mimicking what a real gradient step on `dpo_loss`/`orpo_loss`
    would do to the loss curve. This exercises the batching/epoch/logging
    plumbing a real trainer needs, with a real (already-implemented) loss
    function computed every step -- it's a placeholder for the model-facing
    half of training, not for the loss math.

    For an actual policy update, replace `_pseudo_logprob` and the per-epoch
    nudge with real forward passes through a policy/reference model (e.g. a
    `trl.DPOTrainer`/`trl.ORPOTrainer`, installed via `pip install -e '.[dev,train]'`).
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_examples: list[PreferenceExample],
        output_dir: str | Path,
        num_epochs: int = 5,
        learning_step: float = 0.15,
    ) -> None:
        if not train_examples:
            raise ValueError("train_examples must be non-empty")
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        self.config = config
        self.train_examples = train_examples
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.learning_step = learning_step

    def train(self) -> dict[str, Any]:
        """Run the mock training loop.

        Writes `<output_dir>/train_log.json` with the per-epoch loss history
        and returns the same summary dict, so callers (CLI, tests) don't need
        to re-read the file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ref_chosen = np.array(
            [_pseudo_logprob(ex.chosen, self.config.max_length) for ex in self.train_examples]
        )
        ref_rejected = np.array(
            [_pseudo_logprob(ex.rejected, self.config.max_length) for ex in self.train_examples]
        )
        policy_chosen = ref_chosen.copy()
        policy_rejected = ref_rejected.copy()

        batch_size = max(1, self.config.batch_size)
        loss_history: list[float] = []

        for _epoch in range(self.num_epochs):
            batch_losses: list[float] = []
            for start in range(0, len(self.train_examples), batch_size):
                end = start + batch_size
                pc, pr = policy_chosen[start:end], policy_rejected[start:end]
                rc, rr = ref_chosen[start:end], ref_rejected[start:end]
                batch_losses.append(self._batch_loss(pc, pr, rc, rr))
            loss_history.append(float(np.mean(batch_losses)))

            # Simulated gradient step: pull the policy toward preferring `chosen`
            # over `rejected` a bit more, clipped away from the log-prob boundary.
            policy_chosen = np.clip(policy_chosen + self.learning_step, -30.0, -1e-7)
            policy_rejected = np.clip(policy_rejected - self.learning_step, -30.0, -1e-7)

        summary: dict[str, Any] = {
            "method": self.config.method,
            "num_epochs": self.num_epochs,
            "batch_size": batch_size,
            "num_examples": len(self.train_examples),
            "loss_history": loss_history,
            "note": (
                "Mock CPU trainer: loss computed on simulated log-probabilities, "
                "not a trained neural network. See PreferenceTrainer docstring."
            ),
        }
        (self.output_dir / "train_log.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    def _batch_loss(
        self,
        policy_chosen: np.ndarray,
        policy_rejected: np.ndarray,
        ref_chosen: np.ndarray,
        ref_rejected: np.ndarray,
    ) -> float:
        if self.config.method == "orpo":
            sft_nll = -policy_chosen  # proxy NLL of the chosen response under the policy
            return orpo_loss(sft_nll, policy_chosen, policy_rejected, self.config.lambda_orpo)
        if self.config.method == "dpo":
            return dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, self.config.beta)
        # method == "mock" (or anything else unrecognized): exercise the batching/logging
        # plumbing without committing to a preference objective.
        return 0.0
