from __future__ import annotations

import numpy as np

# Sequence log-probabilities are mathematically in (-inf, 0], but exactly 0.0 (p == 1.0)
# sends log1mexp to -inf (RuntimeWarning: divide by zero) and blows up the odds ratio.
# Clipping away from both boundaries keeps orpo_loss finite and warning-free.
_LOGP_CLIP_MIN = -30.0
_LOGP_CLIP_MAX = -1e-7


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)."""
    result: np.ndarray = -np.logaddexp(0.0, -x)
    return result


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 - exp(x)) for x <= 0 (e.g. x = a log-probability).

    Uses the standard two-branch formulation (Machler, 2012) to avoid cancellation
    error near x == 0 and underflow for very negative x.
    """
    x = np.asarray(x, dtype=np.float64)
    small = x > -np.log(2.0)
    return np.where(
        small,
        np.log(-np.expm1(x)),
        np.log1p(-np.exp(x)),
    )


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    loss_i = -log sigmoid(beta * [(log pi(y_w|x) - log pi(y_l|x))
                                   - (log pi_ref(y_w|x) - log pi_ref(y_l|x))])
    Averaged over the batch. Implemented with `logaddexp`-based log-sigmoid so it
    stays finite even for large policy/reference log-ratio gaps.
    """
    policy_chosen_logps = np.asarray(policy_chosen_logps, dtype=np.float64)
    policy_rejected_logps = np.asarray(policy_rejected_logps, dtype=np.float64)
    ref_chosen_logps = np.asarray(ref_chosen_logps, dtype=np.float64)
    ref_rejected_logps = np.asarray(ref_rejected_logps, dtype=np.float64)

    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratio - ref_logratio)

    losses = -_log_sigmoid(logits)
    return float(np.mean(losses))


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective: L_SFT + lambda * L_OR.

    `chosen_logps`/`rejected_logps` are sequence-level log-probabilities log p(y|x)
    (<= 0). The odds of a sequence are p / (1 - p), so:
        log_odds(y) = log p(y|x) - log(1 - p(y|x)) = log p(y|x) - log1mexp(log p(y|x))
    L_OR = -log sigmoid(log_odds(chosen) - log_odds(rejected)), averaged over the batch,
    and L_SFT is the mean of the (already-computed, per-example) `sft_nll`.
    """
    sft_nll = np.asarray(sft_nll, dtype=np.float64)
    chosen_logps = np.clip(
        np.asarray(chosen_logps, dtype=np.float64), _LOGP_CLIP_MIN, _LOGP_CLIP_MAX
    )
    rejected_logps = np.clip(
        np.asarray(rejected_logps, dtype=np.float64), _LOGP_CLIP_MIN, _LOGP_CLIP_MAX
    )

    chosen_log_odds = chosen_logps - _log1mexp(chosen_logps)
    rejected_log_odds = rejected_logps - _log1mexp(rejected_logps)
    log_odds_ratio = chosen_log_odds - rejected_log_odds

    or_loss = -_log_sigmoid(log_odds_ratio)
    return float(np.mean(sft_nll) + lambda_orpo * np.mean(or_loss))
