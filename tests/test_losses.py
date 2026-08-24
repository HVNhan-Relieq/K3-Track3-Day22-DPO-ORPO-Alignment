import math
import warnings

import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_matches_reference_formula() -> None:
    policy_chosen = np.array([-0.5, -0.2])
    policy_rejected = np.array([-1.5, -0.3])
    ref_chosen = np.array([-0.6, -0.25])
    ref_rejected = np.array([-1.0, -0.4])
    beta = 0.1

    logits = beta * (
        (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    )
    expected = float(np.mean([-math.log(1 / (1 + math.exp(-x))) for x in logits]))

    result = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
    assert result == pytest.approx(expected, rel=1e-6)


def test_dpo_loss_prefers_correctly_ranked_policy() -> None:
    # Policy strongly prefers chosen over rejected, matching the reference model:
    # log-ratio gap is large and positive -> loss should be small (near 0).
    good = dpo_loss(
        np.array([-0.1]), np.array([-5.0]), np.array([-0.1]), np.array([-5.0]), beta=0.5
    )
    # Policy prefers rejected over chosen -> loss should be large.
    bad = dpo_loss(
        np.array([-5.0]), np.array([-0.1]), np.array([-0.1]), np.array([-5.0]), beta=0.5
    )
    assert good < bad
    assert good >= 0.0
    assert bad >= 0.0


def test_dpo_loss_is_finite_for_extreme_logprobs() -> None:
    result = dpo_loss(
        np.array([-1e-8, -50.0]),
        np.array([-50.0, -1e-8]),
        np.array([-1e-8, -1e-8]),
        np.array([-50.0, -50.0]),
        beta=1.0,
    )
    assert math.isfinite(result)


def test_orpo_loss_is_sft_nll_plus_odds_ratio_penalty() -> None:
    sft_nll = np.array([1.0, 0.8])
    chosen_logps = np.array([-0.5, -0.4])
    rejected_logps = np.array([-1.5, -1.2])
    lambda_orpo = 0.1

    result = orpo_loss(sft_nll, chosen_logps, rejected_logps, lambda_orpo=lambda_orpo)

    # The odds-ratio penalty term must be non-negative (it's -log sigmoid(.)),
    # so the total loss should be at least the mean SFT NLL.
    assert result >= float(np.mean(sft_nll))
    assert math.isfinite(result)


def test_orpo_loss_penalizes_reversed_preference_more() -> None:
    sft_nll = np.array([0.5])
    # Chosen much more likely than rejected -> small odds-ratio penalty.
    aligned = orpo_loss(sft_nll, np.array([-0.1]), np.array([-5.0]), lambda_orpo=1.0)
    # Rejected much more likely than chosen -> large odds-ratio penalty.
    reversed_ = orpo_loss(sft_nll, np.array([-5.0]), np.array([-0.1]), lambda_orpo=1.0)
    assert aligned < reversed_


def test_orpo_loss_is_finite_for_extreme_logprobs() -> None:
    result = orpo_loss(
        np.array([0.1]), np.array([-1e-6]), np.array([-30.0]), lambda_orpo=0.1
    )
    assert math.isfinite(result)


def test_orpo_loss_handles_logp_zero_without_warning_or_nan() -> None:
    # logp == 0.0 (p == 1.0) is the documented ORPO edge case: naive log1mexp(0) is
    # -inf, which used to leak into log-odds and trigger "divide by zero" warnings.
    # orpo_loss must clip logp away from the boundary instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = orpo_loss(
            np.array([0.1]), np.array([0.0]), np.array([-1.0]), lambda_orpo=0.1
        )
    assert math.isfinite(result)
