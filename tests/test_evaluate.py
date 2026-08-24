import pytest

from preference_lab.evaluate import pairwise_accuracy, score_response
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_counts_ties_as_half() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="c", rejected="d"),
    ]
    # One win, one tie -> (1.0 + 0.5) / 2
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0]) == pytest.approx(0.75)


def test_pairwise_accuracy_rejects_mismatched_lengths() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="same length"):
        pairwise_accuracy(examples, [1.0, 2.0], [0.0])


def test_pairwise_accuracy_empty_examples_returns_zero() -> None:
    assert pairwise_accuracy([], [], []) == 0.0


def test_score_response_rewards_relevant_detailed_answer_over_generic_one() -> None:
    prompt = "Explain how backpropagation updates neural network weights."
    good = score_response(
        prompt,
        "Backpropagation computes gradients of the loss with respect to each weight "
        "using the chain rule, then updates weights via gradient descent.",
    )
    bad = score_response(prompt, "It just works somehow.")
    assert good > bad


def test_score_response_empty_response_scores_zero() -> None:
    assert score_response("Any prompt", "") == 0.0


def test_score_response_is_deterministic() -> None:
    prompt, response = "What is 2+2?", "2 + 2 equals 4."
    assert score_response(prompt, response) == score_response(prompt, response)
