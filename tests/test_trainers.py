import json
from pathlib import Path

import pytest

from preference_lab.schemas import PreferenceExample
from preference_lab.trainers import PreferenceTrainer, TrainingConfig

EXAMPLES = [
    PreferenceExample(prompt="p1", chosen="a good detailed answer", rejected="bad"),
    PreferenceExample(prompt="p2", chosen="another good detailed answer", rejected="nope"),
    PreferenceExample(prompt="p3", chosen="yet another solid answer here", rejected="wrong"),
]


def test_train_rejects_empty_examples(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PreferenceTrainer(TrainingConfig(method="dpo"), [], tmp_path)


def test_dpo_mock_training_loss_decreases_and_is_logged(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(TrainingConfig(method="dpo", batch_size=2), EXAMPLES, tmp_path, num_epochs=4)
    summary = trainer.train()

    history = summary["loss_history"]
    assert len(history) == 4
    # Each epoch nudges the policy further toward preferring chosen -> loss should
    # be non-increasing across epochs (mock DPO loss trends down as it "learns").
    assert all(history[i] >= history[i + 1] - 1e-9 for i in range(len(history) - 1))

    log_path = tmp_path / "train_log.json"
    assert log_path.exists()
    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged["loss_history"] == pytest.approx(history)
    assert logged["method"] == "dpo"
    assert logged["num_examples"] == len(EXAMPLES)


def test_orpo_mock_training_loss_decreases(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(TrainingConfig(method="orpo", lambda_orpo=0.5), EXAMPLES, tmp_path, num_epochs=4)
    summary = trainer.train()
    history = summary["loss_history"]
    assert len(history) == 4
    assert history[-1] < history[0]


def test_mock_method_runs_without_a_real_objective(tmp_path: Path) -> None:
    trainer = PreferenceTrainer(TrainingConfig(method="mock"), EXAMPLES, tmp_path, num_epochs=2)
    summary = trainer.train()
    assert summary["loss_history"] == [0.0, 0.0]


def test_train_is_deterministic_given_same_inputs(tmp_path: Path) -> None:
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    r1 = PreferenceTrainer(TrainingConfig(method="dpo"), EXAMPLES, out1, num_epochs=3).train()
    r2 = PreferenceTrainer(TrainingConfig(method="dpo"), EXAMPLES, out2, num_epochs=3).train()
    assert r1["loss_history"] == r2["loss_history"]
