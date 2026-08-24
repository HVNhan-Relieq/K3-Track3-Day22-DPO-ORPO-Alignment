from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_load_jsonl_reports_line_number_for_malformed_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(
        '{"prompt": "ok", "chosen": "a", "rejected": "b"}\n'
        '{"prompt": "broken, "chosen": "a", "rejected": "b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_jsonl(bad_file)


def test_load_jsonl_reports_line_number_for_schema_violation(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_schema.jsonl"
    bad_file.write_text(
        '{"prompt": "ok", "chosen": "a", "rejected": "b"}\n'
        '{"prompt": "same", "chosen": "same answer", "rejected": "same answer"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad_schema\.jsonl:2"):
        load_jsonl(bad_file)


def test_load_jsonl_rejects_duplicate_prompts(tmp_path: Path) -> None:
    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text(
        '{"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}\n'
        '{"prompt": "  what is 2+2?  ", "chosen": "four", "rejected": "five"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prompt"):
        load_jsonl(dup_file)


def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)


def test_split_by_prompt_has_no_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.25, seed=42)
    train_prompts = {ex.prompt.strip().lower() for ex in train}
    val_prompts = {ex.prompt.strip().lower() for ex in val}
    assert train_prompts.isdisjoint(val_prompts)
    assert len(val) > 0
    assert len(train) > 0


def test_split_by_prompt_is_deterministic_given_seed() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train1, val1 = split_by_prompt(examples, validation_ratio=0.3, seed=7)
    train2, val2 = split_by_prompt(examples, validation_ratio=0.3, seed=7)
    assert [e.prompt for e in train1] == [e.prompt for e in train2]
    assert [e.prompt for e in val1] == [e.prompt for e in val2]


def test_split_by_prompt_groups_duplicate_normalized_prompts_together() -> None:
    from preference_lab.schemas import PreferenceExample

    examples = [
        PreferenceExample(prompt="Same Prompt", chosen="a1", rejected="b1"),
        PreferenceExample(prompt="same prompt", chosen="a2", rejected="b2"),
        PreferenceExample(prompt="Different Prompt", chosen="a3", rejected="b3"),
    ]
    train, val = split_by_prompt(examples, validation_ratio=0.5, seed=1)
    train_prompts = {ex.prompt.strip().lower() for ex in train}
    val_prompts = {ex.prompt.strip().lower() for ex in val}
    assert train_prompts.isdisjoint(val_prompts)
