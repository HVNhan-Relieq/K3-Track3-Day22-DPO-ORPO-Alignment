# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A production-style *starter/skeleton* for a 2-hour lab on preference alignment (DPO/ORPO). It is **intentionally incomplete**: functions marked `TODO(student)` raise `NotImplementedError` or contain mock logic that must be implemented. Do not "complete" the repo wholesale — this is a teaching artifact, and its incompleteness is deliberate.

## Lab rules (respect these when editing)

1. Do not rewrite the whole repository.
2. Implement only the `TODO(student)` blocks unless there's a clear reason to touch other code.
3. Keep tests passing after each milestone.
4. Never commit secrets, model weights, or private datasets.

## Commands

```bash
pip install -e '.[dev]'          # setup (add ',train' for torch/transformers/trl/peft)
make test                         # pytest -q
make lint                         # ruff check src tests
make typecheck                    # mypy src (strict mode)
make format                       # ruff format src tests
make run-eval                     # pref-lab evaluate --config configs/local.yaml
pytest tests/test_data.py         # run a single test file
pytest tests/test_losses.py -k dpo   # run a single test by keyword
scripts/smoke_test.sh             # validate + evaluate + print metrics end-to-end
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `mypy src`, then `pytest -q` on Python 3.11 — mirror this before considering work done.

## Architecture

- `src/preference_lab/schemas.py` — `PreferenceExample` (pydantic): the `prompt`/`chosen`/`rejected`/`metadata` record that flows through the whole pipeline. Validation added here (e.g. near-duplicate detection) affects loading, training, and eval uniformly.
- `src/preference_lab/data.py` — JSONL loading (`load_jsonl`) and prompt-level train/val splitting (`split_by_prompt`). The split must be by **prompt**, not by row, to avoid leakage between train/val.
- `src/preference_lab/losses.py` — `dpo_loss` and `orpo_loss`, pure numpy functions operating on batches of sequence log-probabilities. No torch dependency here by design — keep it framework-agnostic and testable on CPU.
- `src/preference_lab/trainers.py` — `PreferenceTrainer` wraps a `TrainingConfig` (method: `dpo`/`orpo`/`mock`, `beta`, `lambda_orpo`, `max_length`, `batch_size`) and exposes `.train()`. Implementations should write checkpoints/metrics only to the configured `output_dir` — no other side effects.
- `src/preference_lab/evaluate.py` — `pairwise_accuracy` (compares chosen vs. rejected scores) and `write_metrics` (writes `outputs/metrics.json`).
- `src/preference_lab/config.py` — trivial YAML loader (`load_config`) for files like `configs/local.yaml`.
- `src/preference_lab/cli.py` — Typer app (`pref-lab`) with `validate` and `evaluate` subcommands. `evaluate` currently uses mock scores (`chosen=1.0`, `rejected=0.0`) as a placeholder for real model logprob scoring.
- `scripts/generate_data.py` — optional OpenAI-backed synthetic data generator (`pref-lab`-style Typer app, separate entrypoint) that appends new preference pairs to `data/synthetic_preferences.jsonl`, seeded from `data/sample_preferences.jsonl`. Requires `OPENAI_API_KEY`.

### Data flow

`configs/local.yaml` → `load_config` → `load_jsonl(paths.train_data)` → list of `PreferenceExample` → (student-implemented) split/training/scoring → `write_metrics` → `outputs/metrics.json`.

### Config shape (`configs/local.yaml`)

```yaml
seed: 42
paths: {train_data, output_dir}
training: {method: dpo|orpo|mock, beta, lambda_orpo, max_length, batch_size}
evaluation: {regression_prompts: path}
```

## Docs worth knowing about

- `docs/lab_guide.md` — the milestone-by-milestone task list students follow (data loader → loss function → evaluation → report).
- `docs/REPORT_TEMPLATE.md` — the student deliverable; fill-in-the-blank report on dataset cleaning, DPO/ORPO choice, numerical stability, metrics, and observed failure modes.
- `docs/regression_prompts.md` — small fixed prompt set for safety/behavior regression checks, meant to be run before/after training.
- `docs/data_card_template.md` — dataset documentation template.
