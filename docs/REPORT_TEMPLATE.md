# Preference Alignment Experiment Report (Student Template)

*Instructions: Fill out this report as you complete the lab milestones. Replace all bracketed text `[like this]` with your own findings.*

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: `24` (from `data/sample_preferences.jsonl`; a separate `data/synthetic_preferences.jsonl` with `12` hand-authored pairs across coding/safety/math/troubleshooting domains was added but is not wired into the default run config).
- **Validation issues found**: Line 1 of `data/sample_preferences.jsonl` had unescaped double quotes around `"self-attention"` inside the `prompt` string, producing invalid JSON (`json.JSONDecodeError`) on load.
- **Cleaning steps taken**: Escaped the embedded quotes (`\"self-attention\"`) so the line parses as valid JSON. Also hardened `load_jsonl` (`src/preference_lab/data.py`) so any future malformed line raises `ValueError` with the exact `path:line_number` instead of crashing with an opaque traceback, and so exact/whitespace-insensitive duplicate prompts are rejected at load time (also reported with the offending line number).

### Split Strategy
- **Train/Val Ratio**: `80/20` (`validation_ratio=0.2`), e.g. on the 24-example sample set this yields `19` train / `5` val.
- **Leakage Prevention**: `split_by_prompt` groups examples by normalized prompt (whitespace-collapsed, lowercased) first, then deterministically shuffles the *groups* (not individual rows) with `random.Random(seed)` before allocating whole groups to validation until the target ratio is met. This guarantees a prompt's chosen/rejected pair — and any other example sharing that prompt — never ends up split across train and val. Verified with `tests/test_data.py::test_split_by_prompt_has_no_leakage`, which asserts the train/val prompt sets are disjoint on the sample data, and `test_split_by_prompt_is_deterministic_given_seed`, which checks the same seed reproduces an identical split.

## 2. Implementation: DPO

### Objective Selection
- **Why this method?**: DPO was implemented first because it needs no reward model and maps directly onto the sample data we already have (single chosen/rejected pair per prompt, no separate SFT reference generations). ORPO was also implemented (`orpo_loss`) since it drops the need for a frozen reference model entirely, which is convenient for the CPU-only lab setting, but DPO was used as the primary objective for this report.
- **Key Hyperparameters**:
    - `beta`: `0.1` (from `configs/local.yaml`)
    - `lambda_orpo` (if applicable): `0.1` (used for the ORPO implementation, not the primary run)

### Numerical Stability
- **Challenges**: The naive DPO formula `-log(sigmoid(beta * log_ratio_diff))` overflows/underflows in plain numpy when the log-ratio gap is large (e.g. one sequence log-prob near 0 and the other near -50), since `sigmoid` saturates and its log goes to `-inf` via direct `log(1/(1+exp(-x)))`. The ORPO odds-ratio term has an analogous issue: converting a sequence log-probability `log p` into `log(1-p)` via `log(1 - exp(log p))` loses precision (or returns `-inf`/`nan`) when `log p` is very close to 0.
- **Solutions**: `dpo_loss` uses `-np.logaddexp(0.0, -x)` for `log(sigmoid(x))`, which is the standard numerically stable log-sigmoid (avoids computing `exp` of a possibly large positive number). `orpo_loss` uses a two-branch stable `log1mexp` (Mächler 2012 formulation: `log(-expm1(x))` for `x` close to 0, `log1p(-exp(x))` otherwise) to compute `log(1-p)` from `log p` without cancellation error. Both were verified finite on extreme inputs in `tests/test_losses.py` (`test_dpo_loss_is_finite_for_extreme_logprobs`, `test_orpo_loss_is_finite_for_extreme_logprobs`), e.g. `dpo_loss` with log-probs of `-1e-8` vs `-50.0` returns a finite `50.35` rather than `inf`/`nan`, and `orpo_loss` on a near-zero chosen log-prob (`-1e-6`) vs a very negative rejected log-prob (`-30.0`) returns a finite `0.1`. One edge case slipped through the first pass: `chosen_logps`/`rejected_logps` exactly equal to `0.0` (i.e. `p == 1.0`) sent `log1mexp(0.0)` to `-inf` on both branches of the piecewise formula, which numpy's eager `np.where` evaluates unconditionally — this printed `RuntimeWarning: divide by zero encountered in log`/`log1p` even though the final loss happened to still come out finite. Fixed by `np.clip`-ing both log-probability inputs to `[-30.0, -1e-7]` before computing the odds ratio (`tests/test_losses.py::test_orpo_loss_handles_logp_zero_without_warning_or_nan` asserts this now raises no `RuntimeWarning`).

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | `95.83%` |
| Final Loss (Mock) | `0.635` (DPO, epoch 5/5) via `pref-lab train --config configs/local.yaml` -- see note below. |

Pasted from `outputs/metrics.json` (git-ignored; regenerate with `make run-eval`):
```json
{
  "pairwise_accuracy": 0.9583333333333334
}
```

**Extension implemented: mock CPU trainer.** `PreferenceTrainer.train()` (`src/preference_lab/trainers.py`) is now implemented as the "mock trainer on CPU" option from the lab's extension list, not the TRL-backed one (no torch/GPU available in this environment). It does **not** train a real neural network: each example's `chosen`/`rejected` text gets a deterministic simulated log-probability (`_pseudo_logprob`, a SHA-256-derived value combined with a length penalty), and each epoch nudges the simulated *policy* log-probs toward preferring `chosen` a little more while the simulated *reference* log-probs stay fixed -- a stand-in for what a real gradient step would do. The already-implemented `dpo_loss`/`orpo_loss` is called on this simulated data every batch, so the loop genuinely exercises batching, epoch iteration, and loss logging to `output_dir/train_log.json`, and the loss curve behaves the way a real one should (monotonically decreasing, verified in `tests/test_trainers.py`):
```json
{
  "method": "dpo",
  "num_epochs": 5,
  "batch_size": 2,
  "num_examples": 19,
  "loss_history": [0.6931, 0.6783, 0.6636, 0.6492, 0.6349]
}
```
Swapping in a real policy/reference model (e.g. wrapping `trl.DPOTrainer`/`trl.ORPOTrainer` after `pip install -e '.[dev,train]'`) would only require replacing `_pseudo_logprob` and the per-epoch nudge with real forward passes -- the batching/logging/loss-selection plumbing around it would not need to change.

### Qualitative Review
- **Prompt**: `Explain what a hash table is and why it offers fast average-case lookups.`
- **Chosen Response**: `A hash table stores key-value pairs by applying a hash function to the key to compute an index into an underlying array... giving O(1) average-case time for get/put/delete, with worst-case O(n) if many keys collide.`
- **Rejected Response**: `A hash table is a list that stores things in order and is fast because computers are fast.`
- **Model Preference**: `Correct` — the CLI's deterministic `score_response` scorer (relevance + lexical diversity + length, see `src/preference_lab/evaluate.py`) rates the chosen response higher because it echoes more prompt keywords and is more detailed, matching the intended preference.

## 4. Discussion & Failure Modes

- **What went well?**: The pairwise scorer correctly ranked chosen over rejected for 23/24 examples without any model calls, purely from lexical relevance/diversity/length heuristics — a reasonable CPU-only proxy for "did the model prefer the better response" while a real policy model is not being trained in this lab session.
- **Observed Bias**: The scorer is length- and keyword-overlap-sensitive (`score_response` caps detail credit at 30 tokens but still rewards verbosity up to that point), so it would systematically over-rate a longer, keyword-stuffed answer that never got trained on real preference signal. It also has no way to detect a fluent but factually wrong "chosen" response, since it only measures surface features, not correctness — the one example it failed on had a rejected response similar in length/relevance to the chosen one.
- **Safety**: `docs/regression_prompts.md`'s four prompts (high-risk medical advice, strict-length summary, admitting uncertainty, troubleshooting with missing context) were used as the template for several of the hand-authored pairs in `data/synthetic_preferences.jsonl` (e.g. the chest-pain / bleach-ammonia safety pairs, the stock-price uncertainty pair, and the two "ask for missing context" troubleshooting pairs). No trained policy was run against them in this session since training is out of scope for this pass (`PreferenceTrainer.train()` is unimplemented); running these regression prompts before/after a real training run is the natural next step once a trainer backend is wired up.

### Depth check (lab guide 5.2)
- **Removing the reference anchor**: setting `ref_chosen_logps == ref_rejected_logps` (e.g. both `-1.0`) does *not* crash `dpo_loss` — it still returns a finite number (`0.327` vs. `0.387` with a real reference in one test case). That's the point of the exercise: the loss keeps computing, but `beta * (policy_logratio - ref_logratio)` collapses to `beta * policy_logratio`, i.e. plain preference-ranking on the policy alone with no penalty for how far it has drifted from the reference. The reference term is what stops the policy from degenerating into "always maximize chosen minus rejected" regardless of fluency/behavior elsewhere — losing it removes that anchor silently, not loudly, which is exactly why it's easy to miss in practice.
- **Reproducibility**: ran `pref-lab evaluate --config configs/local.yaml` twice (with `outputs/` cleared between runs) — both produced byte-identical `pairwise_accuracy: 0.9583333333333334`. This is expected: `score_response` is a pure function of `(prompt, response)` text with no RNG calls anywhere in the `validate`/`evaluate` path (the only randomness in the codebase, `random.Random(seed)` in `split_by_prompt`, is explicitly seeded and isn't on the evaluate path at all).
- **Length bias in the scorer**: yes, measurably. Two equally-correct answers to "What is the time complexity of binary search?" — `"O(log n)."` vs. `"The time complexity of binary search is O(log n), since each step halves the search space."` — scored `0.330` and `0.755` respectively purely because the longer one both echoes more prompt keywords and gets more of the length/detail credit (capped at 30 tokens). A control check confirmed it's not *purely* length, though: 60 tokens of repeated filler unrelated to the prompt scored `0.305`, about the same as the short-but-correct answer, because `diversity` (unique-token ratio) and `overlap` (prompt-keyword relevance) collapse for filler text. So the scorer rewards **relevant** length, not length alone — but on this dataset, where `chosen` responses are consistently longer *and* more relevant than `rejected`, `pairwise_accuracy` is partly measuring "is the chosen response more detailed," which is correlated with but not identical to "is it higher quality." A real logprob-based scorer (the `TODO` this heuristic stands in for) would not have this confound.
