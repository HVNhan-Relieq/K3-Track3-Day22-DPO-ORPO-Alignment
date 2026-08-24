from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl, split_by_prompt
from .evaluate import pairwise_accuracy, score_response, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")

@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")

@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML run config")],
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    chosen_scores = [score_response(ex.prompt, ex.chosen) for ex in examples]
    rejected_scores = [score_response(ex.prompt, ex.rejected) for ex in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")

@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", help="Path to a YAML run config")],
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    train_examples, _val_examples = split_by_prompt(
        examples, validation_ratio=0.2, seed=cfg.get("seed", 42)
    )
    training_cfg = TrainingConfig(
        method=cfg["training"]["method"],
        beta=cfg["training"].get("beta", 0.1),
        lambda_orpo=cfg["training"].get("lambda_orpo", 0.1),
        max_length=cfg["training"].get("max_length", 512),
        batch_size=cfg["training"].get("batch_size", 2),
    )
    trainer = PreferenceTrainer(training_cfg, train_examples, cfg["paths"]["output_dir"])
    summary = trainer.train()
    print(f"[green]Wrote training log to {cfg['paths']['output_dir']}/train_log.json[/green]")
    print(f"Final loss ({training_cfg.method}): {summary['loss_history'][-1]:.4f}")

if __name__ == "__main__":
    app()
