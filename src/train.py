from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.engine.trainer import Trainer, TrainerConfig
from src.strategies.grpo import GRPOStrategy
from src.strategies.sft_lora import SFTLoRAStrategy


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def build_strategy(config: Dict[str, Any]):
    method = config["method"]
    if method == "sft_lora":
        return SFTLoRAStrategy(config)
    if method == "grpo":
        return GRPOStrategy(config)
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", type=Path, default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    if args.override:
        override = load_config(args.override)
        config = merge_configs(base, override)
    else:
        config = base

    output_dir = Path(config["output_dir"]) / config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_cfg = TrainerConfig(
        output_dir=output_dir,
        log_every=config["training"]["log_every"],
        save_every=config["training"]["save_every"],
        max_steps=config["training"]["max_steps"],
        grad_accum_steps=config["training"]["grad_accum_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        use_amp=config["training"]["use_amp"],
        seed=config["seed"],
        resume_path=Path(config["training"]["resume_path"]) if config["training"]["resume_path"] else None,
    )

    strategy = build_strategy(config)
    trainer = Trainer(strategy, trainer_cfg)
    summary = trainer.train()

    summary_path = output_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
