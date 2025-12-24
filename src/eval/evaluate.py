from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.engine.checkpointing import load_checkpoint
from src.models.boltz2_wrapper import Boltz2Wrapper, BoltzWrapperConfig
from src.models.peft.lora import LoRAConfig, apply_lora


def load_metrics(path: Path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        records.append(json.loads(line))
    return records


def compute_loss_slope(losses):
    if len(losses) < 2:
        return 0.0
    return (losses[-1] - losses[0]) / max(1, len(losses) - 1)


def evaluate(run_dir: Path, method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    summary = json.loads((run_dir / "train_summary.json").read_text())
    metrics = load_metrics(run_dir / "metrics.jsonl")
    losses = [m.get("loss", 0.0) for m in metrics]
    loss_start = sum(losses[:5]) / max(1, min(5, len(losses)))
    loss_end = sum(losses[-5:]) / max(1, min(5, len(losses)))
    loss_slope = compute_loss_slope(losses)

    checkpoint_path = run_dir / "checkpoint_last.pt"
    checkpoint_save_ok = checkpoint_path.exists()
    checkpoint_load_ok = False
    reload_loss_match = False

    if checkpoint_save_ok:
        model_cfg = BoltzWrapperConfig(
            vocab_size=config["model"]["vocab_size"],
            hidden_dim=config["model"]["hidden_dim"],
            dry_run=False,
        )
        model = Boltz2Wrapper(model_cfg)
        lora_cfg = LoRAConfig(**config["lora"])
        model = apply_lora(model, lora_cfg)
        payload = load_checkpoint(checkpoint_path, model)
        checkpoint_load_ok = True
        reload_loss_match = payload.get("extra", {}).get("loss_history") is not None

    report = {
        "steps_run": summary.get("steps_run", 0),
        "nan_count": summary.get("nan_count", 0),
        "loss_start_mean": loss_start,
        "loss_end_mean": loss_end,
        "loss_slope": loss_slope,
        "checkpoint_save_ok": checkpoint_save_ok,
        "checkpoint_load_ok": checkpoint_load_ok,
        "reload_loss_match": reload_loss_match,
    }

    if method == "grpo":
        reward_mean = sum(m.get("reward_mean", 0.0) for m in metrics) / max(1, len(metrics))
        reward_std = sum(m.get("reward_std", 0.0) for m in metrics) / max(1, len(metrics))
        kl_mean = sum(m.get("kl_mean", 0.0) for m in metrics) / max(1, len(metrics))
        advantage_std = sum(m.get("adv_std", 0.0) for m in metrics) / max(1, len(metrics))
        report.update(
            {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "kl_mean": kl_mean,
                "advantage_std": advantage_std,
            }
        )

    thresholds = config["acceptance"]
    report["pass"] = (
        report["nan_count"] <= thresholds["max_nan"]
        and report["loss_slope"] <= thresholds["max_loss_slope"]
    )
    if method == "grpo":
        report["pass"] = report["pass"] and report["reward_std"] >= thresholds["min_reward_std"]

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["sft", "grpo"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text()) if args.config.suffix == ".json" else None
    if config is None:
        import yaml

        config = yaml.safe_load(args.config.read_text())

    report = evaluate(args.run_dir, args.method, config)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"acceptance_{args.method}.json"
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
