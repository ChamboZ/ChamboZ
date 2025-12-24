from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.engine.checkpointing import load_checkpoint, save_checkpoint
from src.engine.logging import maybe_log_metrics, setup_logger


@dataclass
class TrainerConfig:
    output_dir: Path
    log_every: int
    save_every: int
    max_steps: int
    grad_accum_steps: int
    max_grad_norm: float
    use_amp: bool
    seed: int
    resume_path: Optional[Path] = None


class Trainer:
    """Unified trainer engine handling DDP/AMP/grad-accum/checkpoint/logging."""

    def __init__(self, strategy: Any, config: TrainerConfig) -> None:
        self.strategy = strategy
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.ddp = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_distributed()
        self.logger = setup_logger(self.config.output_dir / "train.log", rank=self.rank)
        self.jsonl_path = self.config.output_dir / "metrics.jsonl"

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        self.model, self.optimizer = self.strategy.setup(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.rank])
        self.strategy.model = self.model

        self.train_loader = self._build_dataloader()
        self.start_step = 0
        if self.config.resume_path is not None and self.config.resume_path.exists():
            payload = load_checkpoint(
                self.config.resume_path,
                self._unwrap_model(),
                optimizer=self.optimizer,
                scaler=self.scaler,
            )
            self.start_step = int(payload.get("step", 0))

    def _setup_distributed(self) -> None:
        if not dist.is_available():
            return
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world_size = world_size
            self.ddp = True
            torch.cuda.set_device(self.rank)
            self.device = torch.device("cuda", self.rank)

    def _unwrap_model(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _build_dataloader(self) -> DataLoader:
        dataset = self.strategy.get_dataset()
        if self.ddp:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=self.strategy.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=0,
            collate_fn=self.strategy.collate_fn,
        )

    def train(self) -> Dict[str, Any]:
        step = self.start_step
        self._unwrap_model().train()
        nan_count = 0
        loss_history = []
        data_iter = iter(self.train_loader)

        progress = tqdm(total=self.config.max_steps, disable=self.rank != 0)
        while step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(self.config.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                batch = self.strategy.prepare_batch(batch, self.device)
                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    loss, metrics = self.strategy.compute_loss(batch, step)
                if torch.isnan(loss):
                    nan_count += 1
                    self.logger.error("NaN loss encountered at step %s", step)
                    break
                accum_loss += loss.item()
                self.scaler.scale(loss / self.config.grad_accum_steps).backward()

            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._unwrap_model().parameters(), self.config.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            step += 1
            loss_history.append(accum_loss / self.config.grad_accum_steps)
            metrics = {"loss": loss_history[-1], **metrics}
            maybe_log_metrics(
                self.logger,
                self.jsonl_path,
                step,
                metrics,
                every=self.config.log_every,
                rank=self.rank,
            )
            if step % self.config.save_every == 0 and self.rank == 0:
                save_checkpoint(
                    self.config.output_dir / f"checkpoint_step_{step}.pt",
                    self._unwrap_model(),
                    self.optimizer,
                    self.scaler,
                    step,
                    extra={"metrics": metrics},
                )
            progress.update(1)
        progress.close()

        if self.rank == 0:
            save_checkpoint(
                self.config.output_dir / "checkpoint_last.pt",
                self._unwrap_model(),
                self.optimizer,
                self.scaler,
                step,
                extra={"loss_history": loss_history},
            )
        return {
            "steps_run": step,
            "nan_count": nan_count,
            "loss_history": loss_history,
        }
