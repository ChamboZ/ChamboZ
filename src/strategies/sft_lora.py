from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.adapters.boltz_rcsb_adapter import BoltzRCSBAdapter, TinySyntheticAdapter
from src.data.collate import collate_batch
from src.models.boltz2_wrapper import Boltz2Wrapper, BoltzWrapperConfig
from src.models.peft.lora import LoRAConfig, apply_lora, mark_only_lora_trainable


class SFTLoRAStrategy:
    """Supervised fine-tuning strategy using LoRA adapters."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config["training"]["batch_size"]
        self.collate_fn = collate_batch
        self.dataset = None
        self.model: nn.Module

    def setup(self, device: torch.device) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model_cfg = BoltzWrapperConfig(
            vocab_size=self.config["model"]["vocab_size"],
            hidden_dim=self.config["model"]["hidden_dim"],
            dry_run=self.config["training"]["dry_run"],
        )
        model = Boltz2Wrapper(model_cfg)
        lora_cfg = LoRAConfig(**self.config["lora"])
        model = apply_lora(model, lora_cfg)
        mark_only_lora_trainable(model)
        model.to(device)
        self.model = model

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config["training"]["lr"])
        return model, optimizer

    def get_dataset(self):
        if self.dataset is None:
            data_cfg = self.config["data"]
            targets = Path(data_cfg["targets_dir"])
            msa = Path(data_cfg["msa_dir"])
            symmetry = Path(data_cfg["symmetry_path"])
            if data_cfg.get("use_synthetic"):
                self.dataset = TinySyntheticAdapter(size=data_cfg["synthetic_size"], seq_len=64)
            else:
                self.dataset = BoltzRCSBAdapter(
                    targets_dir=targets,
                    msa_dir=msa,
                    symmetry_path=symmetry,
                    max_items=data_cfg["max_items"],
                )
        return self.dataset

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        batch["inputs"] = {k: v.to(device) for k, v in batch["inputs"].items()}
        batch["targets"] = {k: v.to(device) for k, v in batch["targets"].items()}
        return batch

    def compute_loss(self, batch: Dict[str, Any], step: int):
        model_outputs = self.model(batch["inputs"])

        loss_type = self.config["training"]["loss_type"]
        if loss_type == "cross_entropy":
            logits = model_outputs["logits"].reshape(-1, model_outputs["logits"].shape[-1])
            labels = batch["targets"]["labels"].reshape(-1)
            loss = F.cross_entropy(logits, labels)
        else:
            # TODO: replace with true Boltz regression objective
            pred = model_outputs["pred"].squeeze(-1)
            target = batch["targets"]["regression"].squeeze(-1)
            loss = F.mse_loss(pred, target)

        metrics = {"loss_type": loss_type}
        return loss, metrics
