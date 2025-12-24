from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.adapters.boltz_rcsb_adapter import BoltzRCSBAdapter, TinySyntheticAdapter
from src.data.collate import collate_batch
from src.models.boltz2_wrapper import Boltz2Wrapper, BoltzWrapperConfig
from src.models.peft.lora import LoRAConfig, apply_lora, mark_only_lora_trainable
from src.rewards.reward_proxy import reward_proxy


class GRPOStrategy:
    """Group-relative policy optimization strategy."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config["training"]["batch_size"]
        self.collate_fn = collate_batch
        self.dataset = None
        self.model: nn.Module
        self.ref_model: nn.Module

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
        self.ref_model = deepcopy(model).eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

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
        outputs = self.model(batch["inputs"])
        logits = outputs["logits"]  # [B, T, V]
        bsz, seq_len, vocab = logits.shape
        k = self.config["grpo"]["num_candidates"]

        dist = torch.distributions.Categorical(logits=logits)
        samples = dist.sample((k,))  # [K, B, T]
        log_probs = dist.log_prob(samples)  # [K, B, T]
        log_prob_sum = log_probs.sum(dim=-1)  # [K, B]

        rewards = reward_proxy(samples, batch)
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        if reward_std == 0.0:
            print("[warning] reward std is zero; check reward proxy")

        advantages = rewards - rewards.mean(dim=0, keepdim=True)
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()

        policy_loss = -(advantages.detach() * log_prob_sum).mean()

        with torch.no_grad():
            ref_outputs = self.ref_model(batch["inputs"])
            ref_logits = ref_outputs["logits"]
        ref_dist = torch.distributions.Categorical(logits=ref_logits)
        ref_log_probs = ref_dist.log_prob(samples).sum(dim=-1)
        kl = (log_prob_sum - ref_log_probs).mean()

        kl_coef = self.config["grpo"]["kl_coef"]
        loss = policy_loss + kl_coef * kl

        metrics = {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "kl_mean": kl.item(),
            "entropy": dist.entropy().mean().item(),
        }
        return loss, metrics
