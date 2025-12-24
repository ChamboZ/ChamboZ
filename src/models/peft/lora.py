from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.lora_b.weight)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.r > 0:
            result = result + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return result


def _matches(name: str, patterns: List[str]) -> bool:
    return any(p in name for p in patterns)


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    if config.target_modules is None:
        config.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "mlp"]
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(name, config.target_modules):
            parent = model
            *path, child_name = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, child_name, LoRALinear(module, config.r, config.alpha, config.dropout))
    return model


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    model_state.update(state)
    model.load_state_dict(model_state)
