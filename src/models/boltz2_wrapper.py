from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class BoltzWrapperConfig:
    vocab_size: int = 32
    hidden_dim: int = 128
    dry_run: bool = False


class DummyBoltzModel(nn.Module):
    """Fallback model when Boltz-2 is unavailable."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        pooled = hidden.mean(dim=1)
        regression = self.reg_head(pooled)
        return {"logits": logits, "pred": regression}


class Boltz2Wrapper(nn.Module):
    """Wrapper for Boltz-2 model with import fallbacks.

    Update the import path below if your local Boltz-2 repository uses a different module name.
    """

    def __init__(self, config: BoltzWrapperConfig):
        super().__init__()
        self.config = config
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        try:
            # Attempt typical import path. Adjust this line if needed.
            from boltz.models.boltz2 import Boltz2Model  # type: ignore
            print("[MODEL] Using Boltz2Model from boltz.models.boltz2")
            return Boltz2Model()
        except Exception as e:
            print(f"[MODEL] Falling back to DummyBoltzModel because boltz import failed: {e}")
            return DummyBoltzModel(self.config.vocab_size, self.config.hidden_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = inputs.get("input_ids")
        features = inputs.get("features")
        outputs = self.model(input_ids=input_ids, features=features)

        if self.config.dry_run:
            print("[dry-run] input_ids", input_ids.shape)
            if features is not None:
                print("[dry-run] features", features.shape)
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"[dry-run] output {key}", value.shape)
        return outputs
