from __future__ import annotations

from typing import Dict

import torch


def reward_proxy(samples: torch.Tensor, batch: Dict) -> torch.Tensor:
    """Compute a cheap reward proxy for GRPO.

    Reward = -length penalty + token overlap with target labels (if available).
    Returns shape [K, B].
    """
    labels = batch["targets"].get("labels")
    k, batch_size, seq_len = samples.shape
    length_penalty = -0.01 * seq_len
    reward = torch.full((k, batch_size), length_penalty, device=samples.device)

    if labels is not None:
        labels_exp = labels.unsqueeze(0).expand(k, -1, -1)
        overlap = (samples == labels_exp).float().mean(dim=-1)
        reward = reward + overlap
    return reward
