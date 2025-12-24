from __future__ import annotations

from typing import Any, Dict, List

import torch


def _pad_tensor(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(t.shape[0] for t in tensors)
    padded = []
    for t in tensors:
        if t.dim() == 1:
            pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
            padded.append(torch.cat([t, pad], dim=0))
        elif t.dim() == 2:
            pad = torch.full(
                (max_len - t.shape[0], t.shape[1]), pad_value, dtype=t.dtype
            )
            padded.append(torch.cat([t, pad], dim=0))
        else:
            padded.append(t)
    return torch.stack(padded, dim=0)


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for unified batch schema."""
    inputs: Dict[str, List[torch.Tensor]] = {}
    targets: Dict[str, List[torch.Tensor]] = {}
    meta: Dict[str, List[Any]] = {}

    for item in batch:
        for key, value in item["inputs"].items():
            inputs.setdefault(key, []).append(value)
        for key, value in item["targets"].items():
            targets.setdefault(key, []).append(value)
        for key, value in item["meta"].items():
            meta.setdefault(key, []).append(value)

    inputs_t = {k: _pad_tensor(v) for k, v in inputs.items()}
    targets_t = {k: _pad_tensor(v) if isinstance(v[0], torch.Tensor) else v for k, v in targets.items()}
    meta_t = meta

    return {"inputs": inputs_t, "targets": targets_t, "meta": meta_t}
