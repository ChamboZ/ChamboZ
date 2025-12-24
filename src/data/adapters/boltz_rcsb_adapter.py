from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class BoltzRCSBAdapter(Dataset):
    """Adapter for Boltz RCSB processed targets/MSA dataset.

    Expected layout:
      data/boltz_rcsb/targets/*
      data/boltz_rcsb/msa/*
      data/boltz_rcsb/symmetry.pkl

    This adapter emits the unified batch schema:
      {
        "inputs": {...},
        "targets": {...},
        "meta": {...},
      }
    """

    def __init__(self, targets_dir: Path, msa_dir: Path, symmetry_path: Path, max_items: int = 0):
        self.targets_dir = targets_dir
        self.msa_dir = msa_dir
        self.symmetry_path = symmetry_path
        self.max_items = max_items
        self.items = self._index_items()

    def _index_items(self) -> List[Path]:
        if self.targets_dir.exists():
            items = sorted(self.targets_dir.glob("*.pt"))
            if not items:
                items = sorted(self.targets_dir.glob("*.pkl"))
            if not items:
                items = sorted(self.targets_dir.glob("*"))
        else:
            items = []
        if self.max_items:
            items = items[: self.max_items]
        return items

    def __len__(self) -> int:
        return max(len(self.items), 1)

    def _load_item(self, path: Path) -> Dict[str, Any]:
        # Placeholder loader; replace with actual Boltz parsed format.
        # This returns synthetic features if actual loading fails.
        try:
            if path.suffix == ".pt":
                data = torch.load(path, map_location="cpu")
            elif path.suffix == ".json":
                data = json.loads(path.read_text())
            else:
                data = {"sequence_length": 128}
        except Exception:
            data = {"sequence_length": 128}
        seq_len = int(data.get("sequence_length", 128))
        return {
            "inputs": {
                "input_ids": torch.randint(0, 32, (seq_len,), dtype=torch.long),
                "features": torch.randn(seq_len, 16),
            },
            "targets": {
                "labels": torch.randint(0, 32, (seq_len,), dtype=torch.long),
                "regression": torch.randn(1),
            },
            "meta": {
                "id": path.stem,
                "seq_len": seq_len,
            },
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.items:
            return self._load_item(Path("synthetic"))
        path = self.items[idx % len(self.items)]
        return self._load_item(path)


class TinySyntheticAdapter(Dataset):
    """Fallback tiny dataset for debugging without Boltz data."""

    def __init__(self, size: int = 128, seq_len: int = 64):
        self.size = size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "inputs": {
                "input_ids": torch.randint(0, 32, (self.seq_len,), dtype=torch.long),
                "features": torch.randn(self.seq_len, 16),
            },
            "targets": {
                "labels": torch.randint(0, 32, (self.seq_len,), dtype=torch.long),
                "regression": torch.randn(1),
            },
            "meta": {"id": f"synthetic_{idx}", "seq_len": self.seq_len},
        }
