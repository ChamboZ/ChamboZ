import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(log_path: Path, rank: int = 0) -> logging.Logger:
    """Create a logger that writes to console and JSONL file."""
    logger = logging.getLogger(f"trainer_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def maybe_log_metrics(
    logger: logging.Logger,
    jsonl_path: Path,
    step: int,
    metrics: Dict[str, Any],
    every: int = 1,
    rank: int = 0,
) -> None:
    if rank != 0:
        return
    if step % every == 0:
        logger.info("step=%s metrics=%s", step, metrics)
        record = {"step": step, **metrics}
        log_jsonl(jsonl_path, record)
