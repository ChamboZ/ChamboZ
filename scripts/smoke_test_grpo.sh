#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config configs/base.yaml --override configs/grpo.yaml
python -m src.eval.evaluate --run-dir outputs/grpo_smoke --method grpo --config configs/base.yaml
