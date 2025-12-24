$ErrorActionPreference = 'Stop'

python -m src.train --config configs/base.yaml --override configs/sft_lora.yaml
python -m src.eval.evaluate --run-dir outputs/sft_lora_smoke --method sft --config configs/base.yaml
