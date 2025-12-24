# Boltz-2 Unified Training Framework

Minimal, production-clean training framework supporting:
- **SFT with LoRA adapters**
- **GRPO (Group-Relative Policy Optimization)**

This repository is designed to wrap the official Boltz-2 model when available, and falls back to a tiny dummy model for smoke tests and debugging.

## Quickstart

```cmd
python -m pip install torch pyyaml tqdm
```

### Download Boltz RCSB data (Windows Command Prompt)

```cmd
scripts\download_boltz_rcsb.bat
```

### Smoke tests (Windows Command Prompt)

```cmd
scripts\smoke_test_sft.bat
scripts\smoke_test_grpo.bat
```

Smoke tests write acceptance reports:
- `reports/acceptance_sft.json`
- `reports/acceptance_grpo.json`

Acceptance thresholds are configured in `configs/base.yaml` under `acceptance:`:
- `max_nan`
- `max_loss_slope`
- `min_reward_std` (GRPO only)

## Structure

```
src/
  engine/          # Trainer engine, checkpointing, logging
  data/            # Dataset adapters + collate
  models/          # Boltz-2 wrapper + LoRA
  strategies/      # SFT and GRPO strategies
  rewards/         # Reward proxy
  eval/            # Acceptance report generation
configs/           # YAML configs
scripts/           # Download + smoke tests
```

## Configuration

- `configs/base.yaml` defines common settings (output dir, model params, training defaults).
- `configs/sft_lora.yaml` overrides for SFT.
- `configs/grpo.yaml` overrides for GRPO.

`configs/base.yaml` includes data paths for:

```
data/boltz_rcsb/targets
  data/boltz_rcsb/msa
  data/boltz_rcsb/symmetry.pkl
```

## Boltz-2 model integration

`src/models/boltz2_wrapper.py` attempts to import the Boltz-2 model:

```python
from boltz.models.boltz2 import Boltz2Model
```

If your local Boltz-2 repo uses a different import path, **edit that one line**. The wrapper will map outputs to `{"logits": ..., "pred": ...}`.

### Dry-run wiring
Set `training.dry_run: true` in a config to print tensor shapes for inputs/outputs and confirm wiring.

## Unified batch schema

Every dataset adapter emits:

```python
{
  "inputs": dict(tensors),
  "targets": dict(tensors),
  "meta": dict(metadata),
}
```

Strategies only consume this schema and **never access raw dataset objects**.

## Notes

- LoRA adapters are applied to module name patterns in `configs/base.yaml`.
- SFT uses a placeholder loss (cross-entropy or MSE). Replace TODOs in `src/strategies/sft_lora.py` with true Boltz objectives.
- GRPO uses a cheap reward proxy in `src/rewards/reward_proxy.py` and logs reward stats + KL mean.
