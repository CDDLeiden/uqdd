---
title: Models & Training
tags:
  - models
  - training
  - uncertainty
---

# Models & Training

Main entry point: `uqdd/models/model_parser.py`.

## Common flags

- `--model`: pnn | ensemble | mcdropout | evidential | eoe | emc
- `--data_name`: papyrus
- `--activity_type`: xc50 | kx
- `--descriptor_protein`: ankh-large | ankh-small | unirep
- `--descriptor_chemical`: ecfp2048 | ecfp1024
- `--split_type`: random | scaffold | time
- `--ext`: pkl | parquet | csv
- `--task_type`: regression | classification
- `--device`: cpu | cuda
- `--seed`, `--epochs`, `--batch_size`, `--lr`
- Logging: `--wandb_project_name`

## Models

- PNN: probabilistic neural network baseline (`uqdd/models/pnn.py`)
- Ensemble: deep ensemble (`uqdd/models/ensemble.py`); set `--ensemble_size`
- MC-Dropout: stochastic forward passes (`uqdd/models/mcdropout.py`); set `--num_mc_samples`
- Evidential: evidential regression (`uqdd/models/evidential.py`)
- EOE: ensemble of evidential networks (`uqdd/models/eoe.py`); set `--ensemble_size`
- EMC: evidential MC-Dropout (`uqdd/models/emc.py`); set `--num_mc_samples`

## Losses & utilities

- Losses: `uqdd/models/loss.py`
- Training loop helpers: `uqdd/models/utils_train.py`
- Model building helpers: `uqdd/models/utils_models.py`
- Metrics: `uqdd/models/utils_metrics.py`

## Shell helpers

- `pnn.sh`, `ensemble.sh`, `mcdropout.sh`, `evidential.sh`: example invocations for batch runs.
- `evidential-sweep-screens-slurm.sh`: example sweep script (screen/Slurm).

## Guidance

- For EOE, prefer moderate ensemble sizes (e.g., 10) for a trade-off between diversity and compute.
- For EMC, balance `--num_mc_samples` (e.g., 50–100) to capture epistemic uncertainty without excessive runtime.
- Use `--seed` to ensure reproducibility; log to W&B for experiment tracking.
