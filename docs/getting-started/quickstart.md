---
title: Quickstart
tags:
  - quickstart
  - getting-started
---

# Quickstart

1. Prepare data (Papyrus++)

```bash
python uqdd/data/data_papyrus.py \
  --activity xc50 \
  --descriptor-protein ankh-large \
  --descriptor-chemical ecfp2048 \
  --split-type time \
  --n-targets -1 \
  --file-ext pkl \
  --sanitize \
  --verbose
```

Outputs: preprocessed splits under `data/` in the chosen format.

2. Train models

- Baseline (PNN):
```bash
python uqdd/models/model_parser.py --model pnn --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name pnn-test
```

- Deep Ensemble:
```bash
python uqdd/models/model_parser.py --model ensemble --ensemble_size 10 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name ensemble-test
```

- MC-Dropout:
```bash
python uqdd/models/model_parser.py --model mcdropout --num_mc_samples 100 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name mcdp-test
```

- Evidential:
```bash
python uqdd/models/model_parser.py --model evidential --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name evidential-test
```

- Ensemble of Evidential (EOE):
```bash
python uqdd/models/model_parser.py --model eoe --ensemble_size 10 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name eoe-test
```

- Evidential MC-Dropout (EMC):
```bash
python uqdd/models/model_parser.py --model emc --num_mc_samples 100 --data_name papyrus --n_targets -1 --activity_type xc50 --descriptor_protein ankh-large --descriptor_chemical ecfp2048 --split_type random --ext pkl --task_type regression --wandb_project_name emc-test
```

Tips:
- Use `--seed`, `--epochs`, `--batch_size`, and `--lr` to control training.
- Set `--device cuda` to train on GPU.
- Logs can be sent to Weights & Biases via `--wandb_project_name`.
