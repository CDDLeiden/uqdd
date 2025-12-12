---
title: Data Pipeline
tags:
  - data
  - preprocessing
---

# Data Pipeline

Use `uqdd/data/data_papyrus.py` to prepare Papyrus++ datasets.

## Key arguments

- `--activity`: xc50 | kx
- `--descriptor-protein`: ankh-large | ankh-small | unirep
- `--descriptor-chemical`: ecfp2048 | ecfp1024
- `--split-type`: random | scaffold | time
- `--n-targets`: -1 for all targets, or a positive integer
- `--file-ext`: pkl | parquet | csv
- `--sanitize`: clean invalid entries
- `--verbose`: print progress

## Example

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

## Outputs

- Preprocessed files stored under `data/`, organized by activity/split.
- Descriptor choices should match dimensions in `uqdd/config/desc_dim.json`.

## Auxiliary scripts

- `uqdd/data/papyrus_prepare_example.sh`: template for preparing a small example subset.
- `uqdd/data/papyrus_extensive.sh`: comprehensive data preparation workflow for large runs.

## Integration notes

- Downstream training scripts (`uqdd/models/model_parser.py`) expect consistent file extensions and split naming.
- Time-based splits better reflect prospective evaluation, while scaffold splits assess generalization across
  chemotypes.
