# Models & Training

Main entry point: `uqdd/models/model_parser.py`.

Common flags:
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

Models:
- PNN: probabilistic neural network baseline
- Ensemble: deep ensemble; set `--ensemble_size`
- MC-Dropout: stochastic forward passes; set `--num_mc_samples`
- Evidential: evidential regression
- EOE: ensemble of evidential networks; set `--ensemble_size`
- EMC: evidential MC-Dropout; set `--num_mc_samples`

Examples are in Quickstart.
