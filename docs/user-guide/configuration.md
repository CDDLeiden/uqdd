# Configuration

Configuration files live under `uqdd/config/` and control data preparation, model defaults, and sweeps.

## Data configuration

- `papyrus.json`: defines data sources, descriptor choices, and splitting strategy for Papyrus++.
    - Typical keys: `activity_type` (xc50|kx), `descriptor_protein`, `descriptor_chemical`, `split_type` (
      random|scaffold|time), `file_ext` (pkl|parquet|csv), and flags (e.g., `sanitize`).

- `desc_dim.json`: maps descriptor names to vector dimensions (e.g., `ecfp2048: 2048`, `ankh-large: <dim>`), used to
  size model inputs correctly.

## Model configuration (defaults)

- `pnn.json`, `ensemble.json`, `mcdropout.json`, `evidential.json`, `eoe.json`, `emc.json`:
    - Set default hyperparameters per model family: architecture (layers, hidden sizes), dropout, learning rate, epochs,
      batch size, and uncertainty-specific knobs (e.g., number of MC samples, ensemble size).
    - These defaults are read by training scripts and can be overridden via CLI (see Models & Training).

## Sweep configurations

- `pnn-sweep.json`, `evidential-sweep.json`:
    - Define parameter grids or distributions for hyperparameter searches (often used with Weights & Biases sweeps),
      e.g., varying learning rates, widths, and regularization.

## How configuration is used

- The main parser `uqdd/models/model_parser.py` reads CLI flags and merges them with the chosen config JSON to produce
  the final run configuration.
- Data preparation scripts reference `papyrus.json` and `desc_dim.json` to validate descriptor dimensions and save
  outputs consistently.

## Customizing experiments

- Start with a base model JSON (e.g., `evidential.json`), then override via CLI flags for quick experiments:
    - Example: `--epochs 100 --batch_size 512 --lr 3e-4`
- For consistent runs across teams, keep custom JSONs under `uqdd/config/` and refer to them with a flag or by adjusting
  the default in `model_parser.py`.

## Best practices

- Keep descriptor names consistent across data and model configs.
- Update `desc_dim.json` when adding new embeddings to avoid shape mismatches.
- Version control any changes to config files and document rationale in commit messages.
