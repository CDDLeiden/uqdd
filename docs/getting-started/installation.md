---
title: Installation
tags:
  - installation
  - environment
  - getting-started
---

# Installation

We recommend using Conda for dependency management.

## Prerequisites

- Python >= 3.8
- PyTorch >= 2.0 (with CUDA if available)
- RDKit
- Weights & Biases
- scikit-learn, numpy, pandas, seaborn, matplotlib

## Setup

```bash
# Linux
conda env create --file=environments/environment_linux.yml
conda activate uqdd-env

# Windows
# If the above fails, try the _conda variant
conda env create --file=environments/environment_windows.yml
conda activate uqdd-env
```

Notes:

- If environments/environment_{OS}.yml gives an error, use the files with the `_conda` suffix (e.g., `environments/environment_linux_conda.yml`).
- For GPU support, install PyTorch with the appropriate CUDA version per https://pytorch.org/get-started/locally/.
- If issues arise, open an issue following CONTRIBUTING.md.
