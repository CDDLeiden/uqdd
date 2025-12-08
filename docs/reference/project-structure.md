---
title: Project Structure
tags:
  - reference
  - project-structure
---

# Project Structure

The project is organized as follows:

```
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── uqdd/  # Source Code Directory
│   ├── __init__.py
│   ├── utils.py           # Shared utilities
│   ├── utils_chem.py      # Chemical descriptor utilities
│   ├── utils_prot.py      # Protein descriptor utilities
│   ├── config/            # Configuration files for models and data
│   │   ├── papyrus.json
│   │   ├── pnn.json
│   │   ├── ensemble.json
│   │   ├── mcdropout.json
│   │   ├── evidential.json
│   │   ├── eoe.json
│   │   ├── emc.json
│   │   ├── pnn-sweep.json
│   │   └── evidential-sweep.json
│   ├── data/              # Scripts and utilities for data processing
│   │   ├── __init__.py
│   │   ├── data_papyrus.py
│   │   └── utils_data.py
│   └── models/            # Model implementations and training scripts
│       ├── __init__.py
│       ├── model_parser.py
│       ├── pnn.py
│       ├── ensemble.py
│       ├── mcdropout.py
│       ├── evidential.py
│       ├── eoe.py
│       ├── emc.py
│       ├── loss.py
│       ├── utils_models.py
│       ├── utils_train.py
│       └── utils_metrics.py
├── notebooks/  # Jupyter Notebooks for exploratory analysis and results visualization
│   ├── metrics_analysis.ipynb
│   ├── metrics_analysis-kx.ipynb
│   └── models_reassessment.ipynb
├── scripts/    # Automation and plotting scripts reflecting notebook analyses
│   ├── metrics_analysis.py
│   └── model_reassessment.py
├── images/     # Figures referenced in README/docs
│   ├── 01_uq_models.png
│   ├── 03_xc50_barplot_tab10_r.png
│   └── 03_kx_barplot_tab10_r.png
├── results_revision/
│   ├── final_xc50.csv
│   └── final_kx.csv
└── .github/workflows/gh-pages.yml # GitHub Pages deployment workflow
```
