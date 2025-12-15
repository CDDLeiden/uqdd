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
├── .github/
│   └── workflows/
│       └── gh-pages.yml        # GitHub Pages deployment workflow
├── .gitattributes
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── mkdocs.yml
├── environments/                # Environment configuration files
│   ├── environment_linux.yml
│   ├── environment_linux_conda.yml
│   ├── environment_windows.yml
│   └── environment_windows_conda.yml
├── docs/                        # Documentation source files
│   ├── index.md
│   ├── changelog.md
│   ├── tags.md
│   ├── api/
│   ├── assets/
│   ├── getting-started/
│   ├── reference/
│   ├── styles/
│   └── user-guide/
├── uqdd/                        # Source Code Directory
│   ├── __init__.py
│   ├── utils.py                 # Shared utilities
│   ├── utils_chem.py            # Chemical descriptor utilities
│   ├── utils_prot.py            # Protein descriptor utilities
│   ├── config/                  # Configuration files for models and data
│   │   ├── papyrus.json
│   │   ├── pnn.json
│   │   ├── ensemble.json
│   │   ├── mcdropout.json
│   │   ├── evidential.json
│   │   ├── eoe.json
│   │   ├── emc.json
│   │   ├── desc_dim.json
│   │   ├── pnn-sweep.json
│   │   └── evidential-sweep.json
│   ├── data/                    # Scripts and utilities for data processing
│   │   ├── __init__.py
│   │   ├── data_papyrus.py
│   │   ├── utils_data.py
│   │   ├── dataset/
│   │   ├── papyrus_extensive.sh
│   │   └── papyrus_prepare_example.sh
│   ├── models/                  # Model implementations and training scripts
│   │   ├── __init__.py
│   │   ├── model_parser.py
│   │   ├── pnn.py
│   │   ├── ensemble.py
│   │   ├── mcdropout.py
│   │   ├── evidential.py
│   │   ├── eoe.py
│   │   ├── emc.py
│   │   ├── loss.py
│   │   ├── utils_models.py
│   │   ├── utils_train.py
│   │   ├── utils_metrics.py
│   │   ├── ensemble.sh
│   │   ├── mcdropout.sh
│   │   ├── pnn.sh
│   │   ├── evidential.sh
│   │   └── evidential-sweep-screens-slurm.sh
│   ├── metrics/                 # Metrics analysis and assessment
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   ├── constants.py
│   │   ├── reassessment.py
│   │   └── stats.py
│   ├── figures/                 # Generated figures and visualizations
│   ├── logs/                    # Training and processing logs
│   └── __pycache__/
├── notebooks/                   # Jupyter Notebooks for exploratory analysis
│   ├── metrics_analysis.ipynb
│   └── models_reassessment.ipynb
├── scripts/                     # Automation and plotting scripts
│   ├── metrics_analysis.py
│   ├── metrics_stats_significance.py
│   └── model_reassessment.py
├── images/                      # Figures referenced in README/docs
│   ├── 01_uq_models.png
│   ├── 01_uq_models.pdf
│   ├── 03_xc50_barplot_tab10_r.png
│   ├── 03_xc50_barplot_tab10_r.pdf
│   ├── 03_kx_barplot_tab10_r.png
│   └── 03_kx_barplot_tab10_r.pdf
├── results/                     # Results and final outputs
│   ├── final_xc50.csv
│   └── final_kx.csv
├── tests/                       # Unit tests
│   └── test_data_papyrus.py
└── site/                        # Generated documentation site (build output)
```
