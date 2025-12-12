"""UQDD package initialization.

Provides global constants, paths, and configuration defaults used across the
library. Importing this package should be lightweight and avoid heavy side
effects. Logging is used instead of printing.
"""

import os
from datetime import date
from pathlib import Path
import logging

import torch
import wandb

# Avoid hard side effects when importing; keep env var as it affects CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Add requirement for wandb core
# wandb.sdk.require("core")
# wandb.require("core")

# Module logger
_logger = logging.getLogger(__name__)

# Define the global device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_logger.debug("UQDD Device: %s", DEVICE)
# _logger.debug(torch.version.cuda) if DEVICE == "cuda" else None

# Define the base directory as the parent of this file
BASE_DIR = Path(__file__).parent

# Define paths using pathlib
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
SCRIPTS_DIR = BASE_DIR / "run_scripts"
MODELS_DIR = BASE_DIR / "models"
FIGS_DIR = BASE_DIR / "figures"

# Date variable
TODAY = date.today().strftime("%Y%m%d")

WANDB_DIR = LOGS_DIR / "wandb"
WANDB_MODE = "online"  # or 'offline'

# Create directories if they do not exist
for dir in [
    DATA_DIR,
    DATASET_DIR,
    LOGS_DIR,
    CONFIG_DIR,
    SCRIPTS_DIR,
    MODELS_DIR,
    FIGS_DIR,
    WANDB_DIR,
]:
    dir.mkdir(parents=True, exist_ok=True)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DATASET_DIR",
    "LOGS_DIR",
    "CONFIG_DIR",
    "SCRIPTS_DIR",
    "MODELS_DIR",
    "WANDB_MODE",
    "WANDB_DIR",
    "FIGS_DIR",
    "TODAY",
    "DEVICE",
]

__author__ = "Bola Khalil"
__contributors__ = (
    "Kajetan Schweighofer, Natalia Dyubankova,"
    "Gerard van Westen*, Herman van Vlijmen*"
)
__copyright__ = "Copyright 2023-2025, Johnson & Johnson, Leiden University, Johannes-Kepler Universität Linz, "
__license__ = "All rights reserved, Johnson & Johnson, Leiden University, Johannes-Kepler Universität Linz"
__version__ = "0.0.6"
__maintainer__ = "Bola Khalil"
__email__ = "b.a.a.khalil@lacdr.leidenuniv.nl"
__status__ = "Development"
