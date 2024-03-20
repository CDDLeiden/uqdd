from pathlib import Path
from datetime import date
import torch

# Define the global device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(DEVICE))
print(torch.version.cuda) if DEVICE == "cuda" else None

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
WANDB_MODE = "online"  # 'offline'

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
__supervisor__ = "Kajetan Schweighofer"
__contributor__ = "Natalia Dyubankova"
__copyright__ = (
    "Copyright 2023-2024, Johnson & Johnson and Johannes-Kepler Universität Linz"
)
__license__ = (
    "All rights reserved, Johnson & Johnson & Johannes-Kepler Universität Linz"
)
__version__ = "0.0.2"
__maintainer__ = "Bola Khalil"
__email__ = "bkhalil@its.jnj.com"
__status__ = "Development"
