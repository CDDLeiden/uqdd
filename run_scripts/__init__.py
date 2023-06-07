import os
import sys

from . import run_baseline
from .. import DATA_DIR, LOGS_DIR, SCRIPTS_DIR, MODELS_DIR, models

# Append the parent folder path to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ['DATA_DIR', 'LOGS_DIR', 'SCRIPTS_DIR', 'MODELS_DIR']