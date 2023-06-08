# import os
# import sys
#
# from . import run_baseline
# from .. import DATA_DIR, LOGS_DIR, SCRIPTS_DIR, MODELS_DIR
#
# # Append the parent folder path to the system path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# __all__ = ['DATA_DIR', 'LOGS_DIR', 'SCRIPTS_DIR', 'MODELS_DIR']

from uqdd.run_scripts.run_baseline import *

__all__ = ['run_baseline']
